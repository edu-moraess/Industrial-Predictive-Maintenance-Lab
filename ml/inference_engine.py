"""
Central ML inference pipeline for the Industrial Predictive Maintenance Lab.

Single source of truth for:
  telemetry -> features -> anomaly -> failure classification -> health -> RUL -> advisory

Used by FastAPI and Streamlit so both surfaces report the same results.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector
from ml.failure_classifier import FailureClassifier
from ml.health_score import HealthScoreCalculator
from ml.rul import RULEstimator


@dataclass
class InferenceResult:
    machine_id: str
    timestamp: str
    health_score: float
    risk_level: str
    is_anomaly: bool
    anomaly_score: float
    failure_mode: str
    failure_probabilities: Dict[str, float]
    failure_probability: float
    rul_hours: float
    maintenance_recommendation: str
    model_status: str
    ground_truth_failure: Optional[str] = None
    prediction_correct: Optional[bool] = None
    inference_ms: float = 0.0
    features_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineStatus:
    ready: bool
    isolation_forest: str
    random_forest: str
    feature_pipeline: str
    training_samples: int
    n_features_anomaly: int
    n_features_classifier: int
    last_training: Optional[str]
    train_accuracy: Optional[float]
    model_version: str = "2.0.0-synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _recommendation(risk: str, failure: str, is_anomaly: bool) -> str:
    if risk in ("FAILURE RISK",) or (is_anomaly and risk == "CRITICAL"):
        return f"Immediate inspection recommended for suspected {failure}."
    if risk == "CRITICAL":
        return f"Priority maintenance inspection: pattern consistent with {failure}."
    if risk == "WARNING":
        return "Monitor vibration and temperature trends; schedule inspection."
    if risk == "NORMAL":
        return "Continue routine monitoring."
    return "No immediate maintenance action required."


class InferenceEngine:
    """Shared inference service. Bootstrap once; call predict() for each cycle."""

    def __init__(self) -> None:
        self.anomaly_detector = AnomalyDetector()
        self.classifier = FailureClassifier()
        self.training_samples = 0
        self.last_training: Optional[str] = None
        self.train_accuracy: Optional[float] = None
        self.last_metrics: Dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        return bool(self.anomaly_detector.is_fitted and self.classifier.is_fitted)

    def bootstrap(self, training_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train both models on synthetic labeled telemetry rows."""
        if not training_rows:
            raise ValueError("bootstrap requires non-empty training_rows")

        df = FeatureEngineer.process_telemetry(training_rows)
        if df.empty:
            raise ValueError("feature engineering produced empty frame")

        self.anomaly_detector.train(df)
        metrics = self.classifier.train(df)
        self.training_samples = len(df)
        self.last_training = datetime.now(timezone.utc).isoformat()
        self.train_accuracy = metrics.get("accuracy")
        self.last_metrics = metrics
        return metrics

    def bootstrap_from_generator(self, hours: int = 6, frequency_minutes: int = 5) -> Dict[str, Any]:
        """Generate synthetic history via DataGenerator and train."""
        from database.repository import DatabaseRepository
        from simulator.generator import DataGenerator

        repo = DatabaseRepository()
        machine_id = "INIT_TRAIN_MACHINE"
        DataGenerator(repo).generate_historical_dataset(
            machine_id, hours=hours, frequency_minutes=frequency_minutes
        )
        history = repo.get_historical_readings(machine_id, limit=500)
        return self.bootstrap(history)

    def status(self) -> EngineStatus:
        return EngineStatus(
            ready=self.is_ready,
            isolation_forest="READY" if self.anomaly_detector.is_fitted else "NOT READY",
            random_forest="READY" if self.classifier.is_fitted else "NOT READY",
            feature_pipeline="READY",
            training_samples=self.training_samples,
            n_features_anomaly=len(self.anomaly_detector.FEATURE_COLUMNS),
            n_features_classifier=len(self.classifier.FEATURE_COLUMNS),
            last_training=self.last_training,
            train_accuracy=self.train_accuracy,
        )

    def predict(
        self,
        readings: List[Dict[str, Any]],
        machine_id: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run full pipeline on a list of telemetry dicts (oldest -> newest).
        The last row is the prediction target.
        """
        t0 = time.perf_counter()

        if not readings:
            raise ValueError("readings must not be empty")

        if not self.is_ready:
            raise RuntimeError("InferenceEngine is not ready; call bootstrap() first")

        mid = machine_id or readings[-1].get("machine_id", "UNKNOWN")
        ground = readings[-1].get("failure_mode")

        df = FeatureEngineer.process_telemetry(readings)
        latest = df.tail(1)
        df_anom = self.anomaly_detector.detect(latest)

        is_anomaly = bool(df_anom.iloc[0]["is_anomaly"])
        anomaly_score = float(df_anom.iloc[0]["anomaly_score"])
        feature_dict = df_anom.iloc[0].to_dict()

        health_score, risk_level = HealthScoreCalculator.calculate(feature_dict)
        preds, probs = self.classifier.predict(df_anom)
        failure_mode = str(preds[0])
        failure_probabilities = probs[0]
        failure_probability = float(failure_probabilities.get(failure_mode, 0.0))

        rul_hours = float(RULEstimator.estimate(health_score, feature_dict))
        rec = _recommendation(risk_level, failure_mode, is_anomaly)

        correct = None
        if ground is not None:
            correct = str(ground) == failure_mode

        elapsed = (time.perf_counter() - t0) * 1000.0
        ts = readings[-1].get("timestamp") or datetime.now(timezone.utc).isoformat()

        return InferenceResult(
            machine_id=str(mid),
            timestamp=str(ts),
            health_score=float(health_score),
            risk_level=str(risk_level),
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            failure_mode=failure_mode,
            failure_probabilities=failure_probabilities,
            failure_probability=round(failure_probability, 2),
            rul_hours=rul_hours,
            maintenance_recommendation=rec,
            model_status="READY",
            ground_truth_failure=str(ground) if ground is not None else None,
            prediction_correct=correct,
            inference_ms=round(elapsed, 2),
            features_used=int(latest.shape[1]),
        )


# Process-wide singleton helpers
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def get_ready_engine() -> InferenceEngine:
    eng = get_engine()
    if not eng.is_ready:
        eng.bootstrap_from_generator()
    return eng
