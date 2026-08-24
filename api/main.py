"""
Industrial Predictive Maintenance Lab — FastAPI backend.

Models are initialized explicitly via lifespan (not at import time).
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.schemas import MachineResponse, PredictResponse, SensorInput
from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector
from ml.failure_classifier import FailureClassifier
from ml.health_score import HealthScoreCalculator
from ml.rul import RULEstimator
from simulator.generator import DataGenerator

repo = DatabaseRepository()
anomaly_detector = AnomalyDetector()
classifier = FailureClassifier()
_models_ready = False


def bootstrap_models(force: bool = False) -> None:
    """Train Isolation Forest + Random Forest on a short synthetic history."""
    global _models_ready
    if _models_ready and not force:
        return

    generator = DataGenerator(repo)
    generator.generate_historical_dataset(
        "INIT_TRAIN_MACHINE", hours=6, frequency_minutes=5
    )
    history = repo.get_historical_readings("INIT_TRAIN_MACHINE", limit=100)

    if history:
        df_features = FeatureEngineer.process_telemetry(history)
        anomaly_detector.train(df_features)
        classifier.train(df_features)
        _models_ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_models()
    yield


app = FastAPI(
    title="Industrial Predictive Maintenance Lab API",
    version="1.0.0",
    description=(
        "Experimental API for synthetic industrial telemetry, "
        "anomaly detection, health scoring, failure classification and RUL estimation."
    ),
    lifespan=lifespan,
)

# Development CORS — restrict via environment in production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health_status() -> Dict[str, str]:
    return {
        "status": "online",
        "service": "Industrial Predictive Maintenance Engine",
        "models_ready": str(_models_ready).lower(),
    }


@app.get("/machines", response_model=List[MachineResponse])
def get_machines():
    return repo.get_machines()


@app.get("/machines/{machine_id}")
def get_machine_details(machine_id: str):
    machines = repo.get_machines()
    machine = next((m for m in machines if m["machine_id"] == machine_id), None)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return machine


@app.get("/machines/{machine_id}/latest")
def get_latest_telemetry(machine_id: str):
    reading = repo.get_latest_reading(machine_id)
    if not reading:
        raise HTTPException(status_code=404, detail="No readings found for this machine.")
    return reading


@app.post("/predict", response_model=PredictResponse)
def predict_machine_health(telemetry: SensorInput):
    if not anomaly_detector.is_fitted:
        bootstrap_models()

    history = repo.get_historical_readings(telemetry.machine_id, limit=20)
    current_dict = telemetry.model_dump()
    raw_list = history + [current_dict]

    df_features = FeatureEngineer.process_telemetry(raw_list)
    latest_row_df = df_features.tail(1)

    df_anomaly = anomaly_detector.detect(latest_row_df)
    is_anomaly = bool(df_anomaly.iloc[0]["is_anomaly"])
    anomaly_score = float(df_anomaly.iloc[0]["anomaly_score"])

    latest_feature_dict = df_anomaly.iloc[0].to_dict()
    health_score, risk_level = HealthScoreCalculator.calculate(latest_feature_dict)

    predictions, probabilities = classifier.predict(df_anomaly)
    predicted_failure = predictions[0]
    failure_probs = probabilities[0]

    rul_hours = RULEstimator.estimate(health_score, latest_feature_dict)

    return PredictResponse(
        machine_id=telemetry.machine_id,
        health_score=health_score,
        risk_level=risk_level,
        anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        failure_type=predicted_failure,
        failure_probabilities=failure_probs,
        rul_hours=rul_hours,
    )


@app.get("/anomalies")
def get_anomalies(machine_id: str, limit: int = 20):
    if not anomaly_detector.is_fitted:
        bootstrap_models()

    history = repo.get_historical_readings(machine_id, limit=limit)
    if not history:
        return []

    df_features = FeatureEngineer.process_telemetry(history)
    df_anomalies = anomaly_detector.detect(df_features)
    anomalies_only = df_anomalies[df_anomalies["is_anomaly"] == True]
    return anomalies_only.to_dict(orient="records")
