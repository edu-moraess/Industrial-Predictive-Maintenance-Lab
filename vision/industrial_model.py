"""Industrial anomaly model facade — prefers v0.2, falls back to v0.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vision.anomaly_detectors import PCAResidualDetector, heatmap_to_bgr, try_load_detector


class IndustrialAnomalyModel:
    def __init__(self, detector: PCAResidualDetector):
        self.detector = detector
        self.meta = detector.meta

    @classmethod
    def try_load(cls, prefer: str = "industrial_anomaly_v0.2") -> Optional["IndustrialAnomalyModel"]:
        det = try_load_detector(prefer)
        if det is None:
            return None
        return cls(det)

    def score_array(self, image_bgr: np.ndarray) -> float:
        return self.detector.score(image_bgr)

    def score_with_heatmap(self, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        score, heat = self.detector.score_with_heatmap(image_bgr)
        return score, heatmap_to_bgr(heat)

    def component_scores(self, image_bgr: np.ndarray, annotations: list) -> Dict[str, float]:
        return self.detector.component_scores(image_bgr, annotations)

    def visual_health(self, anomaly_score: float) -> int:
        return int(np.clip(round(100 * (1.0 - anomaly_score)), 0, 100))

    def status_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.meta.get("model_name", self.detector.name),
            "task": self.meta.get("task", "visual_anomaly"),
            "dataset_version": self.meta.get("dataset_version", "unknown"),
            "feature_mode": self.meta.get("feature_mode", "gray"),
            "available": True,
            "note": self.meta.get("note", ""),
        }


def industrial_model_status() -> Dict[str, Any]:
    m = IndustrialAnomalyModel.try_load()
    if m is None:
        return {
            "available": False,
            "model_name": "industrial_anomaly_v0.2",
            "message": "Not trained. python -m dataset.generator --version v0.2 && python -m training.train --dataset-version v0.2",
        }
    return m.status_dict()
