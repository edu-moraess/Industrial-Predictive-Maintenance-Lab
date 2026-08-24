"""
Industrial visual anomaly model (PCA residual on synthetic domain).

Primary inspection path for the lab — not COCO YOLO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from training.config import CHECKPOINT_DIR, FEATURE_SIZE, MODEL_NAME

ROOT = Path(__file__).resolve().parent.parent


class IndustrialAnomalyModel:
    def __init__(
        self,
        mean: np.ndarray,
        components: np.ndarray,
        res_mean: float,
        res_std: float,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.mean = mean
        self.components = components
        self.res_mean = res_mean
        self.res_std = res_std
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "IndustrialAnomalyModel":
        path = path or (CHECKPOINT_DIR / f"{MODEL_NAME}.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Run: python -m dataset.generator && python -m training.train"
            )
        data = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"]))
            except Exception:
                meta = {}
        return cls(
            mean=data["mean"],
            components=data["components"],
            res_mean=float(data["res_mean"][0]),
            res_std=float(data["res_std"][0]),
            meta=meta,
        )

    @classmethod
    def try_load(cls) -> Optional["IndustrialAnomalyModel"]:
        try:
            return cls.load()
        except Exception:
            return None

    def _vectorize_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 3:
            # approximate luminance from BGR
            gray = (
                0.114 * image_bgr[:, :, 0]
                + 0.587 * image_bgr[:, :, 1]
                + 0.299 * image_bgr[:, :, 2]
            )
        else:
            gray = image_bgr
        img = Image.fromarray(gray.astype(np.uint8)).resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    def score_array(self, image_bgr: np.ndarray) -> float:
        x = self._vectorize_bgr(image_bgr)
        z = x - self.mean
        recon = self.mean + (z @ self.components.T) @ self.components
        residual = float(np.linalg.norm(x - recon))
        # normalize using train residual stats
        zscore = (residual - self.res_mean) / self.res_std
        score = 1.0 / (1.0 + np.exp(-0.8 * (zscore - 1.0)))  # sigmoid
        return float(np.clip(score, 0.0, 1.0))

    def score_path(self, path: Path) -> float:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)[:, :, ::-1].copy()  # RGB->BGR convention
        return self.score_array(arr)

    def visual_health(self, anomaly_score: float) -> int:
        """0-100 visual health (experimental)."""
        return int(np.clip(round(100 * (1.0 - anomaly_score)), 0, 100))

    def status_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.meta.get("model_name", MODEL_NAME),
            "task": self.meta.get("task", "visual_anomaly"),
            "dataset_version": self.meta.get("dataset_version", "unknown"),
            "available": True,
            "note": self.meta.get("note", ""),
        }


def industrial_model_status() -> Dict[str, Any]:
    m = IndustrialAnomalyModel.try_load()
    if m is None:
        return {
            "available": False,
            "model_name": MODEL_NAME,
            "message": "Not trained. Run: python -m dataset.generator && python -m training.train",
        }
    return m.status_dict()
