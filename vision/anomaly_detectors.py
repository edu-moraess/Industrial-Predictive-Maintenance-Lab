"""
Anomaly detector interface + PCA residual implementations (v0.1 / v0.2).

Future: AutoencoderDetector, EmbeddingDistanceDetector, PatchAnomalyDetector.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"


class AnomalyDetector(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, image_bgr: np.ndarray) -> float:
        ...

    def score_with_heatmap(self, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Default: uniform heatmap. Subclasses should override."""
        s = self.score(image_bgr)
        h, w = image_bgr.shape[:2]
        heat = np.full((h, w), s, dtype=np.float32)
        return s, heat

    def component_scores(
        self, image_bgr: np.ndarray, annotations: list
    ) -> Dict[str, float]:
        out = {}
        for ann in annotations:
            x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image_bgr.shape[1], x2)
            y2 = min(image_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            out[ann["class"]] = self.score(crop)
        return out


def _vectorize(image_bgr: np.ndarray, size=(64, 64), mode="gray") -> np.ndarray:
    if image_bgr.ndim == 3:
        if mode == "gray":
            g = (
                0.114 * image_bgr[:, :, 0]
                + 0.587 * image_bgr[:, :, 1]
                + 0.299 * image_bgr[:, :, 2]
            )
            img = Image.fromarray(g.astype(np.uint8))
        else:
            rgb = image_bgr[:, :, ::-1]
            img = Image.fromarray(rgb.astype(np.uint8))
    else:
        img = Image.fromarray(image_bgr.astype(np.uint8))
    img = img.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if mode == "rich" and arr.ndim == 2:
        # grayscale + simple gradient magnitude
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx * gx + gy * gy)
        arr = np.concatenate([arr.reshape(-1), mag.reshape(-1)])
        return arr
    return arr.reshape(-1)


class PCAResidualDetector(AnomalyDetector):
    def __init__(self, mean, components, res_mean, res_std, meta=None, feature_mode="gray", size=(64, 64)):
        self.mean = mean
        self.components = components
        self.res_mean = res_mean
        self.res_std = res_std
        self.meta = meta or {}
        self.feature_mode = feature_mode
        self.size = size
        self.name = self.meta.get("model_name", "pca_residual")

    @classmethod
    def load(cls, path: Path) -> "PCAResidualDetector":
        data = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"]))
            except Exception:
                meta = {}
        mode = meta.get("feature_mode", "gray")
        size = tuple(meta.get("feature_size", [64, 64]))
        return cls(
            mean=data["mean"],
            components=data["components"],
            res_mean=float(data["res_mean"][0]),
            res_std=float(data["res_std"][0]),
            meta=meta,
            feature_mode=mode,
            size=size,
        )

    def score(self, image_bgr: np.ndarray) -> float:
        x = _vectorize(image_bgr, self.size, self.feature_mode)
        # pad/truncate if crop size differs
        if x.shape[0] != self.mean.shape[0]:
            x = _vectorize(image_bgr, self.size, self.feature_mode)
            if x.shape[0] < self.mean.shape[0]:
                x = np.pad(x, (0, self.mean.shape[0] - x.shape[0]))
            else:
                x = x[: self.mean.shape[0]]
        z = x - self.mean
        recon = self.mean + (z @ self.components.T) @ self.components
        residual = float(np.linalg.norm(x - recon))
        zscore = (residual - self.res_mean) / (self.res_std + 1e-9)
        score = 1.0 / (1.0 + np.exp(-0.8 * (zscore - 1.0)))
        return float(np.clip(score, 0.0, 1.0))

    def score_with_heatmap(self, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Patch-wise residual heatmap (coarse localization)."""
        h, w = image_bgr.shape[:2]
        patch = 64
        stride = 32
        heat = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        scores = []
        for y in range(0, max(1, h - patch + 1), stride):
            for x in range(0, max(1, w - patch + 1), stride):
                crop = image_bgr[y : y + patch, x : x + patch]
                if crop.shape[0] < 16 or crop.shape[1] < 16:
                    continue
                s = self.score(crop)
                scores.append(s)
                heat[y : y + patch, x : x + patch] += s
                counts[y : y + patch, x : x + patch] += 1.0
        counts[counts == 0] = 1.0
        heat = heat / counts
        global_s = float(np.mean(scores)) if scores else self.score(image_bgr)
        return global_s, heat


def try_load_detector(model_name: str = "industrial_anomaly_v0.2") -> Optional[PCAResidualDetector]:
    path = CHECKPOINT_DIR / f"{model_name}.npz"
    if not path.exists():
        # fallback v0.1
        alt = CHECKPOINT_DIR / "industrial_anomaly_v0.1.npz"
        if alt.exists():
            return PCAResidualDetector.load(alt)
        return None
    return PCAResidualDetector.load(path)


def heatmap_to_bgr(heat: np.ndarray) -> np.ndarray:
    h = heat.copy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    u8 = (h * 255).astype(np.uint8)
    # simple colormap without OpenCV requirement
    rgb = np.zeros((*u8.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = u8  # R
    rgb[:, :, 1] = (u8.astype(np.uint16) * 0.4).astype(np.uint8)
    rgb[:, :, 2] = 255 - u8  # B inverse
    return rgb[:, :, ::-1]  # BGR
