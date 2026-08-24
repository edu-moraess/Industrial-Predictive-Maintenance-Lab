"""
Train industrial visual anomaly model on synthetic dataset.

Method (v0.1): feature vectors from downscaled grayscale + PCA subspace of NORMAL train images.
Anomaly score = reconstruction / distance residual (0-1).

This is NOT a COCO detector and NOT industrial-validated.

Usage:
    python -m dataset.generator
    python -m training.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.manifest import read_manifest
from training.config import (
    CHECKPOINT_DIR,
    DATASET_PATH,
    FEATURE_SIZE,
    MODEL_NAME,
    RANDOM_SEED,
    REGISTRY_PATH,
)


def _load_gray_vector(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def train(dataset_path: Path | None = None) -> Path:
    root = dataset_path or DATASET_PATH
    manifest = root / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset not found: {manifest}. Run: python -m dataset.generator")

    rows = read_manifest(manifest)
    train_normal = [
        r for r in rows if r["split"] == "train" and r["condition"] == "normal"
    ]
    if len(train_normal) < 3:
        raise RuntimeError("Need at least 3 normal train images")

    X = np.stack([_load_gray_vector(root / r["path"]) for r in train_normal])
    mean = X.mean(axis=0)
    Xc = X - mean
    # PCA via SVD
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(16, vt.shape[0] - 1, X.shape[0] - 1)
    components = vt[:n_comp]

    # residual stats on train normal for score calibration
    residuals = []
    for x in X:
        z = x - mean
        recon = mean + (z @ components.T) @ components
        residuals.append(float(np.linalg.norm(x - recon)))
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals) + 1e-6)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"{MODEL_NAME}.npz"
    payload = {
        "model_name": MODEL_NAME,
        "task": "visual_anomaly",
        "dataset_version": root.name,
        "feature_size": list(FEATURE_SIZE),
        "mean": mean.tolist(),
        "components": components.tolist(),
        "res_mean": res_mean,
        "res_std": res_std,
        "n_train_normal": len(train_normal),
        "seed": RANDOM_SEED,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "note": "PCA residual anomaly on synthetic schematics — experimental",
    }
    np.savez_compressed(
        ckpt,
        mean=mean,
        components=components,
        res_mean=np.array([res_mean]),
        res_std=np.array([res_std]),
        meta=json.dumps({k: v for k, v in payload.items() if k not in ("mean", "components")}),
    )

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            registry = []
    entry = {
        "model_name": MODEL_NAME,
        "task": "visual_anomaly",
        "dataset_version": root.name,
        "checkpoint": str(ckpt.relative_to(Path(__file__).resolve().parent.parent)),
        "trained_at": payload["trained_at"],
        "metrics": {},  # filled by evaluate.py when run
        "note": payload["note"],
    }
    registry = [e for e in registry if e.get("model_name") != MODEL_NAME]
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved checkpoint {ckpt}")
    return ckpt


def main() -> None:
    train()


if __name__ == "__main__":
    main()
