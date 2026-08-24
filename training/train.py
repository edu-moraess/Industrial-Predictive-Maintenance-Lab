"""
Train industrial anomaly models.

    python -m training.train --dataset-version v0.1 --model pca
    python -m training.train --dataset-version v0.2 --model pca --feature-mode rich
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.manifest import read_manifest
from training.config import CHECKPOINT_DIR, REGISTRY_PATH, ROOT


def _load_vector(path: Path, size=(64, 64), mode="gray") -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1].copy()
    from vision.anomaly_detectors import _vectorize

    return _vectorize(arr, size, mode)


def train(
    dataset_version: str = "v0.2",
    model: str = "pca",
    feature_mode: str = "rich",
    seed: int = 42,
) -> Path:
    if dataset_version in ("v0.1", "0.1"):
        ds_name = "industrial_dataset_v0.1"
        model_name = "industrial_anomaly_v0.1"
        feature_mode = "gray"
        size = (64, 64)
    else:
        ds_name = "industrial_dataset_v0.2"
        model_name = "industrial_anomaly_v0.2"
        size = (64, 64)

    root = ROOT / "data" / "synthetic" / ds_name
    manifest = root / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}. Run: python -m dataset.generator --version {dataset_version}")

    rows = read_manifest(manifest)
    # train on NORMAL only (exclude hard_negative from positive normal set but they are condition hard_negative)
    train_normal = [
        r
        for r in rows
        if r["split"] == "train"
        and r["condition"] == "normal"
        and str(r.get("hard_negative", "")).lower() not in ("true", "1")
    ]
    if len(train_normal) < 3:
        raise RuntimeError("Need >=3 normal train images")

    X = np.stack([_load_vector(root / r["path"], size, feature_mode) for r in train_normal])
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(24 if feature_mode == "rich" else 16, vt.shape[0] - 1, X.shape[0] - 1)
    components = vt[: max(1, n_comp)]

    residuals = []
    for x in X:
        z = x - mean
        recon = mean + (z @ components.T) @ components
        residuals.append(float(np.linalg.norm(x - recon)))
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals) + 1e-6)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"{model_name}.npz"
    meta = {
        "model_name": model_name,
        "task": "visual_anomaly",
        "dataset_version": ds_name,
        "feature_mode": feature_mode,
        "feature_size": list(size),
        "method": model,
        "n_train_normal": len(train_normal),
        "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "note": "PCA residual on synthetic domain — experimental",
    }
    np.savez_compressed(
        ckpt,
        mean=mean,
        components=components,
        res_mean=np.array([res_mean]),
        res_std=np.array([res_std]),
        meta=json.dumps(meta),
    )

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            registry = []
    entry = {
        "model_name": model_name,
        "task": "visual_anomaly",
        "dataset_version": ds_name,
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "trained_at": meta["trained_at"],
        "feature_mode": feature_mode,
        "metrics": {},
        "note": meta["note"],
    }
    registry = [e for e in registry if e.get("model_name") != model_name]
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved {ckpt} (train_normal={len(train_normal)}, features={feature_mode})")
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", default="v0.2")
    p.add_argument("--model", default="pca")
    p.add_argument("--feature-mode", default="rich", choices=["gray", "rich"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args.dataset_version, args.model, args.feature_mode, args.seed)


if __name__ == "__main__":
    main()
