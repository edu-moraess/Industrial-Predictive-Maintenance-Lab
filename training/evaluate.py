"""Evaluate industrial anomaly model on validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset.manifest import read_manifest
from training.config import CHECKPOINT_DIR, DATASET_PATH, MODEL_NAME, REGISTRY_PATH
from vision.industrial_model import IndustrialAnomalyModel


def evaluate(dataset_path: Path | None = None) -> dict:
    root = dataset_path or DATASET_PATH
    ckpt = CHECKPOINT_DIR / f"{MODEL_NAME}.npz"
    model = IndustrialAnomalyModel.try_load(MODEL_NAME)
    if model is None:
        version = MODEL_NAME.rsplit("_", 1)[-1]  # e.g. "industrial_anomaly_v0.2" -> "v0.2"
        raise RuntimeError(
            f"MODEL NOT TRAINED — no checkpoint at {ckpt}. "
            f"Run: python -m dataset.generator --version {version} && "
            f"python -m training.train --dataset-version {version}"
        )
    rows = read_manifest(root / "dataset_manifest.csv")
    test_rows = [r for r in rows if r["split"] in ("validation", "test")]

    y_true = []
    y_score = []
    for r in test_rows:
        path = root / r["path"]
        score = model.score_path(path)
        y_score.append(score)
        y_true.append(0 if r["condition"] == "normal" else 1)

    y_true_a = np.array(y_true)
    y_score_a = np.array(y_score)
    # simple threshold 0.35
    y_pred = (y_score_a >= 0.35).astype(int)
    tp = int(((y_pred == 1) & (y_true_a == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_a == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_a == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_a == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    metrics = {
        "n_eval": len(test_rows),
        "threshold": 0.35,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "note": "Metrics on synthetic schematic dataset only",
    }

    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        for e in reg:
            if e.get("model_name") == MODEL_NAME:
                e["metrics"] = metrics
        REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    reports = Path(__file__).resolve().parent.parent / "reports" / "evaluation"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{MODEL_NAME}_eval.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
