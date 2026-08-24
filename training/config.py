"""Shared training paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
REGISTRY_PATH = ROOT / "models" / "registry" / "model_registry.json"
FEATURE_SIZE = (64, 64)
RANDOM_SEED = 42
MODEL_NAME = "industrial_anomaly_v0.2"
DATASET_PATH = ROOT / "data" / "synthetic" / "industrial_dataset_v0.2"
