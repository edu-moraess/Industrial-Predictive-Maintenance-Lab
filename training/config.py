"""Training configuration for industrial anomaly model v0.1."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "industrial_dataset_v0.1"
DATASET_PATH = ROOT / "data" / "synthetic" / DATASET_VERSION
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
REGISTRY_PATH = ROOT / "models" / "registry" / "model_registry.json"

MODEL_NAME = "industrial_anomaly_v0.1"
FEATURE_SIZE = (64, 64)
RANDOM_SEED = 42
