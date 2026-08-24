"""Dataset generation defaults for industrial synthetic versions."""

from __future__ import annotations

COMPONENT_CLASSES = [
    "motor",
    "bearing",
    "shaft",
    "pulley",
    "belt",
    "coupling",
    "housing",
    "panel",
]

CONDITIONS = [
    "normal",
    "misalignment",
    "belt_degradation",
    "component_missing",
    "structural_change",
    "obstruction",
    "surface_damage",
    "hard_negative",
]

VIEWS = ["front", "side", "angled", "closeup"]
LIGHTING = ["bright", "normal", "dark", "side_light", "soft_light"]

SEVERITY_LEVELS_V01 = [0.0, 0.25, 0.5, 0.75, 1.0]
# continuous-ish severity for v0.2
SEVERITY_LEVELS_V02 = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = (640, 480)
DEFAULT_N_PER_CONDITION = 24
DEFAULT_N_PER_CONDITION_V02 = 16

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATASET_VERSION_V01 = "industrial_dataset_v0.1"
DATASET_VERSION_V02 = "industrial_dataset_v0.2"
DATASET_VERSION = DATASET_VERSION_V01  # backward-compatible default
