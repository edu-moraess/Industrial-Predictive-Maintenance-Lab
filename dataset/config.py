"""Dataset generation defaults (industrial synthetic v0.1)."""

from __future__ import annotations

# Industrial component classes (detection / ROI labels)
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

# Visual conditions for generation
CONDITIONS = [
    "normal",
    "misalignment",
    "belt_degradation",
    "component_missing",
    "structural_change",
    "obstruction",
    "surface_damage",
]

SEVERITY_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = (640, 480)
DEFAULT_N_PER_CONDITION = 24  # small, expandable
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATASET_VERSION = "industrial_dataset_v0.1"
