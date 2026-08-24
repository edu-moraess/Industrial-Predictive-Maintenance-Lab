"""Train/val/test split by scene_id to reduce leakage."""

from __future__ import annotations

import random
from typing import Any, Dict, List


def assign_splits(
    rows: List[Dict[str, Any]],
    seed: int = 42,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
) -> List[Dict[str, Any]]:
    scenes = sorted({r["scene_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))
    train_s = set(scenes[:n_train])
    val_s = set(scenes[n_train : n_train + n_val])
    test_s = set(scenes[n_train + n_val :])
    if not test_s and scenes:
        test_s.add(scenes[-1])
        train_s.discard(scenes[-1])

    out = []
    for r in rows:
        sid = r["scene_id"]
        if sid in train_s:
            split = "train"
        elif sid in val_s:
            split = "validation"
        else:
            split = "test"
        out.append({**r, "split": split})
    return out
