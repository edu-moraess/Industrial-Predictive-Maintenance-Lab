"""Dataset manifest CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

FIELDS = [
    "image_id",
    "machine_id",
    "scene_id",
    "split",
    "path",
    "annotation_path",
    "condition",
    "severity",
    "synthetic",
    "camera_angle",
    "lighting",
    "view",
    "hard_negative",
]


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
