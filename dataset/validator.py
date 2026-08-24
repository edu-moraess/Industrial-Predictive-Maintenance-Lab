"""Validate industrial dataset on disk."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset.config import COMPONENT_CLASSES, DATASET_VERSION
from dataset.manifest import read_manifest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data" / "synthetic" / DATASET_VERSION


def validate(root: Path | None = None) -> dict:
    root = root or DEFAULT
    report = {"ok": True, "errors": [], "warnings": [], "stats": {}}
    manifest = root / "dataset_manifest.csv"
    if not manifest.exists():
        report["ok"] = False
        report["errors"].append("missing dataset_manifest.csv")
        return report

    rows = read_manifest(manifest)
    report["stats"]["n_images"] = len(rows)
    report["stats"]["by_condition"] = dict(Counter(r["condition"] for r in rows))
    report["stats"]["by_split"] = dict(Counter(r["split"] for r in rows))

    for r in rows:
        ip = root / r["path"]
        ap = root / r["annotation_path"]
        if not ip.exists():
            report["ok"] = False
            report["errors"].append(f"missing image {r['path']}")
        if not ap.exists():
            report["ok"] = False
            report["errors"].append(f"missing annotation {r['annotation_path']}")
            continue
        try:
            data = json.loads(ap.read_text(encoding="utf-8"))
            for ann in data.get("annotations", []):
                if ann.get("class") not in COMPONENT_CLASSES:
                    report["warnings"].append(f"unknown class {ann.get('class')} in {r['image_id']}")
                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    report["errors"].append(f"bad bbox in {r['image_id']}")
                    report["ok"] = False
        except Exception as exc:  # noqa: BLE001
            report["ok"] = False
            report["errors"].append(f"annotation parse {r['image_id']}: {exc}")

    # scene leakage rough check: same scene in train and test
    train_scenes = {r["scene_id"] for r in rows if r["split"] == "train"}
    test_scenes = {r["scene_id"] for r in rows if r["split"] == "test"}
    leak = train_scenes & test_scenes
    if leak:
        report["ok"] = False
        report["errors"].append(f"split leakage scenes: {sorted(leak)[:5]}")

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(DEFAULT))
    args = p.parse_args()
    rep = validate(Path(args.root))
    print(json.dumps(rep, indent=2))
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
