"""
Industrial Dataset Factory — synthetic rotating-machine imagery.

All images are SYNTHETIC (programmatic drawing). Not industrial photos.

Usage:
    python -m dataset.generator
    python -m dataset.generator --n-per-condition 40 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from dataset.config import (
    COMPONENT_CLASSES,
    CONDITIONS,
    DATASET_VERSION,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_N_PER_CONDITION,
    DEFAULT_SEED,
    SEVERITY_LEVELS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from dataset.manifest import write_manifest
from dataset.splitter import assign_splits

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "synthetic" / DATASET_VERSION


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _base_machine(draw: ImageDraw.ImageDraw, w: int, h: int, sev: float, condition: str) -> Dict[str, Tuple[int, int, int, int]]:
    """Draw schematic rotating machine; return component bboxes (x1,y1,x2,y2)."""
    boxes: Dict[str, Tuple[int, int, int, int]] = {}

    # Housing
    hx1, hy1, hx2, hy2 = int(w * 0.12), int(h * 0.22), int(w * 0.88), int(h * 0.78)
    draw.rectangle([hx1, hy1, hx2, hy2], outline=(180, 180, 190), width=3)
    boxes["housing"] = (hx1, hy1, hx2, hy2)

    # Motor (left)
    mx1, my1 = int(w * 0.16), int(h * 0.32)
    mx2, my2 = int(w * 0.34), int(h * 0.68)
    if condition != "component_missing" or sev < 0.5:
        draw.rectangle([mx1, my1, mx2, my2], fill=(70, 90, 110), outline=(210, 210, 220))
        boxes["motor"] = (mx1, my1, mx2, my2)

    # Coupling
    cx1, cy1 = int(w * 0.34), int(h * 0.44)
    cx2, cy2 = int(w * 0.40), int(h * 0.56)
    if condition != "component_missing" or sev < 0.75:
        draw.ellipse([cx1, cy1, cx2, cy2], fill=(120, 100, 60), outline=(220, 200, 140))
        boxes["coupling"] = (cx1, cy1, cx2, cy2)

    # Shaft — misalignment shifts Y
    shaft_y_off = int(sev * 18) if condition == "misalignment" else 0
    sx1, sy1 = int(w * 0.40), int(h * 0.48) + shaft_y_off
    sx2, sy2 = int(w * 0.62), int(h * 0.52) + shaft_y_off
    draw.rectangle([sx1, sy1, sx2, sy2], fill=(160, 160, 170))
    boxes["shaft"] = (sx1, sy1 - 4, sx2, sy2 + 4)

    # Bearing
    bx1, by1 = int(w * 0.50), int(h * 0.42) + shaft_y_off
    bx2, by2 = int(w * 0.58), int(h * 0.58) + shaft_y_off
    draw.ellipse([bx1, by1, bx2, by2], outline=(200, 160, 80), width=3)
    boxes["bearing"] = (bx1, by1, bx2, by2)

    # Pulley
    px1, py1 = int(w * 0.62), int(h * 0.36) + shaft_y_off // 2
    px2, py2 = int(w * 0.78), int(h * 0.64) + shaft_y_off // 2
    draw.ellipse([px1, py1, px2, py2], outline=(140, 140, 150), width=4)
    boxes["pulley"] = (px1, py1, px2, py2)

    # Belt — degradation thins / shifts stroke
    belt_w = max(1, 4 - int(sev * 3)) if condition == "belt_degradation" else 4
    belt_shift = int(sev * 12) if condition == "belt_degradation" else 0
    by_mid = (py1 + py2) // 2 + belt_shift
    draw.arc([px1 - 20, by_mid - 40, px2 + 10, by_mid + 40], 200, 340, fill=(90, 90, 100), width=belt_w)
    boxes["belt"] = (px1 - 20, by_mid - 40, px2 + 10, by_mid + 40)

    # Panel
    pnx1, pny1 = int(w * 0.72), int(h * 0.24)
    pnx2, pny2 = int(w * 0.86), int(h * 0.36)
    draw.rectangle([pnx1, pny1, pnx2, pny2], fill=(50, 60, 70), outline=(180, 190, 200))
    boxes["panel"] = (pnx1, pny1, pnx2, pny2)

    # Surface damage
    if condition == "surface_damage" and sev > 0:
        for _ in range(int(3 + sev * 8)):
            x = int(w * (0.2 + 0.5 * random.random()))
            y = int(h * (0.3 + 0.4 * random.random()))
            r = int(4 + sev * 12)
            draw.ellipse([x, y, x + r, y + r], fill=(40, 30, 30))

    # Structural change — extra bar
    if condition == "structural_change" and sev > 0.25:
        draw.line([(int(w * 0.2), int(h * 0.3)), (int(w * 0.7), int(h * 0.7))], fill=(200, 80, 80), width=3)

    # Obstruction
    if condition == "obstruction" and sev > 0:
        ox1 = int(w * (0.3 + 0.2 * sev))
        oy1 = int(h * 0.35)
        ox2 = ox1 + int(w * 0.15 * sev)
        oy2 = oy1 + int(h * 0.25)
        draw.rectangle([ox1, oy1, ox2, oy2], fill=(30, 30, 35))

    return boxes


def render_machine(
    condition: str,
    severity: float,
    seed: int,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
    r = _rng(seed)
    random.seed(seed)
    np.random.seed(seed % (2**31 - 1))

    w, h = size
    # Background variation
    bg = int(30 + r.random() * 40)
    img = Image.new("RGB", (w, h), (bg, bg, bg + 5))
    draw = ImageDraw.Draw(img)

    # Floor line
    draw.line([(0, int(h * 0.82)), (w, int(h * 0.82))], fill=(60, 60, 65), width=2)

    boxes = _base_machine(draw, w, h, severity, condition)

    annotations = []
    for cls, (x1, y1, x2, y2) in boxes.items():
        if cls not in COMPONENT_CLASSES:
            continue
        cond = condition if condition != "normal" else "normal"
        # component-level condition simplified
        annotations.append(
            {
                "class": cls,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "condition": cond,
                "severity": float(severity) if condition != "normal" else 0.0,
            }
        )

    # Photometric / geometric variation (still synthetic)
    if r.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(0.7 + r.random() * 0.6)
    if r.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(0.8 + r.random() * 0.5)
    if r.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 1.2)))
    if r.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + np.random.randn(*arr.shape) * (4 + severity * 8), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # Small rotation
    angle = r.uniform(-6, 6) + (severity * 8 if condition == "misalignment" else 0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(bg, bg, bg + 5))

    meta = {
        "condition": condition,
        "severity": float(severity),
        "synthetic": True,
        "camera_angle": round(angle, 2),
        "lighting": round(0.5 + r.random() * 0.5, 3),
        "machine_id": "machine_001",
        "dataset_version": DATASET_VERSION,
    }
    return img, annotations, meta


def generate_dataset(
    n_per_condition: int = DEFAULT_N_PER_CONDITION,
    seed: int = DEFAULT_SEED,
    out_root: Path | None = None,
) -> Path:
    out = out_root or DATA_ROOT
    images_dir = out / "images"
    ann_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    idx = 0
    for condition in CONDITIONS:
        for i in range(n_per_condition):
            severity = 0.0 if condition == "normal" else SEVERITY_LEVELS[(i % (len(SEVERITY_LEVELS) - 1)) + 1]
            img_seed = seed + idx * 17 + hash(condition) % 1000
            img, anns, meta = render_machine(condition, severity, img_seed)
            image_id = f"{condition}_{idx:05d}"
            rel_img = f"images/{image_id}.png"
            rel_ann = f"annotations/{image_id}.json"
            img_path = out / rel_img
            ann_path = out / rel_ann
            img.save(img_path)
            payload = {"image_id": image_id, "annotations": anns, **meta}
            ann_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            scene_id = f"{condition}_{i // 4}"  # group for split leakage control
            rows.append(
                {
                    "image_id": image_id,
                    "machine_id": meta["machine_id"],
                    "scene_id": scene_id,
                    "path": rel_img,
                    "annotation_path": rel_ann,
                    "condition": condition,
                    "severity": severity,
                    "synthetic": True,
                    "camera_angle": meta["camera_angle"],
                    "lighting": meta["lighting"],
                }
            )
            idx += 1

    rows = assign_splits(rows, seed=seed, train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO)
    write_manifest(out / "dataset_manifest.csv", rows)

    meta_path = out / "dataset_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "version": DATASET_VERSION,
                "synthetic": True,
                "n_images": len(rows),
                "classes": COMPONENT_CLASSES,
                "conditions": CONDITIONS,
                "seed": seed,
                "n_per_condition": n_per_condition,
                "note": "Programmatic synthetic schematics — not industrial photographs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} images at {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Industrial Dataset Factory v0.1")
    p.add_argument("--n-per-condition", type=int, default=DEFAULT_N_PER_CONDITION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    generate_dataset(n_per_condition=args.n_per_condition, seed=args.seed)


if __name__ == "__main__":
    main()
