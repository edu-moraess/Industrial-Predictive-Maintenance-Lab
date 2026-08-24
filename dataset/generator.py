"""
Industrial Dataset Factory

    python -m dataset.generator
    python -m dataset.generator --version v0.1
    python -m dataset.generator --version v0.2 --n-per-condition 16 --seed 7
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
    DATASET_VERSION_V01,
    DATASET_VERSION_V02,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_N_PER_CONDITION,
    DEFAULT_N_PER_CONDITION_V02,
    DEFAULT_SEED,
    LIGHTING,
    SEVERITY_LEVELS_V01,
    SEVERITY_LEVELS_V02,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    VIEWS,
)
from dataset.manifest import write_manifest
from dataset.render_v02 import render_machine_v02
from dataset.splitter import assign_splits

ROOT = Path(__file__).resolve().parent.parent


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _base_machine_v01(draw, w, h, sev, condition):
    boxes = {}
    hx1, hy1, hx2, hy2 = int(w * 0.12), int(h * 0.22), int(w * 0.88), int(h * 0.78)
    draw.rectangle([hx1, hy1, hx2, hy2], outline=(180, 180, 190), width=3)
    boxes["housing"] = (hx1, hy1, hx2, hy2)
    if condition != "component_missing" or sev < 0.5:
        mx1, my1, mx2, my2 = int(w * 0.16), int(h * 0.32), int(w * 0.34), int(h * 0.68)
        draw.rectangle([mx1, my1, mx2, my2], fill=(70, 90, 110), outline=(210, 210, 220))
        boxes["motor"] = (mx1, my1, mx2, my2)
    if condition != "component_missing" or sev < 0.75:
        cx1, cy1, cx2, cy2 = int(w * 0.34), int(h * 0.44), int(w * 0.40), int(h * 0.56)
        draw.ellipse([cx1, cy1, cx2, cy2], fill=(120, 100, 60), outline=(220, 200, 140))
        boxes["coupling"] = (cx1, cy1, cx2, cy2)
    shaft_y_off = int(sev * 18) if condition == "misalignment" else 0
    sx1, sy1 = int(w * 0.40), int(h * 0.48) + shaft_y_off
    sx2, sy2 = int(w * 0.62), int(h * 0.52) + shaft_y_off
    draw.rectangle([sx1, sy1, sx2, sy2], fill=(160, 160, 170))
    boxes["shaft"] = (sx1, sy1 - 4, sx2, sy2 + 4)
    bx1, by1 = int(w * 0.50), int(h * 0.42) + shaft_y_off
    bx2, by2 = int(w * 0.58), int(h * 0.58) + shaft_y_off
    draw.ellipse([bx1, by1, bx2, by2], outline=(200, 160, 80), width=3)
    boxes["bearing"] = (bx1, by1, bx2, by2)
    px1, py1 = int(w * 0.62), int(h * 0.36) + shaft_y_off // 2
    px2, py2 = int(w * 0.78), int(h * 0.64) + shaft_y_off // 2
    draw.ellipse([px1, py1, px2, py2], outline=(140, 140, 150), width=4)
    boxes["pulley"] = (px1, py1, px2, py2)
    belt_w = max(1, 4 - int(sev * 3)) if condition == "belt_degradation" else 4
    belt_shift = int(sev * 12) if condition == "belt_degradation" else 0
    by_mid = (py1 + py2) // 2 + belt_shift
    draw.arc([px1 - 20, by_mid - 40, px2 + 10, by_mid + 40], 200, 340, fill=(90, 90, 100), width=belt_w)
    boxes["belt"] = (px1 - 20, by_mid - 40, px2 + 10, by_mid + 40)
    pnx1, pny1, pnx2, pny2 = int(w * 0.72), int(h * 0.24), int(w * 0.86), int(h * 0.36)
    draw.rectangle([pnx1, pny1, pnx2, pny2], fill=(50, 60, 70), outline=(180, 190, 200))
    boxes["panel"] = (pnx1, pny1, pnx2, pny2)
    if condition == "surface_damage" and sev > 0:
        for _ in range(int(3 + sev * 8)):
            x = int(w * (0.2 + 0.5 * random.random()))
            y = int(h * (0.3 + 0.4 * random.random()))
            rr = int(4 + sev * 12)
            draw.ellipse([x, y, x + rr, y + rr], fill=(40, 30, 30))
    if condition == "structural_change" and sev > 0.25:
        draw.line([(int(w * 0.2), int(h * 0.3)), (int(w * 0.7), int(h * 0.7))], fill=(200, 80, 80), width=3)
    if condition == "obstruction" and sev > 0:
        ox1 = int(w * (0.3 + 0.2 * sev))
        oy1 = int(h * 0.35)
        draw.rectangle([ox1, oy1, ox1 + int(w * 0.15 * sev), oy1 + int(h * 0.25)], fill=(30, 30, 35))
    return boxes


def render_machine_v01(condition, severity, seed, size=DEFAULT_IMAGE_SIZE):
    r = _rng(seed)
    random.seed(seed)
    np.random.seed(seed % (2**31 - 1))
    w, h = size
    bg = int(30 + r.random() * 40)
    img = Image.new("RGB", (w, h), (bg, bg, bg + 5))
    draw = ImageDraw.Draw(img)
    draw.line([(0, int(h * 0.82)), (w, int(h * 0.82))], fill=(60, 60, 65), width=2)
    boxes = _base_machine_v01(draw, w, h, severity, condition)
    annotations = []
    for cls, (x1, y1, x2, y2) in boxes.items():
        if cls not in COMPONENT_CLASSES:
            continue
        annotations.append(
            {
                "class": cls,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "condition": condition,
                "severity": float(severity) if condition != "normal" else 0.0,
            }
        )
    if r.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(0.7 + r.random() * 0.6)
    if r.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(0.8 + r.random() * 0.5)
    if r.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 1.2)))
    angle = r.uniform(-6, 6)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(bg, bg, bg + 5))
    meta = {
        "condition": condition,
        "severity": float(severity),
        "synthetic": True,
        "camera_angle": round(angle, 2),
        "lighting": round(0.5 + r.random() * 0.5, 3),
        "machine_id": "machine_001",
        "dataset_version": DATASET_VERSION_V01,
    }
    return img, annotations, meta


def generate_dataset(
    version: str = "v0.1",
    n_per_condition: int | None = None,
    seed: int = DEFAULT_SEED,
    out_root: Path | None = None,
) -> Path:
    if version in ("v0.1", "0.1", DATASET_VERSION_V01):
        ds_name = DATASET_VERSION_V01
        n_per = n_per_condition or DEFAULT_N_PER_CONDITION
        conditions = [c for c in CONDITIONS if c != "hard_negative"]
        sevs = SEVERITY_LEVELS_V01
        renderer = "v01"
    else:
        ds_name = DATASET_VERSION_V02
        n_per = n_per_condition or DEFAULT_N_PER_CONDITION_V02
        conditions = list(CONDITIONS)
        sevs = SEVERITY_LEVELS_V02
        renderer = "v02"

    out = out_root or (ROOT / "data" / "synthetic" / ds_name)
    images_dir = out / "images"
    ann_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    idx = 0
    for condition in conditions:
        for i in range(n_per):
            if condition in ("normal", "hard_negative"):
                severity = 0.0
            else:
                severity = sevs[(i % (len(sevs) - 1)) + 1]
            img_seed = seed + idx * 17 + (hash(condition) % 1000)
            if renderer == "v01":
                img, anns, meta = render_machine_v01(condition, severity, img_seed)
            else:
                view = VIEWS[i % len(VIEWS)]
                lighting = LIGHTING[i % len(LIGHTING)]
                img, anns, meta = render_machine_v02(
                    condition, severity, img_seed, view=view, lighting=lighting
                )
            image_id = f"{condition}_{idx:05d}"
            rel_img = f"images/{image_id}.png"
            rel_ann = f"annotations/{image_id}.json"
            img.save(out / rel_img)
            payload = {"image_id": image_id, "annotations": anns, **meta}
            (out / rel_ann).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            scene_id = f"{condition}_{i // 4}"
            rows.append(
                {
                    "image_id": image_id,
                    "machine_id": meta.get("machine_id", "machine_001"),
                    "scene_id": scene_id,
                    "path": rel_img,
                    "annotation_path": rel_ann,
                    "condition": condition,
                    "severity": meta.get("severity", severity),
                    "synthetic": True,
                    "camera_angle": meta.get("camera_angle", 0),
                    "lighting": meta.get("lighting", ""),
                    "view": meta.get("view", ""),
                    "hard_negative": meta.get("hard_negative", condition == "hard_negative"),
                }
            )
            idx += 1

    rows = assign_splits(rows, seed=seed, train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO)
    write_manifest(out / "dataset_manifest.csv", rows)
    (out / "dataset_meta.json").write_text(
        json.dumps(
            {
                "version": ds_name,
                "synthetic": True,
                "n_images": len(rows),
                "classes": COMPONENT_CLASSES,
                "conditions": conditions,
                "seed": seed,
                "n_per_condition": n_per,
                "renderer": renderer,
                "note": "Synthetic procedural imagery — not industrial photographs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} images at {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Industrial Dataset Factory")
    p.add_argument("--version", type=str, default="v0.2", help="v0.1 or v0.2")
    p.add_argument("--n-per-condition", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    generate_dataset(version=args.version, n_per_condition=args.n_per_condition, seed=args.seed)


if __name__ == "__main__":
    main()
