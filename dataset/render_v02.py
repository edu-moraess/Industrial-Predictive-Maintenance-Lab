"""
Richer procedural rotating-machine renderer for industrial_dataset_v0.2.
Still synthetic (not photographs). More texture, lighting, multi-view cues.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from dataset.config import COMPONENT_CLASSES, DEFAULT_IMAGE_SIZE


def _noise(w: int, h: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w)).astype(np.float32)
    # cheap low-frequency via blur-like box
    k = max(3, int(scale))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(n, pad, mode="edge")
    out = np.zeros_like(n)
    for i in range(h):
        for j in range(w):
            out[i, j] = padded[i : i + k, j : j + k].mean()
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


def _metal_fill(base_rgb: Tuple[int, int, int], tex: np.ndarray, strength: float = 0.25) -> np.ndarray:
    h, w = tex.shape
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for c, v in enumerate(base_rgb):
        arr[:, :, c] = v * (1.0 - strength + strength * tex)
    return np.clip(arr, 0, 255).astype(np.uint8)


def render_machine_v02(
    condition: str,
    severity: float,
    seed: int,
    view: str = "front",
    lighting: str = "normal",
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
    r = random.Random(seed)
    np.random.seed(seed % (2**31 - 1))
    w, h = size

    # Lighting → background + brightness
    light_map = {
        "bright": (55, 1.25),
        "normal": (38, 1.0),
        "dark": (22, 0.72),
        "side_light": (32, 0.95),
        "soft_light": (42, 1.05),
    }
    bg_v, bright = light_map.get(lighting, (38, 1.0))
    tex = _noise(w, h, scale=12 + r.randint(0, 8), seed=seed)
    bg = _metal_fill((bg_v, bg_v, bg_v + 6), tex, 0.15)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img, "RGBA")

    # View shifts layout
    ox, oy, scale = 0, 0, 1.0
    if view == "side":
        ox, scale = int(w * 0.05), 0.92
    elif view == "angled":
        ox, oy, scale = int(w * 0.03), int(h * 0.02), 0.95
    elif view == "closeup":
        ox, oy, scale = -int(w * 0.08), -int(h * 0.05), 1.25

    def T(x: float, y: float) -> Tuple[int, int]:
        cx, cy = w / 2, h / 2
        return int(cx + (x - cx + ox) * scale), int(cy + (y - cy + oy) * scale)

    boxes: Dict[str, Tuple[int, int, int, int]] = {}

    def box_of(pts: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    # Housing
    hx1, hy1 = T(w * 0.12, h * 0.22)
    hx2, hy2 = T(w * 0.88, h * 0.78)
    draw.rounded_rectangle([hx1, hy1, hx2, hy2], radius=8, outline=(190, 195, 205, 255), width=3)
    # inner fill shade
    draw.rectangle([hx1 + 4, hy1 + 4, hx2 - 4, hy2 - 4], fill=(50, 55, 62, 90))
    boxes["housing"] = (hx1, hy1, hx2, hy2)

    # Screws on housing
    for sx, sy in [(0.14, 0.25), (0.14, 0.72), (0.84, 0.25), (0.84, 0.72)]:
        x, y = T(w * sx, h * sy)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(160, 160, 165, 255))

    # Motor
    missing_motor = condition == "component_missing" and severity >= 0.5
    if not missing_motor:
        mx1, my1 = T(w * 0.16, h * 0.32)
        mx2, my2 = T(w * 0.34, h * 0.68)
        draw.rectangle([mx1, my1, mx2, my2], fill=(75, 95, 115, 230), outline=(210, 215, 225, 255))
        # cooling fins
        for i in range(4):
            yy = my1 + int((my2 - my1) * (0.2 + 0.15 * i))
            draw.line([(mx1 + 6, yy), (mx2 - 6, yy)], fill=(40, 50, 60, 200), width=2)
        boxes["motor"] = (mx1, my1, mx2, my2)

    # Coupling
    if not (condition == "component_missing" and severity >= 0.75):
        cx1, cy1 = T(w * 0.34, h * 0.44)
        cx2, cy2 = T(w * 0.40, h * 0.56)
        draw.ellipse([cx1, cy1, cx2, cy2], fill=(130, 110, 70, 255), outline=(230, 210, 150, 255))
        boxes["coupling"] = (cx1, cy1, cx2, cy2)

    shaft_off = int(severity * 22) if condition == "misalignment" else 0
    sx1, sy1 = T(w * 0.40, h * 0.48 + shaft_off * 0.3)
    sx2, sy2 = T(w * 0.62, h * 0.52 + shaft_off * 0.3)
    # actual pixel offset for misalignment
    sy1 += shaft_off
    sy2 += shaft_off
    draw.rectangle([sx1, sy1, sx2, sy2], fill=(170, 172, 180, 255))
    boxes["shaft"] = (sx1, sy1 - 5, sx2, sy2 + 5)

    bx1, by1 = T(w * 0.50, h * 0.42)
    bx2, by2 = T(w * 0.58, h * 0.58)
    by1 += shaft_off
    by2 += shaft_off
    draw.ellipse([bx1, by1, bx2, by2], outline=(210, 170, 90, 255), width=4)
    draw.ellipse([bx1 + 6, by1 + 6, bx2 - 6, by2 - 6], outline=(120, 100, 50, 255), width=2)
    boxes["bearing"] = (bx1, by1, bx2, by2)

    px1, py1 = T(w * 0.62, h * 0.36)
    px2, py2 = T(w * 0.78, h * 0.64)
    py1 += shaft_off // 2
    py2 += shaft_off // 2
    draw.ellipse([px1, py1, px2, py2], outline=(150, 150, 160, 255), width=5)
    # spokes
    cxp, cyp = (px1 + px2) // 2, (py1 + py2) // 2
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        rr = (px2 - px1) // 2 - 4
        draw.line(
            [cxp, cyp, int(cxp + rr * math.cos(rad)), int(cyp + rr * math.sin(rad))],
            fill=(100, 100, 110, 200),
            width=2,
        )
    boxes["pulley"] = (px1, py1, px2, py2)

    belt_w = max(1, 5 - int(severity * 4)) if condition == "belt_degradation" else 5
    belt_shift = int(severity * 14) if condition == "belt_degradation" else 0
    by_mid = (py1 + py2) // 2 + belt_shift
    draw.arc([px1 - 22, by_mid - 42, px2 + 12, by_mid + 42], 200, 340, fill=(80, 80, 90, 255), width=belt_w)
    if condition == "belt_degradation" and severity > 0.3:
        # cracks
        for _ in range(int(severity * 5)):
            ax = r.randint(px1, max(px1 + 1, px2))
            draw.line([(ax, by_mid - 10), (ax + 4, by_mid + 8)], fill=(40, 30, 30, 255), width=1)
    boxes["belt"] = (px1 - 22, by_mid - 42, px2 + 12, by_mid + 42)

    pnx1, pny1 = T(w * 0.72, h * 0.24)
    pnx2, pny2 = T(w * 0.86, h * 0.36)
    draw.rectangle([pnx1, pny1, pnx2, pny2], fill=(45, 55, 65, 255), outline=(190, 200, 210, 255))
    draw.rectangle([pnx1 + 8, pny1 + 8, pnx1 + 18, pny1 + 16], fill=(40, 180, 90, 255))  # status LED
    boxes["panel"] = (pnx1, pny1, pnx2, pny2)

    if condition == "surface_damage" and severity > 0:
        for _ in range(int(4 + severity * 12)):
            x = int(w * (0.2 + 0.5 * r.random()))
            y = int(h * (0.3 + 0.4 * r.random()))
            rr = int(3 + severity * 14)
            draw.ellipse([x, y, x + rr, y + rr // 2], fill=(35, 28, 25, 200))

    if condition == "structural_change" and severity > 0.2:
        draw.line(
            [T(w * 0.22, h * 0.30), T(w * 0.70, h * 0.68)],
            fill=(200, 70, 70, 220),
            width=max(2, int(2 + severity * 3)),
        )

    if condition == "obstruction" and severity > 0:
        ox1 = int(w * (0.28 + 0.15 * severity))
        oy1 = int(h * 0.38)
        ox2 = ox1 + int(w * 0.12 * max(0.3, severity))
        oy2 = oy1 + int(h * 0.22)
        draw.rectangle([ox1, oy1, ox2, oy2], fill=(25, 25, 28, 230))

    # hard_negative: strong lighting/blur/shadow without structural defect
    if condition == "hard_negative":
        severity = 0.0  # label as non-defect for training semantics

    annotations = []
    for cls, bb in boxes.items():
        if cls not in COMPONENT_CLASSES:
            continue
        annotations.append(
            {
                "class": cls,
                "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                "condition": condition if condition != "hard_negative" else "normal",
                "severity": float(severity) if condition not in ("normal", "hard_negative") else 0.0,
            }
        )

    # Photometric pipeline
    img = img.convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(bright * (0.9 + r.random() * 0.2))
    img = ImageEnhance.Contrast(img).enhance(0.85 + r.random() * 0.35)
    if lighting == "side_light":
        arr = np.asarray(img).astype(np.float32)
        ramp = np.linspace(0.7, 1.15, w)[None, :, None]
        arr = np.clip(arr * ramp, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    if condition == "hard_negative":
        if r.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5 + r.random()))
        img = ImageEnhance.Brightness(img).enhance(0.55 + r.random() * 0.9)
    elif r.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4 + r.random() * 0.8))

    angle = r.uniform(-5, 5)
    if condition == "misalignment":
        angle += severity * 6
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(bg_v, bg_v, bg_v + 6))

    meta = {
        "condition": condition,
        "severity": float(severity) if condition != "hard_negative" else 0.0,
        "synthetic": True,
        "view": view,
        "lighting": lighting,
        "camera_angle": round(angle, 2),
        "machine_id": "machine_001",
        "dataset_version": "industrial_dataset_v0.2",
        "hard_negative": condition == "hard_negative",
    }
    return img, annotations, meta
