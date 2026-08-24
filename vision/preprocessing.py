"""Image / frame preprocessing — OpenCV preferred, Pillow fallback (no libGL)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from vision.config import MAX_IMAGE_SIDE
from vision.model_loader import opencv_available, pillow_available


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("empty image")
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3].copy()
    return image


def resize_max_side(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    new_w, new_h = int(w * scale), int(h * scale)
    if opencv_available():
        import cv2

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pillow fallback
    from PIL import Image

    rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
    pil = Image.fromarray(rgb.astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = np.asarray(pil)
    if out.ndim == 3 and out.shape[2] == 3:
        out = out[:, :, ::-1].copy()  # back to BGR convention
    return out


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode to BGR uint8 ndarray. Prefer OpenCV; fall back to Pillow."""
    if not data:
        raise ValueError("Empty image bytes")

    if opencv_available():
        import cv2

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return resize_max_side(ensure_bgr(img))

    if pillow_available():
        import io

        from PIL import Image

        pil = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.asarray(pil)
        bgr = rgb[:, :, ::-1].copy()
        return resize_max_side(ensure_bgr(bgr))

    raise RuntimeError(
        "Cannot decode image: OpenCV and Pillow both unavailable. "
        "Install: pip install pillow opencv-python-headless"
    )


def image_size(image: np.ndarray) -> Tuple[int, int]:
    h, w = image.shape[:2]
    return w, h
