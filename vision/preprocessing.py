"""Image preprocessing — Pillow-first; OpenCV only as optional accelerator."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from vision.config import MAX_IMAGE_SIDE
from vision.input_io import load_image_from_bytes, rgb_to_bgr
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
    from PIL import Image

    # assume BGR
    rgb = image[:, :, ::-1] if image.ndim == 3 else image
    pil = Image.fromarray(rgb.astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = np.asarray(pil)
    if out.ndim == 3:
        out = out[:, :, ::-1].copy()
    return out


def decode_image_bytes(data: bytes) -> np.ndarray:
    """
    Decode upload bytes to BGR uint8 for the vision engine.
    Uses Pillow only (no OpenCV dependency).
    """
    if not data:
        raise ValueError("Empty image bytes")
    if not pillow_available():
        # last resort: OpenCV imdecode
        if opencv_available():
            import cv2

            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image")
            return resize_max_side(ensure_bgr(img))
        raise RuntimeError("Cannot decode image: Pillow and OpenCV unavailable")

    payload = load_image_from_bytes(data)
    bgr = rgb_to_bgr(payload.rgb)
    return resize_max_side(ensure_bgr(bgr))


def image_size(image: np.ndarray) -> Tuple[int, int]:
    h, w = image.shape[:2]
    return w, h
