"""Image / frame preprocessing helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from vision.config import MAX_IMAGE_SIDE


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("empty image")
    if image.ndim == 2:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resize_max_side(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    new_w, new_h = int(w * scale), int(h * scale)
    import cv2

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_image_bytes(data: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return resize_max_side(ensure_bgr(img))


def image_size(image: np.ndarray) -> Tuple[int, int]:
    h, w = image.shape[:2]
    return w, h
