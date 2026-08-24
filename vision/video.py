"""Video helpers — OpenCV optional; image path does not import this for decode."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Generator, Tuple

import numpy as np

from vision.input_io import iter_video_frames as _iter
from vision.model_loader import opencv_available


def save_upload_to_temp(data: bytes, ext: str) -> str:
    suffix = ext if str(ext).startswith(".") else f".{ext}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def video_info(path: str) -> Dict[str, Any]:
    if not opencv_available():
        return {"available": False}
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"available": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    duration = (n / fps) if (fps and n) else None
    cap.release()
    return {
        "available": True,
        "fps": fps,
        "frame_count": n,
        "width": w,
        "height": h,
        "duration_s": round(duration, 3) if duration else None,
    }


def iter_sampled_frames(
    path: str, stride: int = 5
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    yield from _iter(path, stride=stride)
