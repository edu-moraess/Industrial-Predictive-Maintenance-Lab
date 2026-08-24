"""Video frame sampling utilities."""

from __future__ import annotations

import os
import tempfile
from typing import Generator, List, Optional, Tuple

import numpy as np

from vision.config import MAX_VIDEO_FRAMES_ANALYZED
from vision.preprocessing import resize_max_side, ensure_bgr


def open_video_capture(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    return cap


def video_info(path: str) -> dict:
    import cv2

    cap = open_video_capture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else None
        return {
            "fps": fps if fps > 0 else None,
            "frame_count": frame_count if frame_count > 0 else None,
            "width": w or None,
            "height": h or None,
            "duration_s": round(duration, 2) if duration is not None else None,
        }
    finally:
        cap.release()


def iter_sampled_frames(
    path: str,
    stride: int = 5,
    max_frames: int = MAX_VIDEO_FRAMES_ANALYZED,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_s, bgr_image)."""
    import cv2

    cap = open_video_capture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    idx = 0
    yielded = 0
    try:
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                frame = resize_max_side(ensure_bgr(frame))
                ts = idx / fps
                yield idx, ts, frame
                yielded += 1
            idx += 1
    finally:
        cap.release()


def save_upload_to_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path
