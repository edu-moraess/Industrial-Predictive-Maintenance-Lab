"""
Input adapters for Streamlit uploads.

IMAGE path never requires OpenCV (Pillow + NumPy only).
VIDEO path uses OpenCV when available; otherwise graceful unavailable.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from vision.model_loader import opencv_available

SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class ImagePayload:
    """RGB uint8 array + metadata for UI."""

    rgb: np.ndarray
    width: int
    height: int
    format: str
    size_bytes: int
    mode_src: str


@dataclass
class VideoPayload:
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_s: Optional[float] = None
    cleanup_path: bool = True


class InputError(Exception):
    """User-facing input failure (message is safe for UI)."""


def _ext(name: str) -> str:
    return os.path.splitext(name or "")[1].lower()


def load_image_from_bytes(data: bytes, filename: str = "upload") -> ImagePayload:
    if not data:
        raise InputError("Unable to read image. Empty file.")

    ext = _ext(filename)
    if ext and ext not in SUPPORTED_IMAGE:
        raise InputError("Unable to read image. Please try JPG, PNG, WEBP or BMP.")

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise InputError("Unable to read image. Pillow is not installed.") from exc

    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        mode_src = pil.mode
        fmt = (pil.format or ext.replace(".", "").upper() or "UNKNOWN")
        if pil.mode in ("RGBA", "LA", "P"):
            pil = pil.convert("RGBA")
            background = Image.new("RGB", pil.size, (0, 0, 0))
            if pil.mode == "RGBA":
                background.paste(pil, mask=pil.split()[-1])
            else:
                background.paste(pil)
            pil = background
        else:
            pil = pil.convert("RGB")
        rgb = np.asarray(pil, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise InputError("Unable to read image. Unsupported pixel layout.")
        h, w = rgb.shape[:2]
        return ImagePayload(
            rgb=rgb,
            width=w,
            height=h,
            format=str(fmt),
            size_bytes=len(data),
            mode_src=mode_src,
        )
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Unable to read image. Please try JPG, PNG, WEBP or BMP.") from expc if False else exc  # noqa: E501


def load_image_from_upload(uploaded_file) -> ImagePayload:
    """Streamlit UploadedFile or file-like with .getvalue/.read and .name."""
    if uploaded_file is None:
        raise InputError("Unable to read image. No file provided.")
    name = getattr(uploaded_file, "name", "upload.jpg")
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
    return load_image_from_bytes(data, name)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, ::-1].copy()


def save_video_temp(data: bytes, ext: str) -> str:
    if not data:
        raise InputError("Unable to decode video. Empty file.")
    suffix = ext if ext.startswith(".") else f".{ext}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    return path


def load_video_from_upload(uploaded_file) -> VideoPayload:
    if uploaded_file is None:
        raise InputError("Unable to decode video. No file provided.")
    if not opencv_available():
        raise InputError(
            "Video processing unavailable. Image inspection still works. "
            "Install opencv-python-headless in the Streamlit environment to enable video."
        )
    name = getattr(uploaded_file, "name", "upload.mp4")
    ext = _ext(name)
    if ext and ext not in SUPPORTED_VIDEO:
        raise InputError("Unable to decode video. Please try MP4, MOV, AVI or WEBM.")

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    path = save_video_temp(data, ext or ".mp4")

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        try:
            os.remove(path)
        except OSError:
            pass
        raise InputError("Unable to decode video. Please try MP4.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    duration = None
    if fps and n and fps > 0:
        duration = round(n / fps, 3)
    cap.release()

    return VideoPayload(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=n,
        duration_s=duration,
        cleanup_path=True,
    )


def iter_video_frames(
    path: str, stride: int = 5
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_s, BGR uint8). Requires OpenCV."""
    if not opencv_available():
        raise InputError("Video processing unavailable.")
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InputError("Unable to decode video. Please try MP4.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    stride = max(1, int(stride))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                ts = idx / fps
                yield idx, ts, frame
            idx += 1
    finally:
        cap.release()


def cleanup_video(payload: VideoPayload) -> None:
    if payload.cleanup_path and payload.path:
        try:
            os.remove(payload.path)
        except OSError:
            pass
