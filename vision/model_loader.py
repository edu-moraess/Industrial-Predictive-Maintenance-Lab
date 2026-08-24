"""
Safe model loading for the Computer Vision lab.

YOLO weights are loaded once per process. Streamlit should wrap this with
@st.cache_resource at the UI layer; this module also keeps a process-level cache.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from vision.config import YOLO_MODEL_NAME

_ULTRALYTICS_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None
try:
    from ultralytics import YOLO  # type: ignore

    _ULTRALYTICS_AVAILABLE = True
except ImportError as exc:
    YOLO = None  # type: ignore
    _IMPORT_ERROR = str(exc)

_OPENCV_AVAILABLE = False
_OPENCV_ERROR: Optional[str] = None
try:
    import cv2  # noqa: F401

    _OPENCV_AVAILABLE = True
except ImportError as exc:
    _OPENCV_ERROR = str(exc)

_model_cache: dict[str, Any] = {}
_model_error: dict[str, str] = {}


def ultralytics_available() -> bool:
    return _ULTRALYTICS_AVAILABLE


def opencv_available() -> bool:
    return _OPENCV_AVAILABLE


def dependency_status() -> dict:
    return {
        "ultralytics": _ULTRALYTICS_AVAILABLE,
        "opencv": _OPENCV_AVAILABLE,
        "ultralytics_error": _IMPORT_ERROR,
        "opencv_error": _OPENCV_ERROR,
    }


def load_yolo_model(model_name: str = YOLO_MODEL_NAME) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load YOLO weights once. Returns (model, error_message).
    On success error_message is None.
    """
    if not _ULTRALYTICS_AVAILABLE:
        return None, (
            "ultralytics not installed. "
            "Run: pip install ultralytics opencv-python-headless"
        )
    if not _OPENCV_AVAILABLE:
        return None, (
            "opencv not installed. "
            "Run: pip install opencv-python-headless"
        )
    if model_name in _model_cache:
        return _model_cache[model_name], None
    if model_name in _model_error:
        return None, _model_error[model_name]
    try:
        model = YOLO(model_name)
        _model_cache[model_name] = model
        return model, None
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load {model_name}: {exc}"
        _model_error[model_name] = msg
        return None, msg
