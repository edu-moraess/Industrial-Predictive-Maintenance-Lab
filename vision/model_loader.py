"""
Safe dependency detection and YOLO loading (headless-friendly).

libGL.so.1 errors almost always mean the GUI wheel was installed:
  opencv-python  (pulls libGL)
instead of:
  opencv-python-headless

Fix on the SAME interpreter Streamlit uses:
  python -m pip uninstall opencv-python opencv-contrib-python -y
  python -m pip install opencv-python-headless ultralytics pillow
  python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

from vision.config import YOLO_MODEL_NAME

# ---------------------------------------------------------------------------
# Dependency probes (never raise)
# ---------------------------------------------------------------------------
_ULTRALYTICS_AVAILABLE = False
_ULTRALYTICS_VERSION: Optional[str] = None
_IMPORT_ERROR: Optional[str] = None
try:
    from ultralytics import YOLO  # type: ignore
    import ultralytics as _ultralytics_mod

    _ULTRALYTICS_AVAILABLE = True
    _ULTRALYTICS_VERSION = getattr(_ultralytics_mod, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001
    YOLO = None  # type: ignore
    _IMPORT_ERROR = str(exc)

_OPENCV_AVAILABLE = False
_OPENCV_VERSION: Optional[str] = None
_OPENCV_ERROR: Optional[str] = None
try:
    import cv2

    _OPENCV_AVAILABLE = True
    _OPENCV_VERSION = getattr(cv2, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001
    _OPENCV_ERROR = str(exc)
    if "libGL" in str(exc):
        _OPENCV_ERROR = (
            f"{exc} | Cause: GUI OpenCV wheel needs libGL. "
            "Fix: python -m pip uninstall opencv-python opencv-contrib-python -y && "
            "python -m pip install opencv-python-headless"
        )

_PIL_AVAILABLE = False
_PIL_VERSION: Optional[str] = None
_PIL_ERROR: Optional[str] = None
try:
    from PIL import Image  # noqa: F401
    import PIL

    _PIL_AVAILABLE = True
    _PIL_VERSION = getattr(PIL, "__version__", "unknown")
except Exception as exc:  # noqa: BLE001
    _PIL_ERROR = str(exc)

_NUMPY_AVAILABLE = False
_NUMPY_VERSION: Optional[str] = None
try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
    _NUMPY_VERSION = getattr(_np, "__version__", "unknown")
except Exception:
    pass

_model_cache: dict[str, Any] = {}
_model_error: dict[str, str] = {}


def ultralytics_available() -> bool:
    return _ULTRALYTICS_AVAILABLE


def opencv_available() -> bool:
    return _OPENCV_AVAILABLE


def pillow_available() -> bool:
    return _PIL_AVAILABLE


def dependency_status() -> dict:
    return {
        "ultralytics": _ULTRALYTICS_AVAILABLE,
        "opencv": _OPENCV_AVAILABLE,
        "pillow": _PIL_AVAILABLE,
        "numpy": _NUMPY_AVAILABLE,
        "ultralytics_error": _IMPORT_ERROR,
        "opencv_error": _OPENCV_ERROR,
        "pillow_error": _PIL_ERROR,
    }


def get_vision_environment_status() -> Dict[str, Any]:
    """Structured status for the Streamlit diagnostics panel."""
    yolo_model, yolo_err = load_yolo_model()
    image_pipeline_ok = _PIL_AVAILABLE or _OPENCV_AVAILABLE
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy": {
            "available": _NUMPY_AVAILABLE,
            "version": _NUMPY_VERSION,
        },
        "pillow": {
            "available": _PIL_AVAILABLE,
            "version": _PIL_VERSION,
            "error": _PIL_ERROR,
        },
        "opencv": {
            "available": _OPENCV_AVAILABLE,
            "version": _OPENCV_VERSION,
            "error": _OPENCV_ERROR,
            "note": "Use opencv-python-headless only (not opencv-python) on servers",
        },
        "ultralytics": {
            "available": _ULTRALYTICS_AVAILABLE,
            "version": _ULTRALYTICS_VERSION,
            "error": _IMPORT_ERROR,
        },
        "yolo_model": {
            "available": yolo_model is not None,
            "name": YOLO_MODEL_NAME if yolo_model is not None else None,
            "error": yolo_err,
        },
        "capabilities": {
            "image_upload": image_pipeline_ok,
            "baseline_comparison": image_pipeline_ok,
            "anomaly_map": image_pipeline_ok,
            "object_detection": yolo_model is not None,
            "video": _OPENCV_AVAILABLE,
            "tracking": yolo_model is not None and _OPENCV_AVAILABLE,
            "browser_camera": True,  # st.camera_input is browser-side
            "opencv_webcam": False,  # we never use cv2.VideoCapture(0)
        },
        "install_hint": (
            "python -m pip uninstall opencv-python opencv-contrib-python -y && "
            "python -m pip install opencv-python-headless ultralytics pillow && "
            "python -m streamlit run app/dashboard.py"
        ),
    }


def load_yolo_model(model_name: str = YOLO_MODEL_NAME) -> Tuple[Optional[Any], Optional[str]]:
    if not _ULTRALYTICS_AVAILABLE:
        return None, (
            "Ultralytics not installed in this Python. "
            f"Interpreter: {sys.executable}. "
            "Install: python -m pip install ultralytics opencv-python-headless"
        )
    if not _OPENCV_AVAILABLE:
        return None, (
            "OpenCV import failed (often libGL from opencv-python GUI wheel). "
            f"Interpreter: {sys.executable}. "
            f"Detail: {_OPENCV_ERROR}"
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
        if "libGL" in str(exc):
            msg += (
                " | Uninstall opencv-python; install opencv-python-headless "
                f"into {sys.executable}"
            )
        _model_error[model_name] = msg
        return None, msg
