"""
Object detection wrapper.

Uses Ultralytics YOLOv8n (COCO) when installed.
Industrial part classes (bearing, motor, ...) are NOT in COCO —
detections are limited to the model's trained vocabulary.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from vision.config import DEFAULT_CONFIDENCE, YOLO_MODEL_NAME
from vision.schemas import Detection

_HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO  # type: ignore

    _HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None  # type: ignore


class ObjectDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME, confidence: float = DEFAULT_CONFIDENCE):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self._load_error: Optional[str] = None
        if _HAS_ULTRALYTICS:
            try:
                self._model = YOLO(model_name)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                self._model = None
        else:
            self._load_error = (
                "ultralytics not installed. "
                "pip install ultralytics opencv-python-headless"
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def status_message(self) -> str:
        if self.available:
            return f"READY ({self.model_name})"
        return f"NOT AVAILABLE — {self._load_error}"

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.available:
            return []

        results = self._model.predict(
            source=image_bgr,
            conf=self.confidence,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r0 = results[0]
        names = r0.names or {}
        boxes = r0.boxes
        if boxes is None:
            return out
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            x1, y1, x2, y2 = map(float, xyxy[:4])
            class_name = str(names.get(cls_id, f"class_{cls_id}"))
            out.append(
                Detection(
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def detect_and_track(self, image_bgr: np.ndarray, persist: bool = True) -> List[Detection]:
        """Run YOLO tracking on a single frame (ByteTrack via ultralytics)."""
        if not self.available:
            return []
        try:
            results = self._model.track(
                source=image_bgr,
                conf=self.confidence,
                persist=persist,
                verbose=False,
            )
        except Exception:
            return self.detect(image_bgr)

        out: List[Detection] = []
        if not results:
            return out
        r0 = results[0]
        names = r0.names or {}
        boxes = r0.boxes
        if boxes is None:
            return out
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            x1, y1, x2, y2 = map(float, xyxy[:4])
            class_name = str(names.get(cls_id, f"class_{cls_id}"))
            tid = None
            if box.id is not None:
                tid = int(box.id.item())
            out.append(
                Detection(
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    track_id=tid,
                )
            )
        return out
