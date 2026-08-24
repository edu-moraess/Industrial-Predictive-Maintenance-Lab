"""
Object detection wrapper (YOLO / Ultralytics when available).

COCO classes only unless custom weights are provided.
Industrial labels (bearing, motor, ...) are NOT assumed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from vision.config import DEFAULT_CONFIDENCE, YOLO_MODEL_NAME
from vision.model_loader import load_yolo_model, ultralytics_available
from vision.schemas import Detection


class ObjectDetector:
    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self._model, self._load_error = load_yolo_model(model_name)

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def status_message(self) -> str:
        if self.available:
            return f"READY ({self.model_name}) — experimental COCO baseline"
        return f"NOT AVAILABLE — {self._load_error}"

    def set_confidence(self, confidence: float) -> None:
        self.confidence = float(confidence)

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        results = self._model.predict(
            source=image_bgr,
            conf=self.confidence,
            verbose=False,
        )
        return self._boxes_to_detections(results)

    def detect_and_track(self, image_bgr: np.ndarray, persist: bool = True) -> List[Detection]:
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
        return self._boxes_to_detections(results)

    def _boxes_to_detections(self, results) -> List[Detection]:
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
            tid: Optional[int] = None
            if getattr(box, "id", None) is not None:
                try:
                    tid = int(box.id.item())
                except Exception:
                    tid = None
            out.append(
                Detection(
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    track_id=tid,
                )
            )
        return out


def detector_backend_ready() -> bool:
    return ultralytics_available()
