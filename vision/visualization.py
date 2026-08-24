"""Clean detection overlays — OpenCV if available, else NumPy boxes."""

from __future__ import annotations

from typing import List

import numpy as np

from vision.model_loader import opencv_available
from vision.schemas import Detection

BOX_COLOR = (79, 168, 212)
TEXT_COLOR = (242, 242, 242)


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    if opencv_available():
        import cv2

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = det.class_name
            if det.track_id is not None:
                label = f"#{det.track_id} {label}"
            label = f"{label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y1 - 6)
            cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), (23, 26, 33), -1)
            cv2.putText(
                out,
                label,
                (x1 + 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
        return out

    # NumPy-only boxes (no text)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1] - 1, x2), min(out.shape[0] - 1, y2)
        out[y1 : y1 + 2, x1:x2] = BOX_COLOR
        out[y2 - 2 : y2, x1:x2] = BOX_COLOR
        out[y1:y2, x1 : x1 + 2] = BOX_COLOR
        out[y1:y2, x2 - 2 : x2] = BOX_COLOR
    return out


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return image_bgr[:, :, ::-1].copy()
