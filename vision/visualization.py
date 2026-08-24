"""Clean detection overlays — minimal labels on image."""

from __future__ import annotations

from typing import List

import numpy as np

from vision.schemas import Detection

# Restrained palette aligned with industrial UI accent
BOX_COLOR = (79, 168, 212)  # BGR muted amber-ish → actually amber-ish in BGR: (79,168,212) is ok
TEXT_COLOR = (242, 242, 242)


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    import cv2

    out = image_bgr.copy()
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


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
