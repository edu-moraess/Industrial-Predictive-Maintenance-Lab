"""Rectangular regions of interest (optional / future industrial zones)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ROI:
    name: str
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float

    def absolute(self, w: int, h: int) -> Tuple[int, int, int, int]:
        return (
            int(self.x1 * w),
            int(self.y1 * h),
            int(self.x2 * w),
            int(self.y2 * h),
        )


def default_rois() -> List[ROI]:
    """Placeholder zones for future industrial labeling."""
    return [
        ROI("ZONE_A", 0.05, 0.05, 0.45, 0.45),
        ROI("ZONE_B", 0.55, 0.05, 0.95, 0.45),
        ROI("ZONE_C", 0.05, 0.55, 0.95, 0.95),
    ]


def draw_rois(image_bgr: np.ndarray, rois: List[ROI]) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    try:
        import cv2

        for roi in rois:
            x1, y1, x2, y2 = roi.absolute(w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (154, 159, 168), 1)
            cv2.putText(
                out,
                roi.name,
                (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (154, 159, 168),
                1,
                cv2.LINE_AA,
            )
    except Exception:
        pass
    return out


def count_detections_in_rois(detections, rois: List[ROI], w: int, h: int) -> dict:
    """Count detection centers falling inside each ROI."""
    counts = {r.name: 0 for r in rois}
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for roi in rois:
            ax1, ay1, ax2, ay2 = roi.absolute(w, h)
            if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
                counts[roi.name] += 1
    return counts
