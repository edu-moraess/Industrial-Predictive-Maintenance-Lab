"""
Experimental visual anomaly scoring via baseline comparison.

Method: mean absolute pixel difference after resize/grayscale normalization.
This is a HEURISTIC, not a trained industrial defect classifier.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

METHOD_NAME = "baseline_mean_absdiff (heuristic)"


def _to_gray_f32(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    import cv2

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def baseline_anomaly_score(
    current_bgr: np.ndarray,
    baseline_bgr: Optional[np.ndarray],
) -> Tuple[Optional[float], str]:
    """
    Returns (score in [0,1], method_name).
    score ~0 similar to baseline; higher = more visual deviation.
    If baseline is None, returns (None, explanation).
    """
    if baseline_bgr is None:
        return None, "no baseline provided"

    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    diff = np.abs(a - b)
    score = float(np.clip(diff.mean() * 4.0, 0.0, 1.0))  # scale mild diffs
    return round(score, 4), METHOD_NAME


def difference_map(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
) -> np.ndarray:
    """Return a BGR heatmap-style absolute difference for visualization."""
    import cv2

    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    d = np.abs(a - b)
    d_u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
    return colored
