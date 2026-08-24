"""
Experimental visual anomaly vs baseline.

Primary metric: mean absolute difference on normalized grayscale.
Secondary: simple structural similarity (OpenCV-only implementation).
HEURISTIC only — not a trained industrial defect classifier.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

METHOD_NAME = "baseline_absdiff+ssim (heuristic)"


def _to_gray_u8(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    import cv2

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _to_gray_f32(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    return _to_gray_u8(img, size).astype(np.float32) / 255.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM on float images in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / (den + 1e-12))


def baseline_anomaly_score(
    current_bgr: np.ndarray,
    baseline_bgr: Optional[np.ndarray],
) -> Tuple[Optional[float], str]:
    """
    score in [0,1]: 0 ~ similar to baseline, 1 ~ strong visual deviation.
    """
    if baseline_bgr is None:
        return None, "no baseline provided"

    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    absdiff = float(np.abs(a - b).mean())
    ssim = _ssim(a, b)
    # Combine: high absdiff or low SSIM => higher anomaly
    score = float(np.clip(0.6 * (absdiff * 4.0) + 0.4 * (1.0 - ssim), 0.0, 1.0))
    return round(score, 4), METHOD_NAME


def difference_map(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
) -> np.ndarray:
    """BGR heatmap of absolute difference (fixed 256 canvas, for display)."""
    import cv2

    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    d = np.abs(a - b)
    d_u8 = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)


def changed_area_ratio(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
    threshold: float = 0.15,
) -> Optional[float]:
    """Fraction of pixels with absdiff above threshold (0-1)."""
    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    mask = np.abs(a - b) > threshold
    return round(float(mask.mean()), 4)
