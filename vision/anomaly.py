"""
Experimental visual anomaly vs baseline (OpenCV or pure NumPy/Pillow path).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vision.model_loader import opencv_available

METHOD_NAME = "baseline_absdiff+ssim (heuristic)"


def _to_gray_u8(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    if img.ndim == 3:
        # BGR weights approx
        gray = (
            0.114 * img[:, :, 0].astype(np.float32)
            + 0.587 * img[:, :, 1].astype(np.float32)
            + 0.299 * img[:, :, 2].astype(np.float32)
        ).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)

    h, w = gray.shape[:2]
    th, tw = size[1], size[0]
    if opencv_available():
        import cv2

        return cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)

    # nearest/bilinear via simple block (coarse but dependency-free)
    ys = (np.linspace(0, h - 1, th)).astype(np.int32)
    xs = (np.linspace(0, w - 1, tw)).astype(np.int32)
    return gray[ys][:, xs]


def _to_gray_f32(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    return _to_gray_u8(img, size).astype(np.float32) / 255.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
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
    if baseline_bgr is None:
        return None, "no baseline provided"

    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    absdiff = float(np.abs(a - b).mean())
    ssim = _ssim(a, b)
    score = float(np.clip(0.6 * (absdiff * 4.0) + 0.4 * (1.0 - ssim), 0.0, 1.0))
    return round(score, 4), METHOD_NAME


def difference_map(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
) -> np.ndarray:
    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    d = np.abs(a - b)
    d_u8 = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)

    if opencv_available():
        import cv2

        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    # Simple 3-channel heat without OpenCV colormap
    hmap = np.zeros((*d_u8.shape, 3), dtype=np.uint8)
    hmap[:, :, 2] = d_u8  # red channel
    hmap[:, :, 1] = (d_u8.astype(np.uint16) // 2).astype(np.uint8)
    return hmap


def changed_area_ratio(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
    threshold: float = 0.15,
) -> Optional[float]:
    a = _to_gray_f32(current_bgr)
    b = _to_gray_f32(baseline_bgr)
    mask = np.abs(a - b) > threshold
    return round(float(mask.mean()), 4)
