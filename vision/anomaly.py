"""Baseline visual comparison + anomaly map (independent of YOLO)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from vision.model_loader import opencv_available

METHOD_NAME = "aligned_absdiff+ssim (heuristic)"


def _to_gray_u8(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    if img.ndim == 3:
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
    ys = (np.linspace(0, h - 1, th)).astype(np.int32)
    xs = (np.linspace(0, w - 1, tw)).astype(np.int32)
    return gray[ys][:, xs]


def _align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize to common canvas; optional ECC alignment when OpenCV is present."""
    size = (256, 256)
    ga = _to_gray_u8(a, size)
    gb = _to_gray_u8(b, size)
    if not opencv_available():
        return ga, gb
    try:
        import cv2

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        cv2.findTransformECC(
            ga.astype(np.float32) / 255.0,
            gb.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            1,
        )
        gb = cv2.warpAffine(
            gb, warp, (size[0], size[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except Exception:
        pass
    return ga, gb


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a, mu_b = af.mean(), bf.mean()
    sigma_a, sigma_b = af.var(), bf.var()
    sigma_ab = ((af - mu_a) * (bf - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / (den + 1e-12))


def baseline_anomaly_score(
    current_bgr: np.ndarray,
    baseline_bgr: Optional[np.ndarray],
) -> Tuple[Optional[float], str]:
    if baseline_bgr is None:
        return None, "no baseline provided"
    ga, gb = _align_pair(current_bgr, baseline_bgr)
    af = ga.astype(np.float32) / 255.0
    bf = gb.astype(np.float32) / 255.0
    absdiff = float(np.abs(af - bf).mean())
    ssim = _ssim(ga, gb)
    score = float(np.clip(0.6 * (absdiff * 4.0) + 0.4 * (1.0 - ssim), 0.0, 1.0))
    return round(score, 4), METHOD_NAME


def difference_map(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
) -> np.ndarray:
    ga, gb = _align_pair(current_bgr, baseline_bgr)
    d = np.abs(ga.astype(np.float32) - gb.astype(np.float32)) / 255.0
    d_u8 = (np.clip(d / (d.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    if opencv_available():
        import cv2

        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)
    hmap = np.zeros((*d_u8.shape, 3), dtype=np.uint8)
    hmap[:, :, 2] = d_u8
    hmap[:, :, 1] = (d_u8.astype(np.uint16) // 2).astype(np.uint8)
    return hmap


def changed_area_ratio(
    current_bgr: np.ndarray,
    baseline_bgr: np.ndarray,
    threshold: float = 0.15,
) -> Optional[float]:
    ga, gb = _align_pair(current_bgr, baseline_bgr)
    d = np.abs(ga.astype(np.float32) - gb.astype(np.float32)) / 255.0
    return round(float((d > threshold).mean()), 4)


def motion_heatmap_from_frames(frames_bgr: list) -> Optional[np.ndarray]:
    """Accumulate frame-to-frame absdiff (apparent motion), not thermal."""
    if len(frames_bgr) < 2:
        return None
    acc = None
    prev = _to_gray_u8(frames_bgr[0], (320, 240)).astype(np.float32)
    for fr in frames_bgr[1:]:
        cur = _to_gray_u8(fr, (320, 240)).astype(np.float32)
        d = np.abs(cur - prev)
        acc = d if acc is None else acc + d
        prev = cur
    if acc is None:
        return None
    acc = acc / (acc.max() + 1e-6)
    u8 = (acc * 255).astype(np.uint8)
    if opencv_available():
        import cv2

        return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    hmap = np.zeros((*u8.shape, 3), dtype=np.uint8)
    hmap[:, :, 1] = u8
    return hmap
