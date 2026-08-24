"""Vision module unit tests (no GPU required; detector may be unavailable)."""

import numpy as np

from vision.anomaly import baseline_anomaly_score
from vision.preprocessing import resize_max_side, ensure_bgr
from vision.schemas import Detection
from vision.visualization import draw_detections


def test_baseline_anomaly_identical():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    score, method = baseline_anomaly_score(img, img)
    assert score is not None
    assert score < 0.05
    assert "heuristic" in method


def test_baseline_anomaly_different():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    score, _ = baseline_anomaly_score(a, b)
    assert score is not None
    assert score > 0.5


def test_baseline_none():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    score, method = baseline_anomaly_score(img, None)
    assert score is None
    assert "no baseline" in method


def test_draw_detections_no_crash():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = [
        Detection(class_name="person", confidence=0.9, bbox_xyxy=(10, 10, 80, 100)),
    ]
    out = draw_detections(img, dets)
    assert out.shape == img.shape


def test_resize_max_side():
    img = np.zeros((2000, 1000, 3), dtype=np.uint8)
    out = resize_max_side(ensure_bgr(img), max_side=640)
    assert max(out.shape[0], out.shape[1]) <= 640
