"""Vision module unit tests (detector may be unavailable without ultralytics)."""

import numpy as np

from vision.anomaly import baseline_anomaly_score, changed_area_ratio, difference_map
from vision.model_loader import dependency_status, load_yolo_model
from vision.preprocessing import ensure_bgr, resize_max_side
from vision.schemas import Detection
from vision.visualization import draw_detections


def test_dependency_status_dict():
    d = dependency_status()
    assert "ultralytics" in d
    assert "opencv" in d


def test_baseline_anomaly_identical():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    score, method = baseline_anomaly_score(img, img)
    assert score is not None
    assert score < 0.1
    assert "heuristic" in method


def test_baseline_anomaly_different():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    score, _ = baseline_anomaly_score(a, b)
    assert score is not None
    assert score > 0.4


def test_baseline_none():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    score, method = baseline_anomaly_score(img, None)
    assert score is None
    assert "no baseline" in method


def test_changed_area_ratio():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    r = changed_area_ratio(a, b)
    assert r is not None
    assert r > 0.5


def test_difference_map_shape():
    a = np.zeros((40, 40, 3), dtype=np.uint8)
    b = np.ones((40, 40, 3), dtype=np.uint8) * 100
    m = difference_map(a, b)
    assert m.ndim == 3


def test_draw_detections_no_crash():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = [Detection(class_name="person", confidence=0.9, bbox_xyxy=(10, 10, 80, 100))]
    out = draw_detections(img, dets)
    assert out.shape == img.shape


def test_resize_max_side():
    img = np.zeros((2000, 1000, 3), dtype=np.uint8)
    out = resize_max_side(ensure_bgr(img), max_side=640)
    assert max(out.shape[0], out.shape[1]) <= 640


def test_load_yolo_returns_tuple():
    model, err = load_yolo_model()
    # Either model works or a clear error string
    assert (model is not None and err is None) or (model is None and isinstance(err, str))
