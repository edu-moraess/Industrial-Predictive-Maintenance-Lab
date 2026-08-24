"""Typed structures for vision inspection results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    detections: List[Detection] = field(default_factory=list)
    anomaly_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class InspectionReport:
    input_type: str
    model_available: bool
    model_name: str
    processing_ms: float
    detections: List[Detection]
    objects_detected: int
    average_confidence: Optional[float]
    anomaly_score: Optional[float]
    anomaly_method: Optional[str]
    visual_status: str
    recommendation: str
    notes: List[str] = field(default_factory=list)
    video_metrics: Optional[Dict[str, Any]] = None
    frame_results: Optional[List[FrameResult]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
