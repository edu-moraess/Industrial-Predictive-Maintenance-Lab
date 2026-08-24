"""Structured visual events (experimental)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VisualEvent:
    event_type: str
    severity: str  # low | medium | high
    evidence: str
    location: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_image_events(
    *,
    n_objects: int,
    detector_online: bool,
    anomaly_score: Optional[float],
    changed_area: Optional[float],
) -> List[VisualEvent]:
    events: List[VisualEvent] = []
    if not detector_online:
        events.append(
            VisualEvent(
                event_type="detector_offline",
                severity="low",
                evidence="YOLO weights not loaded in this process",
                detail="Baseline comparison may still run",
            )
        )
    elif n_objects == 0:
        events.append(
            VisualEvent(
                event_type="no_objects_detected",
                severity="low",
                evidence="Detector ran; zero boxes above confidence threshold",
            )
        )
    else:
        events.append(
            VisualEvent(
                event_type="objects_detected",
                severity="low",
                evidence=f"{n_objects} detection(s) above threshold",
            )
        )

    if anomaly_score is not None:
        if anomaly_score >= 0.45:
            events.append(
                VisualEvent(
                    event_type="reference_deviation",
                    severity="medium",
                    evidence=f"Anomaly score {anomaly_score:.3f}",
                    detail="Visual difference vs reference — not a mechanical failure label",
                )
            )
        elif anomaly_score < 0.15:
            events.append(
                VisualEvent(
                    event_type="reference_match",
                    severity="low",
                    evidence=f"Anomaly score {anomaly_score:.3f}",
                )
            )

    if changed_area is not None and changed_area >= 0.2:
        events.append(
            VisualEvent(
                event_type="large_visual_change",
                severity="medium",
                evidence=f"Changed area ratio {changed_area:.1%}",
            )
        )
    return events


def build_video_events(
    *,
    frames: int,
    n_detections: int,
    unique_tracks: int,
    mean_anomaly: Optional[float],
) -> List[VisualEvent]:
    events: List[VisualEvent] = [
        VisualEvent(
            event_type="frames_analyzed",
            severity="low",
            evidence=f"{frames} sampled frames",
        )
    ]
    if unique_tracks > 0:
        events.append(
            VisualEvent(
                event_type="tracks_present",
                severity="low",
                evidence=f"{unique_tracks} unique track id(s)",
            )
        )
    if mean_anomaly is not None and mean_anomaly >= 0.4:
        events.append(
            VisualEvent(
                event_type="temporal_visual_deviation",
                severity="medium",
                evidence=f"Mean anomaly {mean_anomaly:.3f}",
            )
        )
    if n_detections == 0:
        events.append(
            VisualEvent(
                event_type="no_objects_detected",
                severity="low",
                evidence="No detections across sampled frames",
            )
        )
    return events
