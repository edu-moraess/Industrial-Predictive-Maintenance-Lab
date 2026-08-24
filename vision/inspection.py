"""Orchestrates image/video visual inspection."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vision.anomaly import (
    baseline_anomaly_score,
    changed_area_ratio,
    difference_map,
    motion_heatmap_from_frames,
)
from vision.config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_STRIDE
from vision.detector import ObjectDetector
from vision.events import VisualEvent, build_image_events, build_video_events
from vision.preprocessing import decode_image_bytes
from vision.roi import count_detections_in_rois, default_rois, draw_rois
from vision.schemas import Detection, FrameResult, InspectionReport
from vision.video import iter_sampled_frames, video_info
from vision.visualization import draw_detections


def _visual_status(
    *,
    detector_online: bool,
    n_objects: int,
    anomaly_score: Optional[float],
) -> Tuple[str, str]:
    if anomaly_score is not None and anomaly_score >= 0.45:
        return (
            "REFERENCE DEVIATION",
            "High visual difference relative to the reference image. Physical inspection recommended.",
        )
    if anomaly_score is not None and anomaly_score < 0.15 and detector_online:
        return (
            "REFERENCE MATCH",
            "Low visual difference vs reference under the heuristic comparison method.",
        )
    if anomaly_score is not None and anomaly_score >= 0.15:
        return (
            "VISUAL DIFFERENCE DETECTED",
            "Moderate visual difference vs reference. Monitor the area of change.",
        )
    if not detector_online and anomaly_score is None:
        return (
            "DETECTOR OFFLINE",
            "Object detector not loaded. Upload a baseline to enable reference comparison, "
            "or install vision deps in this Python environment.",
        )
    if detector_online and n_objects == 0:
        return (
            "NO OBJECTS DETECTED",
            "Detector ran successfully; no boxes above the confidence threshold "
            "(COCO vocabulary may not match industrial parts).",
        )
    if detector_online and n_objects > 0:
        return (
            "ANALYZED",
            "Detections produced by the baseline vision model.",
        )
    return (
        "INSUFFICIENT EVIDENCE",
        "Not enough signal for a visual conclusion.",
    )


class VisionInspectionService:
    def __init__(self, confidence: float = DEFAULT_CONFIDENCE, detector: Optional[ObjectDetector] = None):
        self.detector = detector or ObjectDetector(confidence=confidence)
        self.confidence = confidence
        if detector is not None:
            self.detector.set_confidence(confidence)

    @property
    def model_available(self) -> bool:
        return self.detector.available

    def model_status(self) -> str:
        return self.detector.status_message

    def set_confidence(self, confidence: float) -> None:
        self.confidence = confidence
        self.detector.set_confidence(confidence)

    def inspect_image(
        self,
        image_bytes: bytes,
        baseline_bytes: Optional[bytes] = None,
        show_rois: bool = False,
    ) -> Tuple[InspectionReport, np.ndarray, np.ndarray, Optional[np.ndarray], List[VisualEvent], Dict[str, Any]]:
        t0 = time.perf_counter()
        image = decode_image_bytes(image_bytes)
        baseline = decode_image_bytes(baseline_bytes) if baseline_bytes else None

        detections = self.detector.detect(image) if self.detector.available else []
        annotated = draw_detections(image, detections)
        rois = default_rois()
        if show_rois:
            annotated = draw_rois(annotated, rois)

        h, w = image.shape[:2]
        roi_counts = count_detections_in_rois(detections, rois, w, h) if detections else {}

        anom_score, anom_method = baseline_anomaly_score(image, baseline)
        diff = difference_map(image, baseline) if baseline is not None else None
        area = changed_area_ratio(image, baseline) if baseline is not None else None

        avg_conf = None
        if detections:
            avg_conf = round(sum(d.confidence for d in detections) / len(detections), 4)

        status, rec = _visual_status(
            detector_online=self.detector.available,
            n_objects=len(detections),
            anomaly_score=anom_score,
        )
        events = build_image_events(
            n_objects=len(detections),
            detector_online=self.detector.available,
            anomaly_score=anom_score,
            changed_area=area,
        )

        notes = [
            "COCO baseline — generic classes. Domain weights needed for industrial components.",
        ]
        if area is not None:
            notes.append(f"Changed area ratio: {area:.1%}")

        report = InspectionReport(
            input_type="image",
            model_available=self.detector.available,
            model_name=self.detector.model_name if self.detector.available else "N/A",
            processing_ms=round((time.perf_counter() - t0) * 1000, 2),
            detections=detections,
            objects_detected=len(detections),
            average_confidence=avg_conf,
            anomaly_score=anom_score,
            anomaly_method=anom_method if anom_score is not None else None,
            visual_status=status,
            recommendation=rec,
            notes=notes,
        )
        extra = {"changed_area": area, "roi_counts": roi_counts, "events": [e.to_dict() for e in events]}
        return report, image, annotated, diff, events, extra

    def inspect_video(
        self,
        video_path: str,
        stride: int = DEFAULT_FRAME_STRIDE,
        use_tracking: bool = True,
        baseline_bytes: Optional[bytes] = None,
    ) -> Tuple[InspectionReport, Optional[np.ndarray], Optional[np.ndarray], List[FrameResult], Dict[str, Any], List[VisualEvent], Optional[np.ndarray]]:
        t0 = time.perf_counter()
        info = video_info(video_path)
        baseline = decode_image_bytes(baseline_bytes) if baseline_bytes else None

        frame_results: List[FrameResult] = []
        all_dets: List[Detection] = []
        anomaly_series: List[float] = []
        trajectories: Dict[str, List[Tuple[float, float, float]]] = {}
        sampled_frames: List[np.ndarray] = []
        sample_orig = None
        sample_ann = None

        for fi, ts, frame in iter_sampled_frames(video_path, stride=stride):
            sampled_frames.append(frame)
            if use_tracking and self.detector.available:
                dets = self.detector.detect_and_track(frame, persist=True)
            elif self.detector.available:
                dets = self.detector.detect(frame)
            else:
                dets = []

            ref = baseline if baseline is not None else (sample_orig if sample_orig is not None else None)
            score, _ = baseline_anomaly_score(frame, ref)
            if score is not None:
                anomaly_series.append(score)

            frame_results.append(
                FrameResult(frame_index=fi, timestamp_s=round(ts, 3), detections=dets, anomaly_score=score)
            )
            all_dets.extend(dets)
            for d in dets:
                if d.track_id is not None:
                    key = f"{d.class_name}#{d.track_id}"
                    x1, y1, x2, y2 = d.bbox_xyxy
                    trajectories.setdefault(key, []).append((ts, (x1 + x2) / 2.0, (y1 + y2) / 2.0))

            if sample_orig is None:
                sample_orig = frame
                sample_ann = draw_detections(frame, dets)

        avg_conf = None
        if all_dets:
            avg_conf = round(sum(d.confidence for d in all_dets) / len(all_dets), 4)
        unique_ids = {d.track_id for d in all_dets if d.track_id is not None}
        mean_anom = round(float(np.mean(anomaly_series)), 4) if anomaly_series else None
        status, rec = _visual_status(
            detector_online=self.detector.available,
            n_objects=len(all_dets),
            anomaly_score=mean_anom,
        )
        events = build_video_events(
            frames=len(frame_results),
            n_detections=len(all_dets),
            unique_tracks=len(unique_ids),
            mean_anomaly=mean_anom,
        )
        motion = motion_heatmap_from_frames(sampled_frames)

        metrics = {
            **info,
            "frames_analyzed": len(frame_results),
            "sampling_stride": stride,
            "detections_total": len(all_dets),
            "unique_track_ids": len(unique_ids),
            "anomaly_series": anomaly_series,
            "anomaly_timestamps": [fr.timestamp_s for fr in frame_results if fr.anomaly_score is not None],
            "trajectories": trajectories,
        }

        report = InspectionReport(
            input_type="video",
            model_available=self.detector.available,
            model_name=self.detector.model_name if self.detector.available else "N/A",
            processing_ms=round((time.perf_counter() - t0) * 1000, 2),
            detections=all_dets[:50],
            objects_detected=len(all_dets),
            average_confidence=avg_conf,
            anomaly_score=mean_anom,
            anomaly_method="aligned_absdiff+ssim (heuristic)" if mean_anom is not None else None,
            visual_status=status,
            recommendation=rec,
            notes=["Apparent motion heatmap is not thermal imagery."],
            video_metrics=metrics,
            frame_results=frame_results,
        )
        return report, sample_orig, sample_ann, frame_results, metrics, events, motion
