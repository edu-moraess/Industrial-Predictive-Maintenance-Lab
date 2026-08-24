"""Orchestrates image/video visual inspection."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vision.anomaly import baseline_anomaly_score, difference_map
from vision.config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_STRIDE
from vision.detector import ObjectDetector
from vision.preprocessing import decode_image_bytes
from vision.schemas import Detection, FrameResult, InspectionReport
from vision.video import iter_sampled_frames, video_info
from vision.visualization import draw_detections


def _status_from_anomaly(score: Optional[float], n_det: int) -> Tuple[str, str]:
    if score is None:
        if n_det == 0:
            return (
                "INCONCLUSIVE",
                "No baseline and no detections. Insufficient visual evidence for a conclusion.",
            )
        return (
            "OBSERVED",
            "Objects were detected. No baseline comparison was provided; "
            "detection alone is not a defect diagnosis.",
        )
    if score >= 0.45:
        return (
            "WARNING",
            "High visual deviation relative to the reference image. "
            "Recommend physical inspection of the asset.",
        )
    if score >= 0.25:
        return (
            "ATTENTION",
            "Moderate visual deviation from the reference. Monitor and inspect if needed.",
        )
    return (
        "NOMINAL",
        "Low visual deviation relative to the reference under the heuristic method.",
    )


class VisionInspectionService:
    def __init__(self, confidence: float = DEFAULT_CONFIDENCE):
        self.detector = ObjectDetector(confidence=confidence)
        self.confidence = confidence

    @property
    def model_available(self) -> bool:
        return self.detector.available

    def model_status(self) -> str:
        return self.detector.status_message

    def inspect_image(
        self,
        image_bytes: bytes,
        baseline_bytes: Optional[bytes] = None,
    ) -> Tuple[InspectionReport, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns report, original_bgr, annotated_bgr, difference_map_or_None.
        """
        t0 = time.perf_counter()
        image = decode_image_bytes(image_bytes)
        baseline = decode_image_bytes(baseline_bytes) if baseline_bytes else None

        detections = self.detector.detect(image) if self.detector.available else []
        annotated = draw_detections(image, detections)

        anom_score, anom_method = baseline_anomaly_score(image, baseline)
        diff = difference_map(image, baseline) if baseline is not None else None

        avg_conf = None
        if detections:
            avg_conf = round(sum(d.confidence for d in detections) / len(detections), 4)

        status, rec = _status_from_anomaly(anom_score, len(detections))
        notes: List[str] = []
        if not self.detector.available:
            notes.append(f"Detector unavailable: {self.detector.status_message}")
        notes.append(
            "YOLO COCO classes are generic (person, bottle, ...). "
            "They are not industrial part labels unless a custom model is provided."
        )
        if anom_score is not None:
            notes.append(f"Visual anomaly method: {anom_method}")

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
        return report, image, annotated, diff

    def inspect_video(
        self,
        video_path: str,
        stride: int = DEFAULT_FRAME_STRIDE,
        use_tracking: bool = True,
        baseline_bytes: Optional[bytes] = None,
    ) -> Tuple[InspectionReport, Optional[np.ndarray], Optional[np.ndarray], List[FrameResult], Dict[str, Any]]:
        """
        Sample frames, detect/track, temporal anomaly vs baseline or first frame.
        Returns report, sample_original, sample_annotated, frame_results, trajectories.
        """
        t0 = time.perf_counter()
        info = video_info(video_path)
        baseline = decode_image_bytes(baseline_bytes) if baseline_bytes else None

        frame_results: List[FrameResult] = []
        all_dets: List[Detection] = []
        anomaly_series: List[float] = []
        trajectories: Dict[str, List[Tuple[float, float, float]]] = {}
        sample_orig = None
        sample_ann = None

        for fi, ts, frame in iter_sampled_frames(video_path, stride=stride):
            if use_tracking and self.detector.available:
                dets = self.detector.detect_and_track(frame, persist=True)
            elif self.detector.available:
                dets = self.detector.detect(frame)
            else:
                dets = []

            ref = baseline if baseline is not None else (sample_orig if sample_orig is not None else None)
            score, _method = baseline_anomaly_score(frame, ref)
            if score is not None:
                anomaly_series.append(score)

            fr = FrameResult(
                frame_index=fi,
                timestamp_s=round(ts, 3),
                detections=dets,
                anomaly_score=score,
            )
            frame_results.append(fr)
            all_dets.extend(dets)

            for d in dets:
                if d.track_id is not None:
                    key = f"{d.class_name}#{d.track_id}"
                    x1, y1, x2, y2 = d.bbox_xyxy
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    trajectories.setdefault(key, []).append((ts, cx, cy))

            if sample_orig is None:
                sample_orig = frame
                sample_ann = draw_detections(frame, dets)

        avg_conf = None
        if all_dets:
            avg_conf = round(sum(d.confidence for d in all_dets) / len(all_dets), 4)

        # Unique tracks
        unique_ids = {d.track_id for d in all_dets if d.track_id is not None}

        mean_anom = round(float(np.mean(anomaly_series)), 4) if anomaly_series else None
        status, rec = _status_from_anomaly(mean_anom, len(all_dets))

        notes = [
            f"Frames analyzed: {len(frame_results)} (stride={stride})",
            "Apparent motion from tracking is visual displacement, not calibrated vibration.",
        ]
        if not self.detector.available:
            notes.append(f"Detector unavailable: {self.detector.status_message}")

        metrics = {
            **info,
            "frames_analyzed": len(frame_results),
            "sampling_stride": stride,
            "detections_total": len(all_dets),
            "unique_track_ids": len(unique_ids),
            "anomaly_series": anomaly_series,
            "anomaly_timestamps": [fr.timestamp_s for fr in frame_results if fr.anomaly_score is not None],
        }

        report = InspectionReport(
            input_type="video",
            model_available=self.detector.available,
            model_name=self.detector.model_name if self.detector.available else "N/A",
            processing_ms=round((time.perf_counter() - t0) * 1000, 2),
            detections=all_dets[:50],  # cap list in report
            objects_detected=len(all_dets),
            average_confidence=avg_conf,
            anomaly_score=mean_anom,
            anomaly_method="baseline_mean_absdiff (heuristic)" if mean_anom is not None else None,
            visual_status=status,
            recommendation=rec,
            notes=notes,
            video_metrics=metrics,
            frame_results=frame_results,
        )
        return report, sample_orig, sample_ann, frame_results, trajectories
