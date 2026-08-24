"""
Streamlit UI for Computer Vision Inspection.
Independent from the sensor ML Operations pipeline.
"""

from __future__ import annotations

import os
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from app.theme import apply_industrial_plotly_theme
from vision.config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_STRIDE, SUPPORTED_IMAGE_EXT, SUPPORTED_VIDEO_EXT
from vision.inspection import VisionInspectionService
from vision.video import save_upload_to_temp
from vision.visualization import bgr_to_rgb


def render_computer_vision() -> None:
    st.markdown(
        """
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:4px 0 0 0;">
                Experimental visual inspection laboratory. Not industrial defect certification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input mode",
        ["Image", "Video", "Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        conf = st.slider("Confidence threshold", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05)
    with c2:
        stride = st.select_slider("Frame sampling (video)", options=[1, 2, 5, 10], value=DEFAULT_FRAME_STRIDE)
    with c3:
        use_tracking = st.checkbox("Tracking (video)", value=True)

    service = VisionInspectionService(confidence=conf)

    st.markdown("#### VISION MODEL")
    if service.model_available:
        st.markdown(
            f'<div class="ind-card"><span class="badge badge-success">READY</span> '
            f'<span style="color:#9A9FA8;margin-left:8px;">{service.model_status()}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ind-card">
                <div class="ind-card-header">VISION MODEL NOT AVAILABLE</div>
                <p style="color:#9A9FA8;font-size:0.85rem;">{service.model_status()}</p>
                <p style="color:#9A9FA8;font-size:0.8rem;">
                    Install optional deps: <code>pip install ultralytics opencv-python-headless</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### INPUT")
    baseline_file = st.file_uploader(
        "Optional reference (baseline) image",
        type=["jpg", "jpeg", "png", "webp"],
        key="cv_baseline",
    )
    baseline_bytes = baseline_file.getvalue() if baseline_file else None

    if mode == "Camera":
        st.markdown(
            """
            <div class="ind-card">
                <div class="ind-card-header">CAMERA INPUT</div>
                <p style="color:#9A9FA8;">Optional / environment dependent. Use Image or Video upload in this lab build.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cam = st.camera_input("Capture (if browser permits)")
        if cam is not None and st.button("Run Inspection", type="primary"):
            _run_image(service, cam.getvalue(), baseline_bytes)
        return

    if mode == "Image":
        up = st.file_uploader(
            "Upload machine photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="cv_image",
        )
        if up is None:
            st.markdown(
                """
                <div class="ind-card" style="text-align:center;padding:20px;">
                    <div class="ind-card-header">NO IMAGE</div>
                    <p style="color:#9A9FA8;">Upload a photo to start visual inspection.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        ext = os.path.splitext(up.name)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXT:
            st.error("INVALID INPUT — unsupported image type.")
            return
        if st.button("Run Inspection", type="primary"):
            _run_image(service, up.getvalue(), baseline_bytes)
        return

    # Video
    up = st.file_uploader(
        "Upload machine video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="cv_video",
    )
    if up is None:
        st.markdown(
            """
            <div class="ind-card" style="text-align:center;padding:20px;">
                <div class="ind-card-header">NO VIDEO</div>
                <p style="color:#9A9FA8;">Upload a video to start temporal visual analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    ext = os.path.splitext(up.name)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXT:
        st.error("INVALID INPUT — unsupported video type.")
        return
    if st.button("Run Inspection", type="primary"):
        path = save_upload_to_temp(up.getvalue(), ext)
        try:
            _run_video(service, path, stride, use_tracking, baseline_bytes)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def _run_image(service: VisionInspectionService, data: bytes, baseline: Optional[bytes]) -> None:
    try:
        report, original, annotated, diff = service.inspect_image(data, baseline)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Inspection failed: {exc}")
        return

    st.markdown("#### RESULT")
    cols = st.columns(3 if diff is not None else 2)
    cols[0].image(bgr_to_rgb(original), caption="ORIGINAL", use_container_width=True)
    cols[1].image(bgr_to_rgb(annotated), caption="DETECTION", use_container_width=True)
    if diff is not None:
        cols[2].image(bgr_to_rgb(diff), caption="DIFFERENCE MAP (vs baseline)", use_container_width=True)

    _render_report_cards(report)
    _render_detection_table(report)
    _render_context(report)


def _run_video(
    service: VisionInspectionService,
    path: str,
    stride: int,
    use_tracking: bool,
    baseline: Optional[bytes],
) -> None:
    try:
        report, sample_o, sample_a, frame_results, trajectories = service.inspect_video(
            path, stride=stride, use_tracking=use_tracking, baseline_bytes=baseline
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Video inspection failed: {exc}")
        return

    st.markdown("#### RESULT")
    if sample_o is not None and sample_a is not None:
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(sample_o), caption="SAMPLE FRAME (ORIGINAL)", use_container_width=True)
        c2.image(bgr_to_rgb(sample_a), caption="SAMPLE FRAME (DETECTION)", use_container_width=True)

    _render_report_cards(report)

    vm = report.video_metrics or {}
    st.markdown("#### VIDEO METRICS")
    mcols = st.columns(4)
    mcols[0].markdown(
        f'<div class="ind-card"><div class="ind-card-header">DURATION</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{vm.get("duration_s") if vm.get("duration_s") is not None else "N/A"} s</div></div>',
        unsafe_allow_html=True,
    )
    mcols[1].markdown(
        f'<div class="ind-card"><div class="ind-card-header">FPS</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{vm.get("fps") if vm.get("fps") is not None else "N/A"}</div></div>',
        unsafe_allow_html=True,
    )
    mcols[2].markdown(
        f'<div class="ind-card"><div class="ind-card-header">FRAMES ANALYZED</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{vm.get("frames_analyzed", "N/A")}</div></div>',
        unsafe_allow_html=True,
    )
    mcols[3].markdown(
        f'<div class="ind-card"><div class="ind-card-header">TRACK IDs</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{vm.get("unique_track_ids", "N/A")}</div></div>',
        unsafe_allow_html=True,
    )

    series = vm.get("anomaly_series") or []
    times = vm.get("anomaly_timestamps") or []
    if series and times and len(series) == len(times):
        st.markdown("#### VISUAL ANOMALY TIMELINE")
        st.caption("Heuristic baseline/first-frame difference — not a calibrated defect score.")
        fig = go.Figure(
            go.Scatter(x=times, y=series, mode="lines+markers", line=dict(color="#D4A84F", width=1.5))
        )
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Anomaly score", yaxis=dict(range=[0, 1]))
        apply_industrial_plotly_theme(fig, height=240)
        st.plotly_chart(fig, use_container_width=True)

    if trajectories:
        st.markdown("#### OBJECT TRAJECTORY (APPARENT MOTION)")
        st.caption("Pixel-space path from tracking. Not physical vibration.")
        keys = list(trajectories.keys())
        choice = st.selectbox("Object", keys)
        pts = trajectories[choice]
        fig_t = go.Figure(
            go.Scatter(
                x=[p[1] for p in pts],
                y=[p[2] for p in pts],
                mode="lines+markers",
                line=dict(color="#4CAF78", width=1.5),
            )
        )
        fig_t.update_layout(xaxis_title="x (px)", yaxis_title="y (px)", yaxis=dict(autorange="reversed"))
        apply_industrial_plotly_theme(fig_t, height=260)
        st.plotly_chart(fig_t, use_container_width=True)

    _render_detection_table(report)
    _render_context(report)


def _render_report_cards(report) -> None:
    st.markdown("#### VISUAL INSPECTION")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f'<div class="ind-card"><div class="ind-card-header">OBJECTS</div>'
        f'<div class="ind-card-value">{report.objects_detected}</div></div>',
        unsafe_allow_html=True,
    )
    avg = f"{report.average_confidence:.0%}" if report.average_confidence is not None else "N/A"
    c2.markdown(
        f'<div class="ind-card"><div class="ind-card-header">AVG CONFIDENCE</div>'
        f'<div class="ind-card-value" style="font-size:1.2rem;">{avg}</div></div>',
        unsafe_allow_html=True,
    )
    anom = f"{report.anomaly_score:.3f}" if report.anomaly_score is not None else "N/A"
    c3.markdown(
        f'<div class="ind-card"><div class="ind-card-header">VISUAL ANOMALY</div>'
        f'<div class="ind-card-value" style="font-size:1.2rem;">{anom}</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="ind-card"><div class="ind-card-header">STATUS</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{report.visual_status}</div></div>',
        unsafe_allow_html=True,
    )
    c5.markdown(
        f'<div class="ind-card"><div class="ind-card-header">TIME</div>'
        f'<div class="ind-card-value" style="font-size:1rem;">{report.processing_ms:.0f} ms</div></div>',
        unsafe_allow_html=True,
    )


def _render_detection_table(report) -> None:
    if not report.detections:
        st.markdown(
            """
            <div class="ind-card">
                <div class="ind-card-header">NO OBJECTS DETECTED</div>
                <p style="color:#9A9FA8;">Try another image/frame or lower the confidence threshold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown("#### DETECTED OBJECTS")
    rows = []
    for d in report.detections[:40]:
        rows.append(
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "track_id": d.track_id if d.track_id is not None else "—",
                "bbox": tuple(round(x, 1) for x in d.bbox_xyxy),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_context(report) -> None:
    st.markdown("#### VISUAL ANALYSIS")
    st.markdown(
        f"""
        <div class="ind-card">
            <p style="color:#F2F2F2;margin:0 0 8px 0;">{report.recommendation}</p>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:0;">
                Method: {report.anomaly_method or "N/A"} · Model: {report.model_name}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for note in report.notes:
        st.caption(note)
    st.markdown(
        """
        <p style="color:#9A9FA8;font-size:0.75rem;">
        Language policy: detections are <strong>DETECTED</strong> objects from the vision model.
        Visual anomaly scores are <strong>INFERRED</strong> heuristics vs baseline.
        Nothing here is a <strong>CONFIRMED</strong> industrial defect diagnosis.
        </p>
        """,
        unsafe_allow_html=True,
    )
