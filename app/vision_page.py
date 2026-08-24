"""
Streamlit UI for Computer Vision Inspection.
Independent from the sensor ML Operations pipeline.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from app.theme import apply_industrial_plotly_theme
from vision.config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_STRIDE, SUPPORTED_IMAGE_EXT, SUPPORTED_VIDEO_EXT
from vision.detector import ObjectDetector
from vision.inspection import VisionInspectionService
from vision.model_loader import dependency_status
from vision.video import save_upload_to_temp
from vision.visualization import bgr_to_rgb


@st.cache_resource
def _cached_detector() -> ObjectDetector:
    return ObjectDetector()


def _get_service(confidence: float) -> VisionInspectionService:
    det = _cached_detector()
    det.set_confidence(confidence)
    return VisionInspectionService(confidence=confidence, detector=det)


def render_computer_vision() -> None:
    st.markdown(
        """
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:4px 0 0 0;">
                Experimental visual inspection laboratory \u00b7 Baseline YOLO (COCO) \u00b7 Not industrial certification
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
        key="cv_input_mode",
    )

    st.markdown("#### CONFIGURATION")
    c1, c2, c3 = st.columns(3)
    with c1:
        conf = st.slider("Confidence threshold", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05, key="cv_conf")
    with c2:
        stride = st.select_slider(
            "Frame sampling (video)", options=[1, 2, 5, 10], value=DEFAULT_FRAME_STRIDE, key="cv_stride"
        )
    with c3:
        use_tracking = st.checkbox("Tracking (video)", value=True, key="cv_track")

    deps = dependency_status()
    service = _get_service(conf)

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
                <p style="color:#9A9FA8;font-size:0.8rem;margin:8px 0 0 0;">
                    ultralytics={deps['ultralytics']} \u00b7 opencv={deps['opencv']} \u00b7 pillow={deps['pillow']}<br/><br/>
                    <strong>If you see libGL.so.1 errors:</strong><br/>
                    <code>pip uninstall opencv-python opencv-contrib-python -y</code><br/>
                    <code>pip install opencv-python-headless ultralytics pillow</code><br/><br/>
                    Then <strong>restart Streamlit</strong>. Image baseline comparison works with Pillow even without YOLO.
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
                <p style="color:#9A9FA8;">Optional / environment dependent. Prefer Image upload if camera fails.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cam = st.camera_input("Capture (if browser permits)")
        if cam is not None and st.button("Run Inspection", type="primary", key="cv_run_cam"):
            with st.spinner("Analyzing image..."):
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
            st.error("INVALID INPUT \u2014 unsupported image type.")
            return
        if st.button("Run Inspection", type="primary", key="cv_run_img"):
            with st.spinner("Analyzing image..."):
                _run_image(service, up.getvalue(), baseline_bytes)
        return

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
        st.error("INVALID INPUT \u2014 unsupported video type.")
        return
    if not deps["opencv"]:
        st.warning(
            "Video analysis needs OpenCV. Fix libGL with: "
            "`pip uninstall opencv-python -y && pip install opencv-python-headless`"
        )
    if st.button("Run Inspection", type="primary", key="cv_run_vid"):
        path = save_upload_to_temp(up.getvalue(), ext)
        try:
            with st.spinner("Analyzing video (sampled frames)..."):
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
        msg = str(exc)
        st.error(f"Inspection failed: {msg}")
        if "libGL" in msg:
            st.info(
                "libGL fix:\n"
                "```\n"
                "pip uninstall opencv-python opencv-contrib-python -y\n"
                "pip install opencv-python-headless ultralytics pillow\n"
                "```\n"
                "Then restart Streamlit. Or use **Image** upload with Pillow-only path."
            )
        return

    st.success("Inspection completed.")
    st.markdown("#### RESULT")
    cols = st.columns(3 if diff is not None else 2)
    cols[0].image(bgr_to_rgb(original), caption="ORIGINAL", use_container_width=True)
    cols[1].image(bgr_to_rgb(annotated), caption="DETECTION", use_container_width=True)
    if diff is not None:
        cols[2].image(
            bgr_to_rgb(diff),
            caption="VISUAL ANOMALY MAP (vs baseline)",
            use_container_width=True,
        )
        st.caption(
            "Anomaly map shows relative difference to the reference image \u2014 not a mechanical failure label."
        )

    _render_report_cards(report)
    _render_class_charts(report)
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
        msg = str(exc)
        st.error(f"Video inspection failed: {msg}")
        if "libGL" in msg:
            st.info(
                "pip uninstall opencv-python -y && pip install opencv-python-headless && restart Streamlit"
            )
        return

    st.success("Video analysis completed.")
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
        fig = go.Figure(
            go.Scatter(x=times, y=series, mode="lines+markers", line=dict(color="#D4A84F", width=1.5))
        )
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Anomaly score", yaxis=dict(range=[0, 1]))
        apply_industrial_plotly_theme(fig, height=240)
        st.plotly_chart(fig, use_container_width=True)

    if trajectories:
        st.markdown("#### OBJECT TRAJECTORY (APPARENT MOTION)")
        keys = list(trajectories.keys())
        choice = st.selectbox("Object", keys, key="cv_traj_obj")
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

    _render_class_charts(report)
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


def _render_class_charts(report) -> None:
    if not report.detections:
        return
    counts = Counter(d.class_name for d in report.detections)
    confs = [d.confidence for d in report.detections]
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### DETECTIONS BY CLASS")
        fig = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color="#D4A84F"))
        apply_industrial_plotly_theme(fig, height=220)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("#### CONFIDENCE DISTRIBUTION")
        fig2 = go.Figure(go.Histogram(x=confs, nbinsx=10, marker_color="#4CAF78"))
        fig2.update_layout(xaxis_title="Confidence", yaxis_title="Count")
        apply_industrial_plotly_theme(fig2, height=220)
        st.plotly_chart(fig2, use_container_width=True)


def _render_detection_table(report) -> None:
    if not report.detections:
        st.markdown(
            """
            <div class="ind-card">
                <div class="ind-card-header">NO OBJECTS DETECTED</div>
                <p style="color:#9A9FA8;">Try another image, lower confidence, or use baseline comparison.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown("#### DETECTED OBJECTS")
    rows = [
        {
            "class": d.class_name,
            "confidence": d.confidence,
            "track_id": d.track_id if d.track_id is not None else "\u2014",
            "bbox": tuple(round(x, 1) for x in d.bbox_xyxy),
        }
        for d in report.detections[:40]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_context(report) -> None:
    st.markdown("#### VISUAL ANALYSIS")
    st.markdown(
        f"""
        <div class="ind-card">
            <p style="color:#F2F2F2;margin:0 0 8px 0;">{report.recommendation}</p>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:0;">
                Method: {report.anomaly_method or "N/A"} \u00b7 Model: {report.model_name}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for note in report.notes:
        st.caption(note)
