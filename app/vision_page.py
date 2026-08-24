"""Computer Vision Inspection Lab UI — clean primary surface, diagnostics at bottom."""

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
from vision.model_loader import get_vision_environment_status
from vision.video import save_upload_to_temp
from vision.visualization import bgr_to_rgb


@st.cache_resource
def _cached_detector() -> ObjectDetector:
    return ObjectDetector()


def _get_service(confidence: float) -> VisionInspectionService:
    det = _cached_detector()
    det.set_confidence(confidence)
    return VisionInspectionService(confidence=confidence, detector=det)


def _engine_label(env: dict) -> tuple[str, str]:
    yolo = env["yolo_model"]["available"]
    img = env["capabilities"]["image_upload"]
    if yolo and img:
        return "READY", "badge-success"
    if img:
        return "PARTIALLY AVAILABLE", "badge-warning"
    return "OFFLINE", "badge-critical"


def render_computer_vision() -> None:
    env = get_vision_environment_status()
    caps = env["capabilities"]
    engine_txt, engine_badge = _engine_label(env)

    st.markdown(
        f"""
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:4px 0 0 0;">
                Industrial visual inspection laboratory \u00b7 Experimental \u00b7 Not defect certification
            </p>
        </div>
        <div class="ind-card" style="display:flex;flex-wrap:wrap;gap:16px;align-items:center;">
            <div><div class="ind-card-header">VISION ENGINE</div>
                <span class="badge {engine_badge}">{engine_txt}</span></div>
            <div><div class="ind-card-header">MODEL</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">
                {"YOLO baseline (COCO)" if env["yolo_model"]["available"] else "Detection offline"}</span></div>
            <div><div class="ind-card-header">DEVICE</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">AUTO / CPU</span></div>
            <div><div class="ind-card-header">BASELINE COMPARE</div>
                <span class="badge {'badge-success' if caps['baseline_comparison'] else 'badge-warning'}">
                {'ON' if caps['baseline_comparison'] else 'OFF'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "COCO baseline uses generic classes. Domain-trained weights are required for industrial component labels."
    )

    st.markdown("#### CONFIGURATION")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        conf = st.slider("Confidence", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05, key="cv_conf")
    with c2:
        stride = st.select_slider("Frame sampling", options=[1, 2, 5, 10], value=DEFAULT_FRAME_STRIDE, key="cv_stride")
    with c3:
        use_tracking = st.checkbox("Tracking", value=True, key="cv_track")
    with c4:
        show_rois = st.checkbox("Show ROI zones", value=False, key="cv_rois")

    model_choice = st.radio(
        "Vision model",
        ["YOLO baseline", "Custom YOLO (coming soon)", "Segmentation (coming soon)"],
        horizontal=True,
        key="cv_model_choice",
    )
    if model_choice != "YOLO baseline":
        st.caption("Only YOLO baseline is implemented in this lab version.")

    service = _get_service(conf)

    st.markdown("#### INPUT")
    mode = st.radio("Source", ["Image", "Video", "Camera"], horizontal=True, key="cv_input_mode")

    baseline_file = st.file_uploader(
        "Reference image (optional baseline)",
        type=["jpg", "jpeg", "png", "webp"],
        key="cv_baseline",
    )
    baseline_bytes = baseline_file.getvalue() if baseline_file else None

    if mode == "Image":
        if not caps["image_upload"]:
            st.warning("Image pipeline requires Pillow or OpenCV. See Advanced Diagnostics.")
            _render_diagnostics(env)
            return
        up = st.file_uploader("Machine image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cv_image")
        if up is None:
            st.info("Upload a machine photo to run inspection.")
            _render_diagnostics(env)
            return
        if os.path.splitext(up.name)[1].lower() not in SUPPORTED_IMAGE_EXT:
            st.error("Unsupported image type.")
            return
        if st.button("Run Inspection", type="primary", key="cv_run_img"):
            with st.spinner("Running visual inspection..."):
                _run_image(service, up.getvalue(), baseline_bytes, show_rois)

    elif mode == "Video":
        up = st.file_uploader("Machine video", type=["mp4", "mov", "avi", "mkv", "webm"], key="cv_video")
        if up is None:
            st.info("Upload a video for temporal analysis.")
            _render_diagnostics(env)
            return
        if not caps["video"]:
            st.warning("Video requires OpenCV headless. See Advanced Diagnostics for setup.")
            _render_diagnostics(env)
            return
        if os.path.splitext(up.name)[1].lower() not in SUPPORTED_VIDEO_EXT:
            st.error("Unsupported video type.")
            return
        if st.button("Run Inspection", type="primary", key="cv_run_vid"):
            path = save_upload_to_temp(up.getvalue(), os.path.splitext(up.name)[1].lower())
            try:
                with st.spinner("Analyzing sampled frames..."):
                    _run_video(service, path, stride, use_tracking, baseline_bytes)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    else:
        st.caption("Browser camera via Streamlit — not OpenCV webcam.")
        if not caps["image_upload"]:
            st.warning("Cannot decode camera frames without Pillow/OpenCV.")
            _render_diagnostics(env)
            return
        try:
            cam = st.camera_input("Snapshot")
        except Exception:
            st.info("Camera widget unavailable. Use Image upload.")
            _render_diagnostics(env)
            return
        if cam is None:
            st.caption("Waiting for browser snapshot.")
            _render_diagnostics(env)
            return
        if st.button("Run Inspection", type="primary", key="cv_run_cam"):
            with st.spinner("Analyzing snapshot..."):
                _run_image(service, cam.getvalue(), baseline_bytes, show_rois)

    _render_diagnostics(env)


def _run_image(service, data, baseline, show_rois) -> None:
    try:
        report, original, annotated, diff, events, extra = service.inspect_image(
            data, baseline, show_rois=show_rois
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Inspection could not complete.")
        st.caption(str(exc))
        return

    st.markdown("#### INSPECTION RESULT")
    n = 2 + (1 if diff is not None else 0)
    cols = st.columns(n)
    cols[0].image(bgr_to_rgb(original), caption="ORIGINAL", use_container_width=True)
    cols[1].image(bgr_to_rgb(annotated), caption="DETECTION OVERLAY", use_container_width=True)
    if diff is not None:
        cols[2].image(bgr_to_rgb(diff), caption="ANOMALY MAP", use_container_width=True)

    _metrics(report, extra.get("changed_area"))
    _events(events)
    if extra.get("roi_counts"):
        st.caption(f"ROI hits: {extra['roi_counts']}")
    _class_charts(report)
    _det_table(report)
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">INTERPRETATION</div>
            <p style="color:#F2F2F2;margin:0;">{report.recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for note in report.notes:
        st.caption(note)


def _run_video(service, path, stride, use_tracking, baseline) -> None:
    try:
        report, so, sa, _fr, metrics, events, motion = service.inspect_video(
            path, stride=stride, use_tracking=use_tracking, baseline_bytes=baseline
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Video inspection could not complete.")
        st.caption(str(exc))
        return

    st.markdown("#### INSPECTION RESULT")
    if so is not None and sa is not None:
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(so), caption="SAMPLE FRAME", use_container_width=True)
        c2.image(bgr_to_rgb(sa), caption="DETECTION OVERLAY", use_container_width=True)
    if motion is not None:
        st.image(bgr_to_rgb(motion), caption="MOTION HEATMAP (apparent motion, not thermal)", use_container_width=True)

    _metrics(report, None)
    mcols = st.columns(4)
    mcols[0].metric("Frames analyzed", metrics.get("frames_analyzed", 0))
    mcols[1].metric("Track IDs", metrics.get("unique_track_ids", 0))
    mcols[2].metric("Duration s", metrics.get("duration_s") if metrics.get("duration_s") is not None else "—")
    mcols[3].metric("FPS", metrics.get("fps") if metrics.get("fps") is not None else "—")

    series = metrics.get("anomaly_series") or []
    times = metrics.get("anomaly_timestamps") or []
    if series and len(series) == len(times):
        fig = go.Figure(go.Scatter(x=times, y=series, mode="lines+markers", line=dict(color="#D4A84F", width=1.5)))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Anomaly score", yaxis=dict(range=[0, 1]))
        apply_industrial_plotly_theme(fig, height=220)
        st.plotly_chart(fig, use_container_width=True)

    traj = metrics.get("trajectories") or {}
    if traj:
        st.markdown("#### OBJECT TRAJECTORY")
        key = st.selectbox("Track", list(traj.keys()), key="cv_traj")
        pts = traj[key]
        fig_t = go.Figure(
            go.Scatter(x=[p[1] for p in pts], y=[p[2] for p in pts], mode="lines+markers", line=dict(color="#4CAF78"))
        )
        fig_t.update_layout(xaxis_title="x", yaxis_title="y", yaxis=dict(autorange="reversed"))
        apply_industrial_plotly_theme(fig_t, height=240)
        st.plotly_chart(fig_t, use_container_width=True)

    _events(events)
    _class_charts(report)
    _det_table(report)


def _metrics(report, changed_area) -> None:
    st.markdown("#### METRICS")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Objects", report.objects_detected)
    c2.metric(
        "Mean conf.",
        f"{report.average_confidence:.2f}" if report.average_confidence is not None else "—",
    )
    c3.metric(
        "Anomaly",
        f"{report.anomaly_score:.3f}" if report.anomaly_score is not None else "—",
    )
    c4.metric("Changed area", f"{changed_area:.1%}" if changed_area is not None else "—")
    c5.metric("Time ms", f"{report.processing_ms:.0f}")
    st.markdown(
        f'<span class="badge badge-info">{report.visual_status}</span>',
        unsafe_allow_html=True,
    )


def _events(events) -> None:
    if not events:
        return
    st.markdown("#### VISUAL EVENTS")
    for e in events:
        st.markdown(
            f"- **{e.event_type}** · severity `{e.severity}` · {e.evidence}"
            + (f" · {e.detail}" if e.detail else "")
        )


def _class_charts(report) -> None:
    if not report.detections:
        return
    counts = Counter(d.class_name for d in report.detections)
    confs = [d.confidence for d in report.detections]
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()), marker_color="#D4A84F"))
        apply_industrial_plotly_theme(fig, height=200)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig2 = go.Figure(go.Histogram(x=confs, nbinsx=10, marker_color="#4CAF78"))
        apply_industrial_plotly_theme(fig2, height=200)
        st.plotly_chart(fig2, use_container_width=True)


def _det_table(report) -> None:
    if not report.detections:
        return
    st.markdown("#### DETECTIONS")
    st.dataframe(
        [
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "track_id": d.track_id if d.track_id is not None else "—",
            }
            for d in report.detections[:40]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_diagnostics(env: dict) -> None:
    with st.expander("Advanced Diagnostics", expanded=False):
        st.code(f"{env['python_executable']}\nPython {env['python_version']}")
        st.write(
            {
                "numpy": env["numpy"],
                "pillow": env["pillow"],
                "opencv": {k: env["opencv"][k] for k in ("available", "version")},
                "ultralytics": {k: env["ultralytics"][k] for k in ("available", "version")},
                "yolo_model": env["yolo_model"],
                "capabilities": env["capabilities"],
            }
        )
        if env["opencv"].get("error"):
            st.caption(env["opencv"]["error"])
        if env["ultralytics"].get("error"):
            st.caption(env["ultralytics"]["error"])
        st.markdown("Setup (same interpreter as Streamlit):")
        st.code(env["install_hint"])
