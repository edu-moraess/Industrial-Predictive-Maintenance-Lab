"""Computer Vision Inspection Lab — industrial model primary, COCO optional side path."""

from __future__ import annotations

import os
from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from app.theme import apply_industrial_plotly_theme
from vision.config import DEFAULT_CONFIDENCE, DEFAULT_FRAME_STRIDE, SUPPORTED_IMAGE_EXT, SUPPORTED_VIDEO_EXT
from vision.detector import ObjectDetector
from vision.industrial_model import IndustrialAnomalyModel, industrial_model_status
from vision.inspection import VisionInspectionService
from vision.model_loader import get_vision_environment_status
from vision.preprocessing import decode_image_bytes
from vision.video import save_upload_to_temp
from vision.visualization import bgr_to_rgb
from vision.anomaly import difference_map, changed_area_ratio


@st.cache_resource
def _cached_detector() -> ObjectDetector:
    return ObjectDetector()


@st.cache_resource
def _cached_industrial() -> IndustrialAnomalyModel | None:
    return IndustrialAnomalyModel.try_load()


def render_computer_vision() -> None:
    env = get_vision_environment_status()
    caps = env["capabilities"]
    ind_status = industrial_model_status()
    ind = _cached_industrial()

    if ind is not None:
        engine_txt, engine_badge = "READY", "badge-success"
        model_label = f"{ind_status.get('model_name', 'industrial_anomaly_v0.1')}"
    elif caps["image_upload"]:
        engine_txt, engine_badge = "PARTIALLY AVAILABLE", "badge-warning"
        model_label = "Not trained — generate dataset + train"
    else:
        engine_txt, engine_badge = "OFFLINE", "badge-critical"
        model_label = "Unavailable"

    st.markdown(
        f"""
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:4px 0 0 0;">
                Industrial synthetic domain \u00b7 Experimental \u00b7 Not industrial certification
            </p>
        </div>
        <div class="ind-card" style="display:flex;flex-wrap:wrap;gap:16px;align-items:center;">
            <div><div class="ind-card-header">VISION ENGINE</div>
                <span class="badge {engine_badge}">{engine_txt}</span></div>
            <div><div class="ind-card-header">INDUSTRIAL MODEL</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">{model_label}</span></div>
            <div><div class="ind-card-header">TASK</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">Visual anomaly (PCA residual)</span></div>
            <div><div class="ind-card-header">DATASET</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">
                {ind_status.get('dataset_version', 'industrial_dataset_v0.1') if ind else 'not generated'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Primary model is trained on lab-generated synthetic schematics (not COCO). "
        "Synthetic \u2192 real domain gap applies. Not a mechanical failure diagnosis."
    )

    if ind is None:
        st.info(
            "Train the industrial model once:\n\n"
            "```bash\n"
            "python -m dataset.generator\n"
            "python -m training.train\n"
            "python -m training.evaluate\n"
            "```\n"
            "Then restart Streamlit. Reference comparison still works without training."
        )

    st.markdown("#### CONFIGURATION")
    c1, c2, c3 = st.columns(3)
    with c1:
        conf = st.slider("Optional COCO conf (side tool)", 0.1, 0.9, DEFAULT_CONFIDENCE, 0.05, key="cv_conf")
    with c2:
        stride = st.select_slider("Frame sampling", options=[1, 2, 5, 10], value=DEFAULT_FRAME_STRIDE, key="cv_stride")
    with c3:
        use_coco = st.checkbox("Also run COCO detector (optional)", value=False, key="cv_coco")

    st.markdown("#### INPUT")
    mode = st.radio("Source", ["Image", "Video", "Camera"], horizontal=True, key="cv_input_mode")
    baseline_file = st.file_uploader(
        "Reference image (optional pixel baseline)",
        type=["jpg", "jpeg", "png", "webp"],
        key="cv_baseline",
    )
    baseline_bytes = baseline_file.getvalue() if baseline_file else None

    if mode == "Image":
        if not caps["image_upload"]:
            st.warning("Need Pillow or OpenCV for images.")
            _diagnostics(env, ind_status)
            return
        up = st.file_uploader("Machine image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cv_image")
        if up is None:
            st.info("Upload a machine image (prefer synthetic dataset samples or domain photos).")
            _diagnostics(env, ind_status)
            return
        if os.path.splitext(up.name)[1].lower() not in SUPPORTED_IMAGE_EXT:
            st.error("Unsupported image type.")
            return
        if st.button("Run Inspection", type="primary", key="cv_run_img"):
            with st.spinner("Industrial visual inspection..."):
                _run_image(up.getvalue(), baseline_bytes, ind, conf, use_coco)

    elif mode == "Video":
        up = st.file_uploader("Machine video", type=["mp4", "mov", "avi", "mkv", "webm"], key="cv_video")
        if up is None:
            st.info("Upload a video.")
            _diagnostics(env, ind_status)
            return
        if not caps["video"]:
            st.warning("Video needs OpenCV headless (see Advanced Diagnostics).")
            _diagnostics(env, ind_status)
            return
        if st.button("Run Inspection", type="primary", key="cv_run_vid"):
            path = save_upload_to_temp(up.getvalue(), os.path.splitext(up.name)[1].lower())
            try:
                with st.spinner("Analyzing video..."):
                    svc = VisionInspectionService(confidence=conf)
                    if use_coco:
                        svc = VisionInspectionService(confidence=conf, detector=_cached_detector())
                        svc.detector.set_confidence(conf)
                    _run_video(svc, path, stride, True, baseline_bytes, ind)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
    else:
        try:
            cam = st.camera_input("Snapshot")
        except Exception:
            st.info("Camera unavailable — use Image upload.")
            _diagnostics(env, ind_status)
            return
        if cam is None:
            _diagnostics(env, ind_status)
            return
        if st.button("Run Inspection", type="primary", key="cv_run_cam"):
            with st.spinner("Inspecting snapshot..."):
                _run_image(cam.getvalue(), baseline_bytes, ind, conf, use_coco)

    _diagnostics(env, ind_status)


def _run_image(data, baseline, ind, conf, use_coco) -> None:
    try:
        image = decode_image_bytes(data)
    except Exception as exc:  # noqa: BLE001
        st.error("Could not decode image.")
        st.caption(str(exc))
        return

    baseline_img = None
    if baseline:
        try:
            baseline_img = decode_image_bytes(baseline)
        except Exception:
            baseline_img = None

    industrial_score = None
    visual_health = None
    if ind is not None:
        industrial_score = ind.score_array(image)
        visual_health = ind.visual_health(industrial_score)

    diff = difference_map(image, baseline_img) if baseline_img is not None else None
    area = changed_area_ratio(image, baseline_img) if baseline_img is not None else None

    dets = []
    annotated = image.copy()
    if use_coco:
        det = _cached_detector()
        det.set_confidence(conf)
        if det.available:
            from vision.visualization import draw_detections

            dets = det.detect(image)
            annotated = draw_detections(image, dets)

    st.markdown("#### INSPECTION RESULT")
    n = 2 + (1 if diff is not None else 0)
    cols = st.columns(n)
    cols[0].image(bgr_to_rgb(image), caption="ORIGINAL", use_container_width=True)
    cols[1].image(bgr_to_rgb(annotated), caption="OVERLAY", use_container_width=True)
    if diff is not None:
        cols[2].image(bgr_to_rgb(diff), caption="PIXEL ANOMALY MAP vs reference", use_container_width=True)

    st.markdown("#### METRICS")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Industrial anomaly",
        f"{industrial_score:.3f}" if industrial_score is not None else "—",
    )
    m2.metric("Visual health", f"{visual_health}/100" if visual_health is not None else "—")
    m3.metric("Changed area", f"{area:.1%}" if area is not None else "—")
    m4.metric("COCO objects", len(dets) if use_coco else "off")

    if industrial_score is not None:
        if industrial_score < 0.20:
            status = "NORMAL VISUAL (synthetic domain)"
        elif industrial_score < 0.50:
            status = "SMALL DEVIATION"
        elif industrial_score < 0.75:
            status = "SIGNIFICANT DEVIATION"
        else:
            status = "STRONG VISUAL DEVIATION"
        st.markdown(f'<span class="badge badge-info">{status}</span>', unsafe_allow_html=True)
        st.caption(
            "Score from industrial PCA residual model on synthetic training distribution. "
            "Not failure probability."
        )

    if dets:
        st.dataframe(
            [{"class": d.class_name, "confidence": d.confidence} for d in dets[:30]],
            use_container_width=True,
            hide_index=True,
        )


def _run_video(svc, path, stride, tracking, baseline, ind) -> None:
    try:
        report, so, sa, _fr, metrics, events, motion = svc.inspect_video(
            path, stride=stride, use_tracking=tracking, baseline_bytes=baseline
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Video inspection failed.")
        st.caption(str(exc))
        return

    if so is not None:
        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(so), caption="SAMPLE", use_container_width=True)
        if sa is not None:
            c2.image(bgr_to_rgb(sa), caption="OVERLAY", use_container_width=True)
    if motion is not None:
        st.image(bgr_to_rgb(motion), caption="MOTION HEATMAP (not thermal)", use_container_width=True)

    if ind is not None and so is not None:
        sc = ind.score_array(so)
        st.metric("Industrial anomaly (sample frame)", f"{sc:.3f}")
        st.metric("Visual health", f"{ind.visual_health(sc)}/100")

    st.metric("Frames analyzed", metrics.get("frames_analyzed", 0))
    for e in events:
        st.caption(f"{e.event_type}: {e.evidence}")


def _diagnostics(env, ind_status) -> None:
    with st.expander("Advanced Diagnostics", expanded=False):
        st.write({"industrial_model": ind_status})
        st.code(f"{env['python_executable']}\nPython {env['python_version']}")
        st.write(
            {
                "pillow": env["pillow"],
                "opencv": {"available": env["opencv"]["available"], "version": env["opencv"].get("version")},
                "ultralytics": {"available": env["ultralytics"]["available"]},
            }
        )
        st.caption("Optional COCO path needs opencv-python-headless + ultralytics in THIS interpreter.")
        st.code(env.get("install_hint", ""))
