"""Computer Vision Inspection — industrial_anomaly_v0.2 primary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.theme import apply_industrial_plotly_theme
from vision.anomaly import changed_area_ratio, difference_map
from vision.config import DEFAULT_FRAME_STRIDE, SUPPORTED_IMAGE_EXT, SUPPORTED_VIDEO_EXT
from vision.detector import ObjectDetector
from vision.industrial_model import IndustrialAnomalyModel, industrial_model_status
from vision.model_loader import get_vision_environment_status
from vision.preprocessing import decode_image_bytes
from vision.video import save_upload_to_temp
from vision.visualization import bgr_to_rgb

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def _cached_detector() -> ObjectDetector:
    return ObjectDetector()


@st.cache_resource
def _cached_industrial() -> IndustrialAnomalyModel | None:
    return IndustrialAnomalyModel.try_load("industrial_anomaly_v0.2")


def render_computer_vision() -> None:
    env = get_vision_environment_status()
    caps = env["capabilities"]
    ind_status = industrial_model_status()
    ind = _cached_industrial()

    if ind is not None:
        engine_txt, badge = "READY", "badge-success"
        model_label = ind_status.get("model_name", "industrial_anomaly_v0.2")
    elif caps["image_upload"]:
        engine_txt, badge = "PARTIALLY AVAILABLE", "badge-warning"
        model_label = "Train v0.2 to enable"
    else:
        engine_txt, badge = "OFFLINE", "badge-critical"
        model_label = "Unavailable"

    st.markdown(
        f"""
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:4px 0 0 0;">
                Industrial Vision Engine v0.2 \u00b7 Synthetic domain \u00b7 Not factory certification
            </p>
        </div>
        <div class="ind-card" style="display:flex;flex-wrap:wrap;gap:16px;align-items:center;">
            <div><div class="ind-card-header">VISION ENGINE</div>
                <span class="badge {badge}">{engine_txt}</span></div>
            <div><div class="ind-card-header">MODEL</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">{model_label}</span></div>
            <div><div class="ind-card-header">DATASET</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">
                {ind_status.get('dataset_version', 'industrial_dataset_v0.2')}</span></div>
            <div><div class="ind-card-header">TASK</div>
                <span style="color:#F2F2F2;font-size:0.9rem;">Visual anomaly detection</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Primary path: industrial PCA residual (v0.1 gray baseline / v0.2 richer features). "
        "Not COCO. Synthetic-to-real gap applies. Scores are not failure probabilities."
    )

    if ind is None:
        st.info(
            "```bash\n"
            "python -m dataset.generator --version v0.2\n"
            "python -m training.train --dataset-version v0.2 --feature-mode rich\n"
            "python -m training.evaluate\n"
            "```"
        )

    tab_insp, tab_data = st.tabs(["Inspection", "Dataset Explorer"])

    with tab_insp:
        _inspection_tab(env, caps, ind)
    with tab_data:
        _dataset_explorer()

    with st.expander("Advanced Diagnostics", expanded=False):
        st.write({"industrial_model": ind_status})
        st.code(f"{env['python_executable']}\nPython {env['python_version']}")
        st.write(
            {
                "pillow": env["pillow"].get("available"),
                "opencv": env["opencv"].get("available"),
                "ultralytics": env["ultralytics"].get("available"),
            }
        )


def _inspection_tab(env, caps, ind) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        stride = st.select_slider("Frame sampling", options=[1, 2, 5, 10], value=DEFAULT_FRAME_STRIDE, key="cv_stride")
    with c2:
        use_coco = st.checkbox("Optional COCO detector", value=False, key="cv_coco")
    with c3:
        show_components = st.checkbox("Component scores (if annotations)", value=True, key="cv_comp")

    mode = st.radio("Source", ["Image", "Video", "Camera"], horizontal=True, key="cv_mode")
    baseline_file = st.file_uploader("Reference image (optional)", type=["jpg", "jpeg", "png", "webp"], key="cv_base")
    baseline_bytes = baseline_file.getvalue() if baseline_file else None

    if mode == "Image":
        if not caps["image_upload"]:
            st.warning("Image pipeline needs Pillow or OpenCV.")
            return
        up = st.file_uploader("Machine image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cv_img")
        if up is None:
            st.info("Upload a machine image (synthetic samples or domain photos).")
            return
        if st.button("Run Inspection", type="primary", key="cv_go_img"):
            with st.spinner("Running industrial inspection..."):
                _run_image(up.getvalue(), baseline_bytes, ind, use_coco, show_components)
    elif mode == "Video":
        up = st.file_uploader("Machine video", type=["mp4", "mov", "avi", "mkv", "webm"], key="cv_vid")
        if up is None:
            return
        if not caps["video"]:
            st.warning("Video requires OpenCV headless.")
            return
        if st.button("Run Inspection", type="primary", key="cv_go_vid"):
            path = save_upload_to_temp(up.getvalue(), os.path.splitext(up.name)[1].lower())
            try:
                with st.spinner("Temporal analysis..."):
                    _run_video_temporal(path, stride, ind)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
    else:
        cam = st.camera_input("Snapshot")
        if cam and st.button("Run Inspection", type="primary", key="cv_go_cam"):
            with st.spinner("Inspecting..."):
                _run_image(cam.getvalue(), baseline_bytes, ind, use_coco, show_components)


def _run_image(data, baseline, ind, use_coco, show_components) -> None:
    try:
        image = decode_image_bytes(data)
    except Exception as exc:
        st.error("Decode failed")
        st.caption(str(exc))
        return

    baseline_img = decode_image_bytes(baseline) if baseline else None
    industrial_score = None
    heat_bgr = None
    visual_health = None
    if ind is not None:
        industrial_score, heat_bgr = ind.score_with_heatmap(image)
        visual_health = ind.visual_health(industrial_score)

    diff = difference_map(image, baseline_img) if baseline_img is not None else None
    area = changed_area_ratio(image, baseline_img) if baseline_img is not None else None

    # optional fusion: max of model and mild reference (documented, not sum)
    fused = industrial_score
    if industrial_score is not None and area is not None:
        ref_proxy = float(min(1.0, area * 2.0))
        fused = float(max(industrial_score, 0.5 * industrial_score + 0.5 * ref_proxy))

    st.markdown("#### RESULT")
    cols = st.columns(3 if heat_bgr is not None else 2)
    cols[0].image(bgr_to_rgb(image), caption="ORIGINAL", use_container_width=True)
    if heat_bgr is not None:
        cols[1].image(bgr_to_rgb(heat_bgr), caption="ANOMALY HEATMAP (patch residual)", use_container_width=True)
        if diff is not None:
            cols[2].image(bgr_to_rgb(diff), caption="REFERENCE DIFFERENCE", use_container_width=True)
        else:
            cols[2].image(bgr_to_rgb(image), caption="(no reference)", use_container_width=True)
    else:
        cols[1].image(bgr_to_rgb(image), caption="(model offline)", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomaly score", f"{industrial_score:.3f}" if industrial_score is not None else "—")
    m2.metric("Visual health", f"{visual_health}/100" if visual_health is not None else "—")
    m3.metric("Ref. changed area", f"{area:.1%}" if area is not None else "—")
    m4.metric("Fused (doc.)", f"{fused:.3f}" if fused is not None else "—")

    if industrial_score is not None:
        band = (
            "NORMAL VISUAL band"
            if industrial_score < 0.2
            else "SMALL DEVIATION"
            if industrial_score < 0.5
            else "SIGNIFICANT DEVIATION"
            if industrial_score < 0.75
            else "STRONG VISUAL DEVIATION"
        )
        st.markdown(f"**Status:** {band}")
        evidence = ["deviation from normal training pattern (synthetic domain)"]
        if area is not None and area > 0.1:
            evidence.append("reference pixel difference present")
        if heat_bgr is not None:
            evidence.append("spatial residual heatmap computed")
        st.markdown("**Visual evidence:** " + "; ".join(evidence))

    if show_components and ind is not None:
        # try synthetic annotation layout from default component ROIs if file not provided
        from vision.roi import default_rois

        h, w = image.shape[:2]
        fake_anns = []
        for roi in default_rois():
            x1, y1, x2, y2 = roi.absolute(w, h)
            fake_anns.append({"class": roi.name, "bbox": [x1, y1, x2, y2]})
        scores = ind.component_scores(image, fake_anns)
        if scores:
            st.markdown("#### COMPONENT / ZONE SCORES (ROI proxy)")
            st.dataframe(
                [{"zone": k, "anomaly": round(v, 3), "visual_health": ind.visual_health(v)} for k, v in scores.items()],
                use_container_width=True,
                hide_index=True,
            )

    if use_coco:
        det = _cached_detector()
        if det.available:
            from vision.visualization import draw_detections

            dets = det.detect(image)
            st.image(bgr_to_rgb(draw_detections(image, dets)), caption="Optional COCO overlay", use_container_width=True)


def _run_video_temporal(path, stride, ind) -> None:
    from vision.video import iter_sampled_frames

    times, scores, healths = [], [], []
    sample = None
    for fi, ts, frame in iter_sampled_frames(path, stride=stride):
        if sample is None:
            sample = frame
        if ind is None:
            continue
        s = ind.score_array(frame)
        times.append(ts)
        scores.append(s)
        healths.append(ind.visual_health(s))

    if sample is not None:
        st.image(bgr_to_rgb(sample), caption="Sample frame", use_container_width=True)
    if scores:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=scores, name="Anomaly", line=dict(color="#D4A84F")))
        fig.add_trace(go.Scatter(x=times, y=[h / 100 for h in healths], name="Visual health/100", line=dict(color="#4CAF78")))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Score", yaxis=dict(range=[0, 1]))
        apply_industrial_plotly_theme(fig, height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Temporal anomaly trend — not automatic failure declaration.")
    else:
        st.warning("Industrial model offline or no frames.")


def _dataset_explorer() -> None:
    st.markdown("#### DATASET EXPLORER")
    versions = []
    base = ROOT / "data" / "synthetic"
    if base.exists():
        versions = sorted([p.name for p in base.iterdir() if p.is_dir()])
    if not versions:
        st.info("No local dataset yet. Run dataset.generator.")
        return
    ver = st.selectbox("Version", versions)
    root = base / ver
    meta_path = root / "dataset_meta.json"
    if meta_path.exists():
        st.json(json.loads(meta_path.read_text(encoding="utf-8")))
    img_dir = root / "images"
    if img_dir.exists():
        files = sorted(img_dir.glob("*.png"))[:12]
        if files:
            cols = st.columns(4)
            for i, f in enumerate(files):
                cols[i % 4].image(str(f), caption=f.name, use_container_width=True)
