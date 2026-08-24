"""
Computer Vision Inspection — presentation layer.

Inference / model logic lives in vision/* and is unchanged by UI cleanup.
Technical environment details only appear under "Technical details".
"""

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

    # --- Header ---
    st.markdown(
        """
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:12px;margin-bottom:16px;">
            <h2 style="font-size:1.2rem;font-weight:600;margin:0;letter-spacing:0.02em;">
                COMPUTER VISION INSPECTION
            </h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:6px 0 0 0;">
                Visual inspection of the machine · Experimental laboratory module
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Engine status (user-facing only) ---
    if ind is not None:
        status_label, status_class = "Ready", "badge-success"
        status_hint = "Industrial visual model loaded"
    elif caps.get("image_upload"):
        status_label, status_class = "Limited", "badge-warning"
        status_hint = "Reference comparison available · Train the industrial model for full scoring"
    else:
        status_label, status_class = "Unavailable", "badge-critical"
        status_hint = "Image processing is not available in this session"

    model_name = ind_status.get("model_name") if ind else "—"
    dataset_name = ind_status.get("dataset_version", "—") if ind else "—"

    st.markdown(
        f"""
        <div class="ind-card" style="display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;">
            <div>
                <div class="ind-card-header">ENGINE</div>
                <span class="badge {status_class}">{status_label}</span>
                <div style="color:#9A9FA8;font-size:0.75rem;margin-top:6px;">{status_hint}</div>
            </div>
            <div>
                <div class="ind-card-header">MODEL</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">{model_name}</div>
            </div>
            <div>
                <div class="ind-card-header">DATASET</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">{dataset_name}</div>
            </div>
            <div>
                <div class="ind-card-header">TASK</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">Visual anomaly</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_insp, tab_data = st.tabs(["Inspection", "Dataset"])

    with tab_insp:
        _render_inspection(caps, ind)

    with tab_data:
        _render_dataset_gallery()

    # --- Technical details (collapsed, optional) ---
    with st.expander("Technical details", expanded=False):
        st.caption("Environment and optional tooling — not required for routine inspection.")
        st.write(
            {
                "industrial_model": ind_status,
                "image_upload": caps.get("image_upload"),
                "baseline_comparison": caps.get("baseline_comparison"),
                "video": caps.get("video"),
                "object_detection_optional": caps.get("object_detection"),
            }
        )
        st.markdown("**Runtime**")
        st.code(f"{env.get('python_executable', '')}\nPython {env.get('python_version', '')}")
        st.write(
            {
                "numpy": env.get("numpy"),
                "pillow": env.get("pillow"),
                "opencv": {
                    "available": env.get("opencv", {}).get("available"),
                    "version": env.get("opencv", {}).get("version"),
                },
                "ultralytics": {
                    "available": env.get("ultralytics", {}).get("available"),
                    "version": env.get("ultralytics", {}).get("version"),
                },
            }
        )
        if env.get("opencv", {}).get("error"):
            st.caption(env["opencv"]["error"])
        if env.get("ultralytics", {}).get("error"):
            st.caption(env["ultralytics"]["error"])
        if env.get("install_hint"):
            st.markdown("**Setup hint**")
            st.code(env["install_hint"])


def _render_inspection(caps: dict, ind: IndustrialAnomalyModel | None) -> None:
    st.markdown("#### Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        stride = st.select_slider(
            "Video frame sampling",
            options=[1, 2, 5, 10],
            value=DEFAULT_FRAME_STRIDE,
            key="cv_stride",
        )
    with c2:
        show_zones = st.checkbox("Zone scores", value=True, key="cv_zones")
    with c3:
        use_coco = st.checkbox("Generic detector (optional)", value=False, key="cv_coco")

    st.markdown("#### Input")
    mode = st.radio(
        "Source",
        ["Image", "Video", "Camera"],
        horizontal=True,
        key="cv_mode",
        label_visibility="collapsed",
    )
    source_labels = {"Image": "Image", "Video": "Video", "Camera": "Camera"}
    st.caption(f"Source: {source_labels.get(mode, mode)}")

    baseline_file = st.file_uploader(
        "Reference image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        key="cv_base",
    )
    baseline_bytes = baseline_file.getvalue() if baseline_file else None

    if mode == "Image":
        if not caps.get("image_upload"):
            st.warning("Image input is not available. See Technical details.")
            return
        up = st.file_uploader(
            "Machine image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="cv_img",
        )
        if up is None:
            st.info("Upload a machine image to start inspection.")
            return
        if os.path.splitext(up.name)[1].lower() not in SUPPORTED_IMAGE_EXT:
            st.error("Unsupported file type.")
            return
        if st.button("Run inspection", type="primary", key="cv_go_img"):
            with st.spinner("Inspecting…"):
                _run_image(up.getvalue(), baseline_bytes, ind, use_coco, show_zones)

    elif mode == "Video":
        up = st.file_uploader(
            "Machine video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="cv_vid",
        )
        if up is None:
            st.info("Upload a video to start temporal analysis.")
            return
        if not caps.get("video"):
            st.warning("Video analysis is not available. See Technical details.")
            return
        if os.path.splitext(up.name)[1].lower() not in SUPPORTED_VIDEO_EXT:
            st.error("Unsupported file type.")
            return
        if st.button("Run inspection", type="primary", key="cv_go_vid"):
            path = save_upload_to_temp(up.getvalue(), os.path.splitext(up.name)[1].lower())
            try:
                with st.spinner("Analyzing video…"):
                    _run_video_temporal(path, stride, ind)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    else:
        st.caption("Uses the browser camera when permitted.")
        if not caps.get("image_upload"):
            st.warning("Cannot process camera frames. See Technical details.")
            return
        try:
            cam = st.camera_input("Capture")
        except Exception:
            st.info("Camera is not available. Use Image upload instead.")
            return
        if cam is None:
            st.info("Waiting for a snapshot.")
            return
        if st.button("Run inspection", type="primary", key="cv_go_cam"):
            with st.spinner("Inspecting…"):
                _run_image(cam.getvalue(), baseline_bytes, ind, use_coco, show_zones)


def _run_image(data, baseline, ind, use_coco, show_zones) -> None:
    try:
        image = decode_image_bytes(data)
    except Exception as exc:
        st.error("Could not read the image.")
        st.caption(str(exc))
        return

    baseline_img = None
    if baseline:
        try:
            baseline_img = decode_image_bytes(baseline)
        except Exception:
            baseline_img = None

    industrial_score = None
    heat_bgr = None
    visual_health = None
    if ind is not None:
        industrial_score, heat_bgr = ind.score_with_heatmap(image)
        visual_health = ind.visual_health(industrial_score)

    diff = difference_map(image, baseline_img) if baseline_img is not None else None
    area = changed_area_ratio(image, baseline_img) if baseline_img is not None else None

    fused = industrial_score
    if industrial_score is not None and area is not None:
        ref_proxy = float(min(1.0, area * 2.0))
        fused = float(max(industrial_score, 0.5 * industrial_score + 0.5 * ref_proxy))

    st.markdown("#### Result")
    n_cols = 1 + (1 if heat_bgr is not None else 0) + (1 if diff is not None else 0)
    n_cols = max(2, min(3, n_cols))
    cols = st.columns(n_cols)
    cols[0].image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    col_i = 1
    if heat_bgr is not None and col_i < n_cols:
        cols[col_i].image(bgr_to_rgb(heat_bgr), caption="Anomaly map", use_container_width=True)
        col_i += 1
    if diff is not None and col_i < n_cols:
        cols[col_i].image(bgr_to_rgb(diff), caption="vs reference", use_container_width=True)

    st.markdown("#### Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomaly score", f"{industrial_score:.3f}" if industrial_score is not None else "—")
    m2.metric("Visual health", f"{visual_health}/100" if visual_health is not None else "—")
    m3.metric("Reference change", f"{area:.0%}" if area is not None else "—")
    m4.metric("Combined", f"{fused:.3f}" if fused is not None else "—")

    if industrial_score is not None:
        if industrial_score < 0.20:
            band = "Within normal visual range"
        elif industrial_score < 0.50:
            band = "Small visual deviation"
        elif industrial_score < 0.75:
            band = "Significant visual deviation"
        else:
            band = "Strong visual deviation"
        st.markdown(f"**Assessment:** {band}")
        bits = ["Pattern differs from the trained normal baseline"]
        if area is not None and area > 0.1:
            bits.append("Visible difference against the reference image")
        if heat_bgr is not None:
            bits.append("Spatial anomaly map available")
        st.caption(" · ".join(bits))
        st.caption("Scores describe visual deviation in this lab setting — not confirmed mechanical failure.")

    if show_zones and ind is not None:
        from vision.roi import default_rois

        h, w = image.shape[:2]
        anns = [{"class": r.name, "bbox": list(r.absolute(w, h))} for r in default_rois()]
        scores = ind.component_scores(image, anns)
        if scores:
            st.markdown("#### Zones")
            st.dataframe(
                [
                    {
                        "Zone": k,
                        "Anomaly": round(v, 3),
                        "Visual health": ind.visual_health(v),
                    }
                    for k, v in scores.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

    if use_coco:
        det = _cached_detector()
        if det.available:
            from vision.visualization import draw_detections

            dets = det.detect(image)
            st.markdown("#### Optional generic detections")
            st.image(
                bgr_to_rgb(draw_detections(image, dets)),
                caption="Generic object detector (not used for visual health)",
                use_container_width=True,
            )
        else:
            st.caption("Generic detector is not available in this session.")


def _run_video_temporal(path, stride, ind) -> None:
    from vision.video import iter_sampled_frames

    times, scores, healths = [], [], []
    sample = None
    for _fi, ts, frame in iter_sampled_frames(path, stride=stride):
        if sample is None:
            sample = frame
        if ind is None:
            continue
        s = ind.score_array(frame)
        times.append(ts)
        scores.append(s)
        healths.append(ind.visual_health(s))

    st.markdown("#### Result")
    if sample is not None:
        st.image(bgr_to_rgb(sample), caption="Sample frame", use_container_width=True)

    st.markdown("#### Analysis")
    if scores:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=times, y=scores, name="Anomaly", line=dict(color="#D4A84F", width=1.5))
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[h / 100.0 for h in healths],
                name="Visual health (scaled)",
                line=dict(color="#4CAF78", width=1.5),
            )
        )
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
        )
        apply_industrial_plotly_theme(fig, height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Trend over sampled frames — not an automatic failure call.")
    else:
        st.info("No anomaly scores for this video (model offline or no frames).")


def _render_dataset_gallery() -> None:
    st.markdown("#### Dataset samples")
    base = ROOT / "data" / "synthetic"
    if not base.exists():
        st.info("No local dataset found yet.")
        return
    versions = sorted(p.name for p in base.iterdir() if p.is_dir())
    if not versions:
        st.info("No local dataset found yet.")
        return
    ver = st.selectbox("Version", versions, key="cv_ds_ver")
    root = base / ver
    meta_path = root / "dataset_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Images", meta.get("n_images", "—"))
            c2.metric("Version", meta.get("version", ver))
            c3.metric("Synthetic", "Yes" if meta.get("synthetic") else "No")
        except Exception:
            pass
    img_dir = root / "images"
    if not img_dir.exists():
        return
    files = sorted(img_dir.glob("*.png"))[:12]
    if not files:
        st.caption("No sample images in this version.")
        return
    cols = st.columns(4)
    for i, f in enumerate(files):
        cols[i % 4].image(str(f), caption=f.name, use_container_width=True)
