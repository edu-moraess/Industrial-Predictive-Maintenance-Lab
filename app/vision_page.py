"""
Computer Vision Inspection UI.
Image input: Pillow-only. Video: OpenCV when available.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.theme import apply_industrial_plotly_theme
from vision.anomaly import changed_area_ratio, difference_map
from vision.config import DEFAULT_FRAME_STRIDE, MAX_VIDEO_FRAMES_ANALYZED
from vision.detector import ObjectDetector
from vision.industrial_model import IndustrialAnomalyModel, industrial_model_status
from vision.input_io import (
    InputError,
    cleanup_video,
    load_image_from_upload,
    load_video_from_upload,
    rgb_to_bgr,
)
from vision.model_loader import get_vision_environment_status
from vision.preprocessing import ensure_bgr, resize_max_side
from vision.video import iter_sampled_frames
from vision.visualization import bgr_to_rgb

ROOT = Path(__file__).resolve().parent.parent

# Explicit UI states
NO_INPUT = "NO_INPUT"
INPUT_LOADED = "INPUT_LOADED"
PROCESSING = "PROCESSING"
SUCCESS = "SUCCESS"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
DECODER_UNAVAILABLE = "DECODER_UNAVAILABLE"
INVALID_INPUT = "INVALID_INPUT"
PROCESSING_ERROR = "PROCESSING_ERROR"


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

    st.markdown(
        """
        <div style="border-bottom:1px solid #2A2F38;padding-bottom:12px;margin-bottom:16px;">
            <h2 style="font-size:1.2rem;font-weight:600;margin:0;">COMPUTER VISION INSPECTION</h2>
            <p style="color:#9A9FA8;font-size:0.8rem;margin:6px 0 0 0;">
                Visual inspection of the machine \u00b7 Experimental laboratory module
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ind is not None:
        status_label, status_class = "Ready", "badge-success"
        status_hint = "Industrial visual model loaded"
    elif caps.get("image_upload"):
        status_label, status_class = "Limited", "badge-warning"
        status_hint = "Images work \u00b7 Train the industrial model for anomaly scores"
    else:
        status_label, status_class = "Unavailable", "badge-critical"
        status_hint = "Image processing is not available"

    model_name = ind_status.get("model_name") if ind else "\u2014"
    dataset_name = ind_status.get("dataset_version", "\u2014") if ind else "\u2014"

    st.markdown(
        f"""
        <div class="ind-card" style="display:flex;flex-wrap:wrap;gap:20px;">
            <div>
                <div class="ind-card-header">ENGINE</div>
                <span class="badge {status_class}">{status_label}</span>
                <div style="color:#9A9FA8;font-size:0.75rem;margin-top:6px;">{status_hint}</div>
            </div>
            <div><div class="ind-card-header">MODEL</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">{model_name}</div></div>
            <div><div class="ind-card-header">DATASET</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">{dataset_name}</div></div>
            <div><div class="ind-card-header">TASK</div>
                <div style="color:#F2F2F2;font-size:0.9rem;">Visual anomaly</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_insp, tab_data = st.tabs(["Inspection", "Dataset"])
    with tab_insp:
        _render_inspection(caps, ind)
    with tab_data:
        _render_dataset_gallery()

    with st.expander("Technical details", expanded=False):
        st.caption("Environment diagnostics \u2014 not required for routine inspection.")
        st.write(
            {
                "industrial_model": ind_status,
                "image_upload": caps.get("image_upload"),
                "video": caps.get("video"),
                "object_detection_optional": caps.get("object_detection"),
            }
        )
        st.code(f"{env.get('python_executable', '')}\nPython {env.get('python_version', '')}")
        st.write(
            {
                "pillow": env.get("pillow"),
                "opencv": {
                    "available": env.get("opencv", {}).get("available"),
                    "version": env.get("opencv", {}).get("version"),
                },
                "ultralytics": {"available": env.get("ultralytics", {}).get("available")},
            }
        )
        if env.get("opencv", {}).get("error"):
            st.caption(env["opencv"]["error"])
        if env.get("install_hint"):
            st.code(env["install_hint"])
        last_err = st.session_state.get("cv_last_error")
        if last_err:
            st.caption("Last processing error:")
            st.code(str(last_err))


def _render_inspection(caps: dict, ind) -> None:
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
    mode = st.radio("Source", ["Image", "Video", "Camera"], horizontal=True, key="cv_mode")

    baseline_file = st.file_uploader(
        "Reference image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        key="cv_base",
    )

    if mode == "Image":
        up = st.file_uploader(
            "Machine image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="cv_img",
        )
        if up is None:
            st.info("Upload a machine image to start inspection.")
            return

        try:
            payload = load_image_from_upload(up)
        except InputError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error("Unable to read image. Please try JPG, PNG, WEBP or BMP.")
            st.session_state["cv_last_error"] = str(exc)
            return

        kb = payload.size_bytes / 1024.0
        st.success("Image loaded")
        st.caption(f"{payload.width} \u00d7 {payload.height} \u00b7 {payload.format} \u00b7 {kb:.0f} KB")
        st.image(payload.rgb, caption="Preview", use_container_width=True)

        if st.button("Inspect image", type="primary", key="cv_go_img"):
            with st.spinner("Inspecting\u2026"):
                ref_rgb = None
                if baseline_file is not None:
                    try:
                        ref_rgb = load_image_from_upload(baseline_file).rgb
                    except InputError as exc:
                        st.warning(str(exc))
                _inspect_rgb(payload.rgb, ref_rgb, ind, use_coco, show_zones)

    elif mode == "Video":
        up = st.file_uploader(
            "Machine video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="cv_vid",
        )
        if up is None:
            st.info("Upload a video for temporal analysis.")
            return
        if not caps.get("video"):
            st.warning(
                "Video processing unavailable. Image inspection still works. "
                "Enable OpenCV headless in this environment to analyze video."
            )
            return

        try:
            vpayload = load_video_from_upload(up)
        except InputError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error("Unable to decode video. Please try MP4.")
            st.session_state["cv_last_error"] = str(exc)
            return

        st.success("Video loaded")
        meta_bits = []
        if vpayload.width and vpayload.height:
            meta_bits.append(f"{vpayload.width} \u00d7 {vpayload.height}")
        if vpayload.fps:
            meta_bits.append(f"{vpayload.fps:.1f} FPS")
        if vpayload.duration_s is not None:
            meta_bits.append(f"{vpayload.duration_s:.1f} s")
        if meta_bits:
            st.caption(" \u00b7 ".join(meta_bits))

        if st.button("Inspect video", type="primary", key="cv_go_vid"):
            try:
                with st.spinner("Analyzing video\u2026"):
                    _inspect_video(vpayload.path, stride, ind)
            finally:
                cleanup_video(vpayload)

    else:
        st.caption("Browser camera when permitted.")
        try:
            cam = st.camera_input("Capture")
        except Exception:
            st.info("Camera is not available. Use Image upload.")
            return
        if cam is None:
            st.info("Waiting for a snapshot.")
            return
        try:
            payload = load_image_from_upload(cam)
        except InputError as exc:
            st.error(str(exc))
            return
        st.success("Snapshot loaded")
        st.image(payload.rgb, caption="Preview", use_container_width=True)
        if st.button("Inspect image", type="primary", key="cv_go_cam"):
            ref_rgb = None
            if baseline_file is not None:
                try:
                    ref_rgb = load_image_from_upload(baseline_file).rgb
                except InputError:
                    pass
            with st.spinner("Inspecting\u2026"):
                _inspect_rgb(payload.rgb, ref_rgb, ind, use_coco, show_zones)


def _inspect_rgb(rgb, ref_rgb, ind, use_coco, show_zones) -> None:
    image_bgr = rgb_to_bgr(rgb)
    baseline_bgr = rgb_to_bgr(ref_rgb) if ref_rgb is not None else None
    # Cap resolution for scoring/heatmap only — the original `rgb` is still
    # shown as-is in the "Original" panel below. Large phone photos (e.g.
    # 12MP) made the patch-wise heatmap take ~1.5s; capped to MAX_IMAGE_SIDE
    # it drops to ~0.2s with no meaningful change in the anomaly score.
    proc_bgr = resize_max_side(ensure_bgr(image_bgr))
    proc_baseline_bgr = resize_max_side(ensure_bgr(baseline_bgr)) if baseline_bgr is not None else None

    industrial_score = None
    heat_bgr = None
    visual_health = None
    state = SUCCESS

    if ind is None:
        state = MODEL_UNAVAILABLE
        st.warning("Industrial model not trained yet. Preview and reference comparison still run.")
    else:
        try:
            industrial_score, heat_bgr = ind.score_with_heatmap(proc_bgr)
            visual_health = ind.visual_health(industrial_score)
        except Exception as exc:
            state = PROCESSING_ERROR
            st.error("Inspection could not complete.")
            st.session_state["cv_last_error"] = str(exc)
            return

    diff = None
    area = None
    if proc_baseline_bgr is not None:
        try:
            diff = difference_map(proc_bgr, proc_baseline_bgr)
            area = changed_area_ratio(proc_bgr, proc_baseline_bgr)
        except Exception:
            pass

    fused = industrial_score
    if industrial_score is not None and area is not None:
        ref_proxy = float(min(1.0, area * 2.0))
        fused = float(max(industrial_score, 0.5 * industrial_score + 0.5 * ref_proxy))

    st.markdown("#### Result")
    cols = st.columns(3 if (heat_bgr is not None or diff is not None) else 1)
    cols[0].image(rgb, caption="Original", use_container_width=True)
    if heat_bgr is not None:
        cols[1].image(bgr_to_rgb(heat_bgr), caption="Anomaly map", use_container_width=True)
    if diff is not None:
        idx = 2 if heat_bgr is not None else 1
        if idx < len(cols):
            cols[idx].image(bgr_to_rgb(diff), caption="vs reference", use_container_width=True)

    st.markdown("#### Analysis")
    if state == MODEL_UNAVAILABLE:
        st.info("Status: model unavailable \u2014 not classified as normal or anomalous.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Anomaly score", f"{industrial_score:.3f}" if industrial_score is not None else "\u2014")
        m2.metric("Visual health", f"{visual_health}/100" if visual_health is not None else "\u2014")
        m3.metric("Reference change", f"{area:.0%}" if area is not None else "\u2014")
        m4.metric("Combined", f"{fused:.3f}" if fused is not None else "\u2014")
        if area is not None:
            st.caption(
                "Reference change is a raw pixel-difference ratio \u2014 lighting or "
                "camera-angle changes can also raise it, not only physical defects."
            )
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
            st.caption("Visual deviation in this lab setting \u2014 not confirmed mechanical failure.")

    if show_zones and ind is not None:
        from vision.roi import default_rois

        h, w = proc_bgr.shape[:2]
        anns = [{"class": r.name, "bbox": list(r.absolute(w, h))} for r in default_rois()]
        scores = ind.component_scores(proc_bgr, anns)
        if scores:
            st.markdown("#### Zones")
            st.dataframe(
                [
                    {"Zone": k, "Anomaly": round(v, 3), "Visual health": ind.visual_health(v)}
                    for k, v in scores.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

    if use_coco:
        det = _cached_detector()
        if det.available:
            from vision.visualization import draw_detections

            dets = det.detect(image_bgr)
            st.markdown("#### Optional generic detections")
            st.image(
                bgr_to_rgb(draw_detections(image_bgr, dets)),
                caption="Not used for visual health",
                use_container_width=True,
            )
        else:
            st.caption("Generic detector is not available in this session.")


def _inspect_video(path: str, stride: int, ind) -> None:
    times, scores, healths = [], [], []
    sample_bgr = None
    n_seen = 0
    truncated = False
    try:
        for fi, ts, frame in iter_sampled_frames(path, stride=stride):
            if sample_bgr is None:
                sample_bgr = frame
            if n_seen >= MAX_VIDEO_FRAMES_ANALYZED:
                truncated = True
                break
            n_seen += 1
            if ind is None:
                continue
            s = ind.score_array(frame)
            times.append(ts)
            scores.append(s)
            healths.append(ind.visual_health(s))
    except InputError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error("Unable to decode video. Please try MP4.")
        st.session_state["cv_last_error"] = str(exc)
        return

    st.markdown("#### Result")
    if sample_bgr is not None:
        st.image(bgr_to_rgb(sample_bgr), caption="Sample frame", use_container_width=True)

    st.markdown("#### Analysis")
    if ind is None:
        st.warning("Industrial model not trained yet. Frames were read but not scored.")
        return
    if not scores:
        st.info("No frames analyzed.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Frames analyzed", len(scores))
    m2.metric("Average anomaly", f"{sum(scores)/len(scores):.3f}")
    m3.metric("Peak anomaly", f"{max(scores):.3f}")
    if truncated:
        st.caption(
            f"Capped at {MAX_VIDEO_FRAMES_ANALYZED} sampled frames for this inspection. "
            "Increase the frame sampling stride to cover more of a long video."
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=scores, name="Anomaly", line=dict(color="#D4A84F", width=1.5)))
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[h / 100.0 for h in healths],
            name="Visual health (scaled)",
            line=dict(color="#4CAF78", width=1.5),
        )
    )
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Score", yaxis=dict(range=[0, 1]))
    apply_industrial_plotly_theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Trend over sampled frames \u2014 not an automatic failure call.")


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
            c1.metric("Images", meta.get("n_images", "\u2014"))
            c2.metric("Version", meta.get("version", ver))
            c3.metric("Synthetic", "Yes" if meta.get("synthetic") else "No")
        except Exception:
            pass
    img_dir = root / "images"
    if not img_dir.exists():
        return
    files = sorted(img_dir.glob("*.png"))[:12]
    cols = st.columns(4)
    for i, f in enumerate(files):
        cols[i % 4].image(str(f), caption=f.name, use_container_width=True)
