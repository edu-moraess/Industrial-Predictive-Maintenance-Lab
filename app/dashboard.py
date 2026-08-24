"""
Industrial Predictive Maintenance Lab V2 — Operations Center (Streamlit).

Telemetry: VirtualMachine
Inference: ml.inference_engine.InferenceEngine (same pipeline as FastAPI)

Computer Vision is an independent module (sidebar switch).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import streamlit as st

from app.styles import INDUSTRIAL_THEME_CSS
from app.theme import apply_industrial_plotly_theme
from config.settings import settings
from ml.inference_engine import InferenceEngine, InferenceResult
from simulator.failures import FailureMode, MachineState
from simulator.machine import VirtualMachine

st.set_page_config(
    page_title="Industrial Operations Center",
    page_icon="\u2699\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)

# Module router — Computer Vision does not depend on sensor ML bootstrap
with st.sidebar:
    st.markdown("#### MODULE")
    _module = st.radio(
        "Module",
        ["Operations", "Computer Vision"],
        label_visibility="collapsed",
    )
    st.markdown("---")

if _module == "Computer Vision":
    from app.vision_page import render_computer_vision

    render_computer_vision()
    st.stop()


@st.cache_resource
def load_inference_engine() -> InferenceEngine:
    eng = InferenceEngine()
    eng.bootstrap_from_generator(hours=6, frequency_minutes=5)
    return eng


def _ensure_state() -> None:
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "readings" not in st.session_state:
        st.session_state.readings = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "health_history" not in st.session_state:
        st.session_state.health_history = []
    if "anomaly_history" not in st.session_state:
        st.session_state.anomaly_history = []
    if "rul_history" not in st.session_state:
        st.session_state.rul_history = []
    if "vm" not in st.session_state:
        st.session_state.vm = VirtualMachine("MACHINE_001")


_ensure_state()

try:
    engine = load_inference_engine()
    engine_error = None
except Exception as exc:  # noqa: BLE001
    engine = None
    engine_error = str(exc)

with st.sidebar:
    st.markdown("#### CONTROL")
    st.markdown("---")
    st.markdown("##### MACHINE")
    machine_id = st.selectbox(
        "Machine ID",
        ["MACHINE_001", "MACHINE_002", "MACHINE_003"],
        label_visibility="collapsed",
    )
    st.markdown("##### SIMULATION")
    machine_state_ui = st.selectbox(
        "Machine State",
        ["RUNNING", "IDLE", "MAINTENANCE"],
        label_visibility="collapsed",
    )
    st.markdown("##### FAILURE INJECTION")
    failure_mode_ui = st.selectbox(
        "Failure Mode",
        [
            "NORMAL_OPERATION",
            "BEARING_FAILURE",
            "OVERHEATING",
            "ELECTRICAL_FAULT",
            "IMBALANCE",
        ],
        label_visibility="collapsed",
    )
    st.markdown("##### TELEMETRY")
    refresh_interval = st.slider("Refresh Interval (s)", 0.5, 5.0, 2.0, 0.5)
    history_window = st.slider("History Window", 10, 80, 30, 5)

    st.markdown("##### CONTROLS")
    c1, c2, c3, c4 = st.columns(4)
    step_clicked = False
    with c1:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.sim_running = True
    with c2:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.sim_running = False
    with c3:
        if st.button("STEP", use_container_width=True):
            step_clicked = True
    with c4:
        if st.button("RESET", use_container_width=True):
            st.session_state.readings = []
            st.session_state.last_result = None
            st.session_state.health_history = []
            st.session_state.anomaly_history = []
            st.session_state.rul_history = []
            st.session_state.vm = VirtualMachine(machine_id)
            st.session_state.sim_running = False

    st.markdown("---")
    st.markdown(
        f"**Sim:** `{'RUNNING' if st.session_state.sim_running else 'PAUSED'}`"
    )
    if engine is not None:
        st_status = engine.status()
        st.markdown(
            f"**ML:** `{'READY' if st_status.ready else 'NOT READY'}`"
        )
        st.caption(
            f"Train samples: {st_status.training_samples} \u00b7 "
            f"Acc (train): {st_status.train_accuracy if st_status.train_accuracy is not None else 'N/A'}"
        )
    else:
        st.markdown("**ML:** `NOT READY`")
        st.caption(str(engine_error))

if st.session_state.vm.machine_id != machine_id:
    st.session_state.vm = VirtualMachine(machine_id)

_FAIL_MAP = {
    "NORMAL_OPERATION": (MachineState.NORMAL, FailureMode.NORMAL_OPERATION),
    "BEARING_FAILURE": (MachineState.CRITICAL, FailureMode.BEARING_FAILURE),
    "OVERHEATING": (MachineState.CRITICAL, FailureMode.OVERHEATING),
    "ELECTRICAL_FAULT": (MachineState.CRITICAL, FailureMode.ELECTRICAL_FAULT),
    "IMBALANCE": (MachineState.DEGRADING, FailureMode.IMBALANCE),
}
state_enum, fail_enum = _FAIL_MAP[failure_mode_ui]
st.session_state.vm.set_condition(state_enum, fail_enum)


def _run_one_cycle() -> None:
    if engine is None or not engine.is_ready:
        return
    frame = st.session_state.vm.generate_telemetry()
    st.session_state.readings.append(frame)
    while len(st.session_state.readings) > history_window:
        st.session_state.readings.pop(0)
    result = engine.predict(st.session_state.readings, machine_id=machine_id)
    st.session_state.last_result = result
    st.session_state.health_history.append(result.health_score)
    st.session_state.anomaly_history.append(result.anomaly_score)
    st.session_state.rul_history.append(result.rul_hours)
    for key in ("health_history", "anomaly_history", "rul_history"):
        while len(st.session_state[key]) > history_window:
            st.session_state[key].pop(0)


if st.session_state.sim_running or step_clicked:
    _run_one_cycle()

result: InferenceResult | None = st.session_state.last_result
readings: List[Dict[str, Any]] = st.session_state.readings

st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:baseline;
                border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
        <div>
            <h2 style="font-size:1.15rem;font-weight:600;margin:0;">INDUSTRIAL OPERATIONS CENTER</h2>
            <p style="font-family:SFMono-Regular,Consolas,monospace;font-size:0.75rem;
                      color:#9A9FA8;margin:2px 0 0 0;">{machine_id}</p>
        </div>
        <div>
            <span class="badge {'badge-success' if st.session_state.sim_running else 'badge-warning'}">
                {'RUNNING' if st.session_state.sim_running else 'PAUSED'}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if engine is None:
    st.markdown(
        """
        <div class="ind-card" style="text-align:center;padding:24px;">
            <div class="ind-card-header">ML ENGINE NOT READY</div>
            <p style="color:#9A9FA8;">Initialize failed. Check dependencies and database path.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not readings or result is None:
    st.markdown(
        """
        <div class="ind-card" style="text-align:center;padding:24px;">
            <div class="ind-card-header">NO TELEMETRY DATA</div>
            <p style="color:#9A9FA8;">Press START or STEP to collect simulated telemetry and run inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

frame = readings[-1]

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    filled = max(0, min(20, int(result.health_score / 5)))
    bar = "\u2588" * filled + "\u2591" * (20 - filled)
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">HEALTH</div>
            <div class="ind-card-value">{result.health_score} <span class="ind-card-unit">/ 100</span></div>
            <div style="font-family:monospace;font-size:0.65rem;color:#D4A84F;">{bar}</div>
            <div class="ind-card-desc">{result.risk_level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">ANOMALY</div>
            <div class="ind-card-value">{result.anomaly_score:.3f}</div>
            <div class="ind-card-desc">
                <span class="badge {'badge-critical' if result.is_anomaly else 'badge-success'}">
                    {'ACTIVE' if result.is_anomaly else 'NOMINAL'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">RUL (EXPERIMENTAL)</div>
            <div class="ind-card-value">{result.rul_hours:.0f} <span class="ind-card-unit">HRS</span></div>
            <div class="ind-card-desc">Synthetic degradation model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">PREDICTED FAILURE</div>
            <div class="ind-card-value" style="font-size:0.95rem;padding-top:6px;">{result.failure_mode}</div>
            <div class="ind-card-desc">{result.failure_probability:.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k5:
    gt = result.ground_truth_failure or "\u2014"
    match = (
        "CORRECT"
        if result.prediction_correct is True
        else ("MISMATCH" if result.prediction_correct is False else "N/A")
    )
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">GROUND TRUTH</div>
            <div class="ind-card-value" style="font-size:0.95rem;padding-top:6px;">{gt}</div>
            <div class="ind-card-desc">{match} \u00b7 {result.inference_ms:.1f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("#### ML SYSTEM STATUS")
ms = engine.status()
mcols = st.columns(4)
mcols[0].markdown(
    f'<div class="ind-card"><div class="ind-card-header">ENGINE</div>'
    f'<div class="ind-card-value" style="font-size:1rem;">{"ONLINE" if ms.ready else "OFFLINE"}</div></div>',
    unsafe_allow_html=True,
)
mcols[1].markdown(
    f'<div class="ind-card"><div class="ind-card-header">ISOLATION FOREST</div>'
    f'<div class="ind-card-value" style="font-size:1rem;">{ms.isolation_forest}</div></div>',
    unsafe_allow_html=True,
)
mcols[2].markdown(
    f'<div class="ind-card"><div class="ind-card-header">RANDOM FOREST</div>'
    f'<div class="ind-card-value" style="font-size:1rem;">{ms.random_forest}</div></div>',
    unsafe_allow_html=True,
)
mcols[3].markdown(
    f'<div class="ind-card"><div class="ind-card-header">TRAINING SAMPLES</div>'
    f'<div class="ind-card-value" style="font-size:1rem;">{ms.training_samples}</div>'
    f'<div class="ind-card-desc">Last: {ms.last_training or "N/A"}</div></div>',
    unsafe_allow_html=True,
)

st.markdown("#### LIVE SENSOR MONITORING")
st.caption("SIMULATED TELEMETRY \u00b7 baselines from config/settings.py")

s1, s2, s3, s4, s5 = st.columns(5)
for col, label, val, unit, base in [
    (s1, "Temperature", frame["temperature"], "\u00b0C", settings.BASE_TEMP),
    (s2, "Vibration", frame["vibration"], "mm/s", settings.BASE_VIBRATION),
    (s3, "Current", frame["current"], "A", settings.BASE_CURRENT),
    (s4, "RPM", frame["rpm"], "RPM", settings.BASE_RPM),
    (s5, "Noise", frame["noise"], "dB", settings.BASE_NOISE),
]:
    col.markdown(
        f"""
        <div class="ind-card" style="padding:10px;">
            <div class="ind-card-header">{label}</div>
            <div class="ind-card-value" style="font-size:1.1rem;">{val:.2f}
                <span class="ind-card-unit">{unit}</span></div>
            <div class="ind-card-desc">Baseline {base} (simulated)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

temps = [r["temperature"] for r in readings]
vibs = [r["vibration"] for r in readings]
currs = [r["current"] for r in readings]

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=temps, name="Temperature", line=dict(color="#D4A84F", width=1.5)))
    fig.add_hline(y=settings.BASE_TEMP, line_dash="dot", line_color="#9A9FA8",
                  annotation_text="SIMULATED BASELINE")
    fig.add_trace(go.Scatter(y=vibs, name="Vibration", line=dict(color="#9A9FA8", width=1.5), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Temp (\u00b0C)"),
        yaxis2=dict(title="Vib (mm/s)", overlaying="y", side="right"),
    )
    apply_industrial_plotly_theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=currs, name="Current", line=dict(color="#4CAF78", width=1.5)))
    fig2.add_hline(y=settings.BASE_CURRENT, line_dash="dot", line_color="#9A9FA8",
                   annotation_text="SIMULATED BASELINE")
    apply_industrial_plotly_theme(fig2, height=260)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### HEALTH / ANOMALY / RUL TRENDS")
h1, h2, h3 = st.columns(3)
with h1:
    fh = go.Figure(go.Scatter(y=st.session_state.health_history, line=dict(color="#D4A84F", width=1.5), name="Health"))
    fh.add_hrect(y0=90, y1=100, fillcolor="rgba(76,175,120,0.08)", line_width=0)
    fh.add_hrect(y0=50, y1=70, fillcolor="rgba(217,164,65,0.08)", line_width=0)
    fh.add_hrect(y0=0, y1=25, fillcolor="rgba(217,92,92,0.08)", line_width=0)
    apply_industrial_plotly_theme(fh, height=220)
    st.plotly_chart(fh, use_container_width=True)
with h2:
    fa = go.Figure(go.Scatter(y=st.session_state.anomaly_history, line=dict(color="#D95C5C", width=1.5), name="Anomaly"))
    apply_industrial_plotly_theme(fa, height=220)
    st.plotly_chart(fa, use_container_width=True)
with h3:
    fr = go.Figure(go.Scatter(y=st.session_state.rul_history, line=dict(color="#9A9FA8", width=1.5), name="RUL"))
    apply_industrial_plotly_theme(fr, height=220)
    st.caption("EXPERIMENTAL RUL \u2014 synthetic degradation model, not industrial prognosis.")
    st.plotly_chart(fr, use_container_width=True)

st.markdown("#### FAILURE DIAGNOSTICS")
probs = result.failure_probabilities
labels = list(probs.keys())
values = [probs[k] for k in labels]
order = sorted(zip(values, labels), reverse=True)
values_s, labels_s = zip(*order) if order else ([], [])
figd = go.Figure(
    go.Bar(
        x=list(values_s),
        y=list(labels_s),
        orientation="h",
        marker_color=["#D95C5C" if v == max(values_s) else "#2A2F38" for v in values_s],
    )
)
figd.update_layout(xaxis=dict(title="Model probability (%)", range=[0, 100]))
apply_industrial_plotly_theme(figd, height=220)
st.plotly_chart(figd, use_container_width=True)

d1, d2 = st.columns(2)
with d1:
    st.markdown("#### MAINTENANCE INSIGHT")
    st.markdown(
        f"""
        <div class="ind-card">
            <div style="margin-bottom:8px;"><span class="badge badge-info">FROM INFERENCE ENGINE</span></div>
            <p style="color:#F2F2F2;margin:0 0 8px 0;">{result.maintenance_recommendation}</p>
            <p style="font-size:0.75rem;color:#9A9FA8;margin:0;">
                Risk: {result.risk_level} \u00b7 Predicted: {result.failure_mode}
                ({result.failure_probability:.1f}%) \u00b7 GT: {result.ground_truth_failure or 'N/A'}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d2:
    st.markdown("#### MACHINE DIGITAL TWIN (SCHEMATIC)")
    bearing_color = "#D95C5C" if "BEARING" in result.failure_mode else "#4CAF78"
    motor_color = "#D95C5C" if "ELECTRICAL" in result.failure_mode or "OVERHEAT" in result.failure_mode else "#4CAF78"
    shaft_color = "#D95C5C" if "IMBALANCE" in result.failure_mode else "#4CAF78"
    st.markdown(
        f"""
        <div class="ind-card" style="font-family:monospace;font-size:0.8rem;line-height:1.6;">
            <div style="text-align:center;border:1px solid {motor_color};padding:8px;margin:4px 40px;">MOTOR</div>
            <div style="text-align:center;color:#9A9FA8;">|</div>
            <div style="text-align:center;border:1px solid {bearing_color};padding:8px;margin:4px 40px;">BEARING</div>
            <div style="text-align:center;color:#9A9FA8;">|</div>
            <div style="text-align:center;border:1px solid {shaft_color};padding:8px;margin:4px 40px;">SHAFT</div>
            <p style="color:#9A9FA8;font-size:0.7rem;margin-top:10px;">Highlight driven by predicted failure mode (model output).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.sim_running:
    time.sleep(refresh_interval)
    st.rerun()
