"""
Industrial Predictive Maintenance Lab — Streamlit Control Center.

Telemetry is generated from the official VirtualMachine simulator.
Health / anomaly / RUL values shown here are provisional simulation
outputs for the UI lab. Production-grade inference is available via
the FastAPI /predict endpoint.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path so `from app.styles` works when
# Streamlit runs this file as a script (script dir is app/, not project root).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.styles import INDUSTRIAL_THEME_CSS
from app.theme import apply_industrial_plotly_theme
from config.settings import settings
from simulator.failures import FailureMode, MachineState
from simulator.machine import VirtualMachine

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="\u2699\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)


def _init_history(window: int = 20) -> Dict[str, List[float]]:
    return {
        "temp": list(np.random.normal(settings.BASE_TEMP, 1.0, window)),
        "vib": list(np.random.normal(settings.BASE_VIBRATION, 0.1, window)),
        "current": list(np.random.normal(settings.BASE_CURRENT, 0.3, window)),
        "rpm": list(np.random.normal(settings.BASE_RPM, 10.0, window)),
        "noise": list(np.random.normal(settings.BASE_NOISE, 1.5, window)),
    }


if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "history" not in st.session_state:
    st.session_state.history = _init_history()
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "vm" not in st.session_state:
    st.session_state.vm = VirtualMachine("MACHINE_001")


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
    history_window = st.slider("History Window", 10, 50, 20, 5)

    st.markdown("##### CONTROLS")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.sim_running = True
    with c2:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.sim_running = False
    with c3:
        if st.button("RESET", use_container_width=True):
            st.session_state.history = _init_history(history_window)
            st.session_state.last_frame = None
            st.session_state.vm = VirtualMachine(machine_id)
            st.session_state.sim_running = False

    st.markdown("---")
    status_label = "RUNNING" if st.session_state.sim_running else "PAUSED"
    st.markdown(f"**Status:** `{status_label}`")
    st.caption("Simulated telemetry \u00b7 Experimental models")


if st.session_state.vm.machine_id != machine_id:
    st.session_state.vm = VirtualMachine(machine_id)

_STATE_MAP = {
    "RUNNING": MachineState.NORMAL,
    "IDLE": MachineState.NORMAL,
    "MAINTENANCE": MachineState.WARNING,
}
_FAIL_MAP = {
    "NORMAL_OPERATION": (MachineState.NORMAL, FailureMode.NORMAL_OPERATION),
    "BEARING_FAILURE": (MachineState.CRITICAL, FailureMode.BEARING_FAILURE),
    "OVERHEATING": (MachineState.CRITICAL, FailureMode.OVERHEATING),
    "ELECTRICAL_FAULT": (MachineState.CRITICAL, FailureMode.ELECTRICAL_FAULT),
    "IMBALANCE": (MachineState.DEGRADING, FailureMode.IMBALANCE),
}

if failure_mode_ui in _FAIL_MAP:
    state_enum, fail_enum = _FAIL_MAP[failure_mode_ui]
    st.session_state.vm.set_condition(state_enum, fail_enum)
else:
    st.session_state.vm.set_condition(
        _STATE_MAP.get(machine_state_ui, MachineState.NORMAL),
        FailureMode.NORMAL_OPERATION,
    )


def _provisional_scores(telemetry: Dict[str, Any], is_failure: bool) -> Dict[str, Any]:
    """Provisional UI scores. Full ML pipeline lives in FastAPI /predict."""
    temp = telemetry["temperature"]
    vib = telemetry["vibration"]
    current = telemetry["current"]

    score = 100.0
    vib_dev = max(0.0, (vib - settings.BASE_VIBRATION) / settings.BASE_VIBRATION)
    temp_dev = max(0.0, (temp - settings.BASE_TEMP) / settings.BASE_TEMP)
    curr_dev = max(0.0, (current - settings.BASE_CURRENT) / settings.BASE_CURRENT)
    score -= min(30.0, vib_dev * 20.0)
    score -= min(30.0, temp_dev * 25.0)
    score -= min(20.0, curr_dev * 15.0)
    if is_failure:
        score = min(score, 48.0)
    score = max(0.0, min(100.0, score))

    if score >= 90:
        risk = "HEALTHY"
    elif score >= 70:
        risk = "NORMAL"
    elif score >= 50:
        risk = "WARNING"
    elif score >= 25:
        risk = "CRITICAL"
    else:
        risk = "FAILURE RISK"

    anomaly_score = min(1.0, (vib_dev + temp_dev + curr_dev) / 3.0)
    if is_failure:
        anomaly_score = max(anomaly_score, 0.75)

    degradation_speed = 1.0 + vib_dev + temp_dev
    rul = max(0.0, (score / 100.0) * 720.0 / degradation_speed)

    return {
        "health_score": round(score, 1),
        "risk_level": risk,
        "anomaly_score": round(anomaly_score, 3),
        "is_anomaly": is_failure or anomaly_score > 0.55,
        "rul_hours": round(rul, 1),
    }


if st.session_state.sim_running:
    frame = st.session_state.vm.generate_telemetry()
    is_anomaly = frame["failure_mode"] != FailureMode.NORMAL_OPERATION.value
    scores = _provisional_scores(frame, is_anomaly)
    st.session_state.last_frame = {**frame, **scores}

    hist = st.session_state.history
    for key, sensor in [
        ("temp", "temperature"),
        ("vib", "vibration"),
        ("current", "current"),
        ("rpm", "rpm"),
        ("noise", "noise"),
    ]:
        hist[key].append(frame[sensor])
        while len(hist[key]) > history_window:
            hist[key].pop(0)

frame = st.session_state.last_frame
has_data = frame is not None

if has_data:
    temp = frame["temperature"]
    vib = frame["vibration"]
    current = frame["current"]
    rpm = frame["rpm"]
    noise = frame["noise"]
    health_score = frame["health_score"]
    risk_level = frame["risk_level"]
    anomaly_score = frame["anomaly_score"]
    is_anomaly = frame["is_anomaly"]
    rul_hours = frame["rul_hours"]
    active_failure = frame["failure_mode"]
else:
    temp = vib = current = rpm = noise = 0.0
    health_score = 0.0
    risk_level = "\u2014"
    anomaly_score = 0.0
    is_anomaly = False
    rul_hours = 0.0
    active_failure = "\u2014"


st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:baseline;
                border-bottom:1px solid #2A2F38;padding-bottom:10px;margin-bottom:14px;">
        <div>
            <h2 style="font-size:1.1rem;font-weight:600;margin:0;letter-spacing:0.02em;">
                INDUSTRIAL PREDICTIVE MAINTENANCE
            </h2>
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

if not has_data:
    st.markdown(
        """
        <div class="ind-card" style="text-align:center;padding:28px 16px;">
            <div class="ind-card-header">NO TELEMETRY DATA</div>
            <p style="color:#9A9FA8;font-size:0.9rem;margin:8px 0 0 0;">
                Start the virtual machine to begin collecting simulated telemetry.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


st.markdown(
    f"""
    <div class="ind-card" style="display:flex;justify-content:space-between;align-items:center;padding:10px 16px;flex-wrap:wrap;gap:8px;">
        <span style="font-size:0.8rem;color:#9A9FA8;">MACHINE STATUS:
            <strong style="color:#F2F2F2;">{machine_state_ui}</strong></span>
        <span style="font-size:0.8rem;color:#9A9FA8;">FAILURE MODE:
            <strong style="color:{'#D95C5C' if is_anomaly else '#4CAF78'};">{active_failure}</strong></span>
        <span style="font-size:0.8rem;color:#9A9FA8;">ANOMALY STATUS:
            <strong style="color:{'#D95C5C' if is_anomaly else '#4CAF78'};">
                {'ANOMALY DETECTED' if is_anomaly else 'NORMAL'}
            </strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    filled = int(health_score / 5)
    bar = "\u2588" * filled + "\u2591" * (20 - filled)
    condition = risk_level if risk_level != "\u2014" else "UNKNOWN"
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">HEALTH SCORE</div>
            <div class="ind-card-value">{health_score} <span class="ind-card-unit">/ 100</span></div>
            <div style="font-family:monospace;font-size:0.65rem;color:#D4A84F;margin:4px 0;">{bar}</div>
            <div class="ind-card-desc">{condition} CONDITION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    risk_cls = (
        "badge-success"
        if risk_level in ("HEALTHY", "NORMAL")
        else ("badge-warning" if risk_level == "WARNING" else "badge-critical")
    )
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">RISK LEVEL</div>
            <div class="ind-card-value" style="font-size:1.2rem;padding-top:4px;">
                <span class="badge {risk_cls}">{risk_level}</span>
            </div>
            <div class="ind-card-desc">Operational risk status</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    anom_cls = "badge-critical" if is_anomaly else "badge-success"
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">ANOMALY SCORE</div>
            <div class="ind-card-value">{anomaly_score:.3f}</div>
            <div class="ind-card-desc">
                <span class="badge {anom_cls}">{'ACTIVE' if is_anomaly else 'NOMINAL'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    pred = active_failure if is_anomaly else "NONE"
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">PREDICTED FAILURE</div>
            <div class="ind-card-value" style="font-size:0.95rem;padding-top:6px;
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{pred}</div>
            <div class="ind-card-desc">Experimental classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""
        <div class="ind-card">
            <div class="ind-card-header">RUL</div>
            <div class="ind-card-value">{rul_hours:.0f} <span class="ind-card-unit">HRS</span></div>
            <div class="ind-card-desc">Estimated Remaining Useful Life</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("#### TELEMETRY")
st.caption("SIMULATED TELEMETRY \u00b7 units are synthetic reference values (\u00b0C, mm/s, A, RPM, dB)")

t1, t2, t3, t4, t5 = st.columns(5)
_sensor_cards = [
    (t1, "Temperature", f"{temp:.1f}", "\u00b0C"),
    (t2, "Vibration", f"{vib:.2f}", "mm/s"),
    (t3, "Current", f"{current:.1f}", "A"),
    (t4, "RPM", f"{rpm:.0f}", "RPM"),
    (t5, "Noise", f"{noise:.1f}", "dB"),
]
for col, label, value, unit in _sensor_cards:
    col.markdown(
        f"""
        <div class="ind-card" style="padding:10px;">
            <div class="ind-card-header">{label}</div>
            <div class="ind-card-value" style="font-size:1.1rem;">
                {value} <span class="ind-card-unit">{unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

c_col1, c_col2 = st.columns(2)
hist = st.session_state.history

with c_col1:
    fig_tv = go.Figure()
    fig_tv.add_trace(
        go.Scatter(y=hist["temp"], name="Temperature", line=dict(color="#D4A84F", width=1.5))
    )
    fig_tv.add_trace(
        go.Scatter(
            y=hist["vib"], name="Vibration", line=dict(color="#9A9FA8", width=1.5), yaxis="y2"
        )
    )
    fig_tv.update_layout(
        yaxis=dict(title="Temp (\u00b0C)"),
        yaxis2=dict(title="Vib (mm/s)", overlaying="y", side="right"),
    )
    apply_industrial_plotly_theme(fig_tv, height=240)
    st.plotly_chart(fig_tv, use_container_width=True)

with c_col2:
    fig_cn = go.Figure()
    fig_cn.add_trace(
        go.Scatter(y=hist["current"], name="Current", line=dict(color="#4CAF78", width=1.5))
    )
    fig_cn.add_trace(
        go.Scatter(
            y=hist["noise"], name="Noise", line=dict(color="#9A9FA8", width=1.5), yaxis="y2"
        )
    )
    fig_cn.update_layout(
        yaxis=dict(title="Current (A)"),
        yaxis2=dict(title="Noise (dB)", overlaying="y", side="right"),
    )
    apply_industrial_plotly_theme(fig_cn, height=240)
    st.plotly_chart(fig_cn, use_container_width=True)

st.markdown("#### FAILURE DIAGNOSTICS")

diag_labels = [
    "Bearing Failure",
    "Overheating",
    "Electrical Fault",
    "Imbalance",
    "Normal Operation",
]
if active_failure == "BEARING_FAILURE":
    probs = [88.5, 6.2, 3.1, 1.5, 0.7]
elif active_failure == "OVERHEATING":
    probs = [4.1, 89.3, 4.2, 1.4, 1.0]
elif active_failure == "ELECTRICAL_FAULT":
    probs = [3.0, 4.5, 87.0, 3.5, 2.0]
elif active_failure == "IMBALANCE":
    probs = [5.0, 3.0, 2.5, 86.0, 3.5]
else:
    probs = [2.1, 1.5, 1.0, 0.5, 94.9]

pairs = sorted(zip(probs, diag_labels), reverse=True)
probs_s, labels_s = zip(*pairs)

fig_diag = go.Figure(
    go.Bar(
        x=list(probs_s),
        y=list(labels_s),
        orientation="h",
        marker_color=["#D95C5C" if p > 50 else "#2A2F38" for p in probs_s],
    )
)
fig_diag.update_layout(xaxis=dict(title="Probability (%)", range=[0, 100]))
apply_industrial_plotly_theme(fig_diag, height=200)
st.plotly_chart(fig_diag, use_container_width=True)

m1, m2 = st.columns(2)

with m1:
    st.markdown("#### MACHINE CONDITION")
    st.markdown(
        f"""
        <div class="ind-card" style="font-family:SFMono-Regular,Consolas,monospace;font-size:0.75rem;">
            <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A2F38;">
                <span style="color:#9A9FA8;">Machine ID</span>
                <span style="color:#F2F2F2;">{machine_id}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A2F38;">
                <span style="color:#9A9FA8;">State</span>
                <span style="color:#D4A84F;">{machine_state_ui}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A2F38;">
                <span style="color:#9A9FA8;">Failure Mode</span>
                <span style="color:{'#D95C5C' if is_anomaly else '#4CAF78'};">{active_failure}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A2F38;">
                <span style="color:#9A9FA8;">Health / Risk</span>
                <span style="color:#F2F2F2;">{health_score} / {risk_level}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:4px 0;">
                <span style="color:#9A9FA8;">RUL (est.)</span>
                <span style="color:#F2F2F2;">{rul_hours:.0f} hrs</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown("#### MAINTENANCE INSIGHT")
    if health_score >= 90:
        insight = "No immediate maintenance action required."
        badge = "badge-success"
    elif health_score >= 70:
        insight = "Monitor vibration and temperature trend."
        badge = "badge-info"
    elif health_score >= 50:
        insight = "Maintenance inspection recommended."
        badge = "badge-warning"
    else:
        insight = "Immediate maintenance intervention recommended."
        badge = "badge-critical"

    st.markdown(
        f"""
        <div class="ind-card" style="padding:16px;">
            <div style="margin-bottom:6px;">
                <span class="badge {badge}">EXPERIMENTAL ADVISORY</span>
            </div>
            <p style="font-size:0.85rem;color:#F2F2F2;margin:0 0 8px 0;">{insight}</p>
            <p style="font-size:0.7rem;color:#9A9FA8;margin:0;">
                Simulation-based estimate. Not a real industrial prognosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.sim_running:
    time.sleep(refresh_interval)
    st.rerun()
