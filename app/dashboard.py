import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

from app.styles import INDUSTRIAL_THEME_CSS
from app.theme import apply_industrial_plotly_theme

# Page Configuration
st.set_page_config(
    page_title="Industrial Predictive Maintenance Control Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS Styles
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)

# Session State Initialization
if "sim_running" not in st.session_state:
    st.session_state.sim_running = True
if "history_temp" not in st.session_state:
    st.session_state.history_temp = list(np.random.normal(42.0, 1.0, 20))
if "history_vib" not in st.session_state:
    st.session_state.history_vib = list(np.random.normal(1.8, 0.1, 20))
if "history_current" not in st.session_state:
    st.session_state.history_current = list(np.random.normal(14.5, 0.3, 20))

# --- SIDEBAR: CONTROL ---
with st.sidebar:
    st.markdown("#### CONTROL")
    st.markdown("---")
    
    st.markdown("##### MACHINE")
    machine_id = st.selectbox("Machine ID", ["MACHINE_001", "MACHINE_002", "MACHINE_003"], label_visibility="collapsed")
    
    st.markdown("##### SIMULATION")
    machine_state = st.selectbox("Machine State", ["RUNNING", "IDLE", "MAINTENANCE"], label_visibility="collapsed")
    
    st.markdown("##### FAILURE INJECTION")
    failure_mode = st.selectbox("Failure Mode", ["NORMAL_OPERATION", "BEARING_FAILURE", "OVERHEATING", "ELECTRICAL_FAULT", "IMBALANCE"], label_visibility="collapsed")
    
    st.markdown("##### TELEMETRY")
    refresh_interval = st.slider("Refresh Interval (s)", 0.5, 5.0, 2.0, 0.5)
    history_window = st.slider("History Window", 10, 50, 20, 5)
    
    st.markdown("##### CONTROLS")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        if st.button("START", use_container_width=True):
            st.session_state.sim_running = True
    with col_c2:
        if st.button("PAUSE", use_container_width=True):
            st.session_state.sim_running = False
    with col_c3:
        if st.button("RESET", use_container_width=True):
            st.session_state.history_temp = list(np.random.normal(42.0, 1.0, history_window))
            st.session_state.history_vib = list(np.random.normal(1.8, 0.1, history_window))
            st.session_state.history_current = list(np.random.normal(14.5, 0.3, history_window))

    st.markdown("---")
    sim_status_text = "RUNNING" if st.session_state.sim_running else "PAUSED"
    st.markdown(f"**Status:** `{sim_status_text}`")


# --- HEADER ---
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #2A2F38; padding-bottom: 10px; margin-bottom: 15px;">
        <div>
            <h2 style="font-size: 1.1rem; font-weight: 600; margin: 0; letter-spacing: 0.02em;">INDUSTRIAL PREDICTIVE MAINTENANCE</h2>
            <p style="font-family: monospace; font-size: 0.75rem; color: #9A9FA8; margin: 2px 0 0 0;">{machine_id}</p>
        </div>
        <div>
            <span class="badge {'badge-success' if st.session_state.sim_running else 'badge-warning'}">
                {'RUNNING' if st.session_state.sim_running else 'PAUSED'}
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)


# --- SIMULATION LOGIC ---
if st.session_state.sim_running:
    is_anomaly = (failure_mode != "NORMAL_OPERATION")
    if is_anomaly:
        temp = round(82.5 + np.random.uniform(-1, 2), 1)
        vib = round(7.8 + np.random.uniform(-0.3, 0.8), 2)
        current = round(28.4 + np.random.uniform(-0.5, 1), 1)
        rpm = round(1610.0 + np.random.uniform(-10, 10), 1)
        noise = round(89.2 + np.random.uniform(-0.5, 1.5), 1)
        health_score = round(max(5.0, 42.5 - np.random.uniform(0, 3)), 1)
        rul_hours = round(max(10.0, 120.0 - np.random.uniform(0, 5)), 1)
        risk_level = "CRITICAL"
        anomaly_score = 0.892
    else:
        temp = round(42.1 + np.random.uniform(-0.5, 0.5), 1)
        vib = round(1.8 + np.random.uniform(-0.1, 0.1), 2)
        current = round(14.8 + np.random.uniform(-0.2, 0.2), 1)
        rpm = round(1792.0 + np.random.uniform(-3, 3), 1)
        noise = round(54.1 + np.random.uniform(-0.5, 0.5), 1)
        health_score = round(min(100.0, 94.5 + np.random.uniform(-0.5, 0.5)), 1)
        rul_hours = round(580.0 + np.random.uniform(-3, 3), 1)
        risk_level = "HEALTHY" if health_score >= 90 else "NORMAL"
        anomaly_score = 0.042

    st.session_state.history_temp.append(temp)
    if len(st.session_state.history_temp) > history_window: st.session_state.history_temp.pop(0)
    
    st.session_state.history_vib.append(vib)
    if len(st.session_state.history_vib) > history_window: st.session_state.history_vib.pop(0)

    st.session_state.history_current.append(current)
    if len(st.session_state.history_current) > history_window: st.session_state.history_current.pop(0)


# --- MACHINE STATUS BAR ---
st.markdown(f"""
    <div class="ind-card" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 16px;">
        <span style="font-size: 0.8rem; color: #9A9FA8;">MACHINE STATUS: <strong style="color: #F2F2F2;">{machine_state}</strong></span>
        <span style="font-size: 0.8rem; color: #9A9FA8;">FAILURE MODE: <strong style="color: {'#D95C5C' if is_anomaly else '#4CAF78'};">{failure_mode}</strong></span>
        <span style="font-size: 0.8rem; color: #9A9FA8;">ANOMALY STATUS: <strong style="color: {'#D95C5C' if is_anomaly else '#4CAF78'};">{'ANOMALY DETECTED' if is_anomaly else 'NORMAL'}</strong></span>
    </div>
""", unsafe_allow_html=True)


# --- KPI ROW (5 Core Metrics) ---
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    bar_filled = int(health_score / 5)
    bar_empty = 20 - bar_filled
    progress_bar_str = "█" * bar_filled + "░" * bar_empty
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">HEALTH SCORE</div>
            <div class="ind-card-value">{health_score} <span class="ind-card-unit">/ 100</span></div>
            <div style="font-family: monospace; font-size: 0.65rem; color: #D4A84F; margin: 4px 0;">{progress_bar_str}</div>
            <div class="ind-card-desc">{risk_level} CONDITION</div>
        </div>
    """, unsafe_allow_html=True)

with k2:
    risk_badge_cls = "badge-success" if risk_level in ["HEALTHY", "NORMAL"] else ("badge-warning" if risk_level == "WARNING" else "badge-critical")
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">RISK LEVEL</div>
            <div class="ind-card-value" style="font-size: 1.25rem; padding-top: 4px;"><span class="badge {risk_badge_cls}">{risk_level}</span></div>
            <div class="ind-card-desc">Operational risk status</div>
        </div>
    """, unsafe_allow_html=True)

with k3:
    anom_badge_cls = "badge-critical" if is_anomaly else "badge-success"
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">ANOMALY SCORE</div>
            <div class="ind-card-value">{anomaly_score:.3f}</div>
            <div class="ind-card-desc"><span class="badge {anom_badge_cls}">{'ACTIVE' if is_anomaly else 'NOMINAL'}</span></div>
        </div>
    """, unsafe_allow_html=True)

with k4:
    pred_fail = failure_mode if is_anomaly else "NONE"
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">PREDICTED FAILURE</div>
            <div class="ind-card-value" style="font-size: 1.0rem; padding-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{pred_fail}</div>
            <div class="ind-card-desc">Isolation Forest / RF</div>
        </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">RUL</div>
            <div class="ind-card-value">{rul_hours:.0f} <span class="ind-card-unit">HRS</span></div>
            <div class="ind-card-desc">Remaining Useful Life</div>
        </div>
    """, unsafe_allow_html=True)


# --- TELEMETRY ---
st.markdown("#### TELEMETRY")
t1, t2, t3, t4, t5 = st.columns(5)
t1.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">Temperature</div><div class="ind-card-value" style="font-size: 1.1rem;">{temp} <span class="ind-card-unit">sim units</span></div></div>', unsafe_allow_html=True)
t2.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">Vibration</div><div class="ind-card-value" style="font-size: 1.1rem;">{vib} <span class="ind-card-unit">sim units</span></div></div>', unsafe_allow_html=True)
t3.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">Current</div><div class="ind-card-value" style="font-size: 1.1rem;">{current} <span class="ind-card-unit">sim units</span></div></div>', unsafe_allow_html=True)
t4.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">RPM</div><div class="ind-card-value" style="font-size: 1.1rem;">{rpm} <span class="ind-card-unit">sim</span></div></div>', unsafe_allow_html=True)
t5.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">Noise</div><div class="ind-card-value" style="font-size: 1.1rem;">{noise} <span class="ind-card-unit">sim units</span></div></div>', unsafe_allow_html=True)

# Charts
c_col1, c_col2 = st.columns(2)
with c_col1:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(y=st.session_state.history_temp, name="Temperature", line=dict(color="#D4A84F", width=1.5)))
    fig_temp.add_trace(go.Scatter(y=st.session_state.history_vib, name="Vibration", line=dict(color="#9A9FA8", width=1.5), yaxis="y2"))
    fig_temp.update_layout(yaxis=dict(title="Temp"), yaxis2=dict(title="Vib", overlaying="y", side="right"))
    apply_industrial_plotly_theme(fig_temp, height=240)
    st.plotly_chart(fig_temp, use_container_width=True)

with c_col2:
    fig_curr = go.Figure()
    fig_curr.add_trace(go.Scatter(y=st.session_state.history_current, name="Current", line=dict(color="#4CAF78", width=1.5)))
    apply_industrial_plotly_theme(fig_curr, height=240)
    st.plotly_chart(fig_curr, use_container_width=True)


# --- FAILURE DIAGNOSTICS ---
st.markdown("#### FAILURE DIAGNOSTICS")
if is_anomaly:
    probs = [88.5, 6.2, 3.1, 1.5, 0.7] if failure_mode == "BEARING_FAILURE" else [4.1, 89.3, 4.2, 1.4, 1.0]
else:
    probs = [2.1, 1.5, 1.0, 0.5, 94.9]
diag_labels = ["Bearing Failure", "Overheating", "Electrical Fault", "Imbalance", "Normal Operation"]

fig_diag = go.Figure(go.Bar(
    x=probs,
    y=diag_labels,
    orientation='h',
    marker_color=['#D95C5C' if p > 50 else '#2A2F38' for p in probs]
))
fig_diag.update_layout(xaxis=dict(title="Probability (%)", range=[0, 100]))
apply_industrial_plotly_theme(fig_diag, height=200)
st.plotly_chart(fig_diag, use_container_width=True)


# --- MACHINE CONDITION & MAINTENANCE INSIGHT ---
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("#### MACHINE CONDITION")
    st.markdown(f"""
        <div class="ind-card" style="font-family: monospace; font-size: 0.75rem;">
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2A2F38;">
                <span style="color: #9A9FA8;">Machine ID:</span>
                <span style="color: #F2F2F2;">{machine_id}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2A2F38;">
                <span style="color: #9A9FA8;">State:</span>
                <span style="color: #D4A84F;">{machine_state}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2A2F38;">
                <span style="color: #9A9FA8;">Failure Mode:</span>
                <span style="color: {'#D95C5C' if is_anomaly else '#4CAF78'};">{failure_mode}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                <span style="color: #9A9FA8;">Health / Risk:</span>
                <span style="color: #F2F2F2;">{health_score}% / {risk_level}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("#### MAINTENANCE INSIGHT")
    if health_score >= 90:
        insight_text = "No immediate maintenance action required. Asset operating within nominal envelope."
        insight_badge = "badge-success"
    elif health_score >= 70:
        insight_text = "Minor variance detected in telemetry signature. Monitor vibration trends."
        insight_badge = "badge-warning"
    else:
        insight_text = "Critical anomaly detected. Maintenance inspection recommended."
        insight_badge = "badge-critical"
        
    st.markdown(f"""
        <div class="ind-card" style="padding: 16px;">
            <div style="margin-bottom: 6px;"><span class="badge {insight_badge}">EXPERIMENTAL ADVISORY</span></div>
            <p style="font-size: 0.85rem; color: #F2F2F2; margin: 0;">{insight_text}</p>
        </div>
    """, unsafe_allow_html=True)


# Real-time loop
if st.session_state.sim_running:
    time.sleep(refresh_interval)
    st.rerun()
