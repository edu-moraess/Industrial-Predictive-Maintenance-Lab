import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

from app.styles import INDUSTRIAL_THEME_CSS
from app.theme import apply_industrial_plotly_theme

# Page Configuration
st.set_page_config(
    page_title="Industrial Predictive Maintenance Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS Styles
st.markdown(INDUSTRIAL_THEME_CSS, unsafe_allow_html=True)

# Session State Initialization
if "sim_running" not in st.session_state:
    st.session_state.sim_running = True
if "history_temp" not in st.session_state:
    st.session_state.history_temp = list(np.random.normal(42.0, 1.5, 20))
if "history_vib" not in st.session_state:
    st.session_state.history_vib = list(np.random.normal(1.8, 0.2, 20))
if "history_current" not in st.session_state:
    st.session_state.history_current = list(np.random.normal(14.5, 0.5, 20))

# --- SIDEBAR: INDUSTRIAL CONTROL ---
with st.sidebar:
    st.markdown("### INDUSTRIAL CONTROL")
    st.markdown("---")
    
    st.markdown("##### MACHINE")
    machine_id = st.selectbox("MACHINE ID", ["MACHINE_001", "MACHINE_002", "MACHINE_003"], label_visibility="collapsed")
    
    st.markdown("##### SIMULATION")
    machine_state = st.selectbox("MACHINE STATE", ["RUNNING", "IDLE", "MAINTENANCE"], label_visibility="collapsed")
    failure_mode = st.selectbox("FAILURE MODE", ["NORMAL_OPERATION", "BEARING_FAILURE", "OVERHEATING", "ELECTRICAL_FAULT", "IMBALANCE"], label_visibility="collapsed")
    
    st.markdown("##### TELEMETRY")
    refresh_interval = st.slider("REFRESH INTERVAL (s)", 0.5, 5.0, 2.0, 0.5)
    history_window = st.slider("HISTORY WINDOW", 10, 50, 20, 5)
    
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
            st.session_state.history_temp = list(np.random.normal(42.0, 1.5, history_window))
            st.session_state.history_vib = list(np.random.normal(1.8, 0.2, history_window))
            st.session_state.history_current = list(np.random.normal(14.5, 0.5, history_window))

    st.markdown("---")
    st.markdown("##### STATUS")
    if st.session_state.sim_running:
        st.markdown('<span class="badge badge-success">● ONLINE</span> <span class="badge badge-info">RUNNING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-warning">● OFFLINE</span> <span class="badge badge-warning">PAUSED</span>', unsafe_allow_html=True)


# --- HEADER ---
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #253044; padding-bottom: 12px; margin-bottom: 20px;">
        <div>
            <h1 style="font-size: 1.25rem; font-weight: 700; margin: 0; letter-spacing: -0.025em;">INDUSTRIAL PREDICTIVE MAINTENANCE CONTROL CENTER</h1>
            <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; margin: 4px 0 0 0;">ACTIVE UNIT: <span style="color: #38BDF8;">{machine_id}</span> | PIPELINE: ML-INFERENCE-V2</p>
        </div>
        <div>
            <span class="badge {'badge-success' if st.session_state.sim_running else 'badge-warning'}">
                {'● LIVE SIMULATION' if st.session_state.sim_running else '■ SIMULATION PAUSED'}
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)


# --- TELEMETRY DATA SIMULATION (Preserving Logic) ---
if st.session_state.sim_running:
    is_anomaly = (failure_mode != "NORMAL_OPERATION")
    if is_anomaly:
        temp = round(82.5 + np.random.uniform(-2, 4), 1)
        vib = round(7.8 + np.random.uniform(-0.5, 1.5), 2)
        current = round(28.4 + np.random.uniform(-1, 2), 1)
        rpm = round(1610.0 + np.random.uniform(-20, 20), 1)
        noise = round(89.2 + np.random.uniform(-1, 3), 1)
        health_score = round(max(5.0, 42.5 - np.random.uniform(0, 5)), 1)
        rul_hours = round(max(10.0, 120.0 - np.random.uniform(0, 10)), 1)
        risk_level = "CRITICAL"
        anomaly_score = 0.892
    else:
        temp = round(42.1 + np.random.uniform(-1, 1), 1)
        vib = round(1.8 + np.random.uniform(-0.2, 0.2), 2)
        current = round(14.8 + np.random.uniform(-0.3, 0.3), 1)
        rpm = round(1792.0 + np.random.uniform(-5, 5), 1)
        noise = round(54.1 + np.random.uniform(-1, 1), 1)
        health_score = round(min(100.0, 94.5 + np.random.uniform(-1, 1)), 1)
        rul_hours = round(580.0 + np.random.uniform(-5, 5), 1)
        risk_level = "HEALTHY" if health_score >= 90 else "NORMAL"
        anomaly_score = 0.042

    # Update history buffers
    st.session_state.history_temp.append(temp)
    if len(st.session_state.history_temp) > history_window: st.session_state.history_temp.pop(0)
    
    st.session_state.history_vib.append(vib)
    if len(st.session_state.history_vib) > history_window: st.session_state.history_vib.pop(0)

    st.session_state.history_current.append(current)
    if len(st.session_state.history_current) > history_window: st.session_state.history_current.pop(0)


# --- KPI CARDS (5 Core Metrics) ---
k1, k2, k3, k4, k5 = st.columns(5)

health_badge_class = "badge-success" if health_score >= 70 else ("badge-warning" if health_score >= 50 else "badge-critical")

with k1:
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">HEALTH SCORE</div>
            <div class="ind-card-value">{health_score}<span class="ind-card-unit">%</span></div>
            <div class="ind-card-desc"><span class="badge {health_badge_class}">{risk_level}</span></div>
        </div>
    """, unsafe_allow_html=True)

with k2:
    risk_badge = "badge-success" if risk_level in ["HEALTHY", "NORMAL"] else ("badge-warning" if risk_level == "WARNING" else "badge-critical")
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">RISK LEVEL</div>
            <div class="ind-card-value" style="font-size: 1.35rem; padding-top: 4px;">{risk_level}</div>
            <div class="ind-card-desc">Operational classification</div>
        </div>
    """, unsafe_allow_html=True)

with k3:
    anom_badge = "badge-critical" if is_anomaly else "badge-success"
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">ANOMALY SCORE</div>
            <div class="ind-card-value">{anomaly_score:.3f}</div>
            <div class="ind-card-desc"><span class="badge {anom_badge}">{'ANOMALY DETECTED' if is_anomaly else 'NORMAL OPERATION'}</span></div>
        </div>
    """, unsafe_allow_html=True)

with k4:
    pred_fail = failure_mode if is_anomaly else "NONE"
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">PREDICTED FAILURE</div>
            <div class="ind-card-value" style="font-size: 1.1rem; padding-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{pred_fail}</div>
            <div class="ind-card-desc">Isolation Forest / RF</div>
        </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
        <div class="ind-card">
            <div class="ind-card-header">REMAINING USEFUL LIFE</div>
            <div class="ind-card-value">{rul_hours:.0f}<span class="ind-card-unit">HRS</span></div>
            <div class="ind-card-desc">Projected lifespan</div>
        </div>
    """, unsafe_allow_html=True)


# --- TELEMETRY READOUTS BAR ---
st.markdown("##### LIVE TELEMETRY SENSORS")
t1, t2, t3, t4, t5 = st.columns(5)
t1.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">TEMPERATURE</div><div class="ind-card-value" style="font-size: 1.2rem;">{temp} <span class="ind-card-unit">°C</span></div></div>', unsafe_allow_html=True)
t2.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">VIBRATION</div><div class="ind-card-value" style="font-size: 1.2rem;">{vib} <span class="ind-card-unit">mm/s</span></div></div>', unsafe_allow_html=True)
t3.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">CURRENT</div><div class="ind-card-value" style="font-size: 1.2rem;">{current} <span class="ind-card-unit">A</span></div></div>', unsafe_allow_html=True)
t4.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">RPM</div><div class="ind-card-value" style="font-size: 1.2rem;">{rpm} <span class="ind-card-unit">RPM</span></div></div>', unsafe_allow_html=True)
t5.markdown(f'<div class="ind-card" style="padding: 10px;"><div class="ind-card-header">NOISE LEVEL</div><div class="ind-card-value" style="font-size: 1.2rem;">{noise} <span class="ind-card-unit">dB</span></div></div>', unsafe_allow_html=True)


# --- ANALYTICS CHARTS (PROCESS TRENDS & FAILURE DIAGNOSTICS) ---
c_col1, c_col2 = st.columns(2)

with c_col1:
    st.markdown("##### PROCESS TRENDS (TEMP & VIBRATION)")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(y=st.session_state.history_temp, name="Temperature (°C)", line=dict(color="#38BDF8", width=2)))
    fig_temp.add_trace(go.Scatter(y=st.session_state.history_vib, name="Vibration (mm/s)", line=dict(color="#F59E0B", width=2), yaxis="y2"))
    
    fig_temp.update_layout(
        yaxis=dict(title="Temp (°C)", titlefont=dict(color="#38BDF8"), tickfont=dict(color="#38BDF8")),
        yaxis2=dict(title="Vib (mm/s)", titlefont=dict(color="#F59E0B"), tickfont=dict(color="#F59E0B"), overlaying="y", side="right")
    )
    apply_industrial_plotly_theme(fig_temp, height=260)
    st.plotly_chart(fig_temp, use_container_width=True)

with c_col2:
    st.markdown("##### FAILURE DIAGNOSTICS (PROBABILITY)")
    if is_anomaly:
        probs = [88.5, 6.2, 3.1, 1.5, 0.7] if failure_mode == "BEARING_FAILURE" else [4.1, 89.3, 4.2, 1.4, 1.0]
    else:
        probs = [2.1, 1.5, 1.0, 0.5, 94.9]
        
    diag_labels = ["Bearing Failure", "Overheating", "Electrical Fault", "Imbalance", "Normal Operation"]
    
    fig_diag = go.Figure(go.Bar(
        x=probs,
        y=diag_labels,
        orientation='h',
        marker_color=['#EF4444' if p > 50 else '#38BDF8' for p in probs]
    ))
    fig_diag.update_layout(xaxis=dict(title="Probability (%)", range=[0, 100]))
    apply_industrial_plotly_theme(fig_diag, height=260)
    st.plotly_chart(fig_diag, use_container_width=True)


# --- MAINTENANCE INSIGHT & MACHINE CONDITION ---
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("##### MAINTENANCE INSIGHT")
    if health_score >= 90:
        insight_text = "No immediate maintenance action required. Asset operating within nominal envelope."
        insight_badge = "badge-success"
    elif health_score >= 70:
        insight_text = "Minor variance detected in vibration signature. Monitor thermal trends."
        insight_badge = "badge-warning"
    else:
        insight_text = "Critical anomaly detected. Immediate technician inspection and maintenance intervention recommended."
        insight_badge = "badge-critical"
        
    st.markdown(f"""
        <div class="ind-card" style="padding: 18px;">
            <div style="margin-bottom: 8px;"><span class="badge {insight_badge}">DIAGNOSTIC ADVISORY</span></div>
            <p style="font-size: 0.9rem; color: #F8FAFC; margin: 0; font-family: 'Inter', sans-serif;">{insight_text}</p>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("##### MACHINE CONDITION SUMMARY")
    st.markdown(f"""
        <div class="ind-card" style="padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #253044;">
                <span style="color: #94A3B8;">MACHINE STATE:</span>
                <span style="color: #38BDF8; font-weight: 600;">{machine_state}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #253044;">
                <span style="color: #94A3B8;">ACTIVE FAILURE MODE:</span>
                <span style="color: {'#EF4444' if is_anomaly else '#22C55E'}; font-weight: 600;">{failure_mode}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #253044;">
                <span style="color: #94A3B8;">ANOMALY STATUS:</span>
                <span style="color: {'#EF4444' if is_anomaly else '#22C55E'}; font-weight: 600;">{'ACTIVE WARNING' if is_anomaly else 'NORMAL'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                <span style="color: #94A3B8;">ESTIMATED RUL:</span>
                <span style="color: #F8FAFC; font-weight: 600;">{rul_hours:.1f} HOURS</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Real-time streaming loop
if st.session_state.sim_running:
    time.sleep(refresh_interval)
    st.rerun()
