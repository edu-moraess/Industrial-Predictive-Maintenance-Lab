import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# Configuração da Página
st.set_page_config(
    page_title="SCADA Industrial Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Dark Mode SCADA
st.markdown("""
<style>
    .stApp { background-color: #0B0F17; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #111723; border-right: 1px solid #1E293B; }
    .scada-card {
        background-color: #151C28;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .scada-label {
        font-size: 0.70rem;
        font-family: monospace;
        color: #94A3B8;
        text-transform: uppercase;
    }
    .scada-value {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: monospace;
        color: #FFFFFF;
        margin-top: 2px;
    }
    .scada-unit { font-size: 0.8rem; color: #00E5FF; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)

# Menu Lateral (Controles Operacionais)
st.sidebar.markdown("### 🎛️ CONTROLE OPERACIONAL")
machine_id = st.sidebar.selectbox("Ativo Monitorado", ["MACHINE_001", "MACHINE_002", "MACHINE_003"])
simular_falha = st.sidebar.checkbox("Injetar Anomalia no Sistema", value=False)
intervalo = st.sidebar.slider("Intervalo de Atualização (s)", 0.5, 5.0, 2.0)

# Inicializa histórico na sessão
if "history_temp" not in st.session_state:
    st.session_state.history_temp = list(np.random.normal(42, 2, 15))
if "history_vib" not in st.session_state:
    st.session_state.history_vib = list(np.random.normal(1.8, 0.3, 15))

# Atualiza dados simulados a cada ciclo
if simular_falha:
    temp = round(82.0 + np.random.uniform(-3, 5), 1)
    vib = round(7.5 + np.random.uniform(-1, 2), 2)
    corr, rpm, ruida = 27.5, 1610.0, 89.0
    health, rul, risk, anomaly = 38.5, 120.0, "CRÍTICO", "DETECTADA"
    status_color = "#F43F5E"
else:
    temp = round(42.0 + np.random.uniform(-2, 2), 1)
    vib = round(1.8 + np.random.uniform(-0.2, 0.4), 2)
    corr, rpm, ruida = 14.8, 1788.0, 53.8
    health, rul, risk, anomaly = 94.2, 580.0, "SAUDÁVEL", "NORMAL"
    status_color = "#10B981"

st.session_state.history_temp.append(temp)
st.session_state.history_temp.pop(0)
st.session_state.history_vib.append(vib)
st.session_state.history_vib.pop(0)

# Cabeçalho Principal
st.markdown(f"## ⚡ SALA DE CONTROLE SCADA — LINHA 01")
st.caption(f"Telemetria ao vivo · Modelo ML ativo · Unidade **{machine_id}**")

# KPI Top Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'<div class="scada-card"><div class="scada-label">Integridade</div><div class="scada-value" style="color: {status_color};">{health}%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="scada-card"><div class="scada-label">RUL Estimado</div><div class="scada-value">{rul} <span class="scada-unit">h</span></div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="scada-card"><div class="scada-label">Nível de Risco</div><div class="scada-value" style="color: {status_color};">{risk}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="scada-card"><div class="scada-label">Status Anomalia</div><div class="scada-value">{anomaly}</div></div>', unsafe_allow_html=True)

# Sensores
s1, s2, s3, s4, s5 = st.columns(5)
s1.markdown(f'<div class="scada-card"><div class="scada-label">TEMP</div><div class="scada-value">{temp}<span class="scada-unit">°C</span></div></div>', unsafe_allow_html=True)
s2.markdown(f'<div class="scada-card"><div class="scada-label">VIBRAÇÃO</div><div class="scada-value">{vib}<span class="scada-unit">mm/s</span></div></div>', unsafe_allow_html=True)
s3.markdown(f'<div class="scada-card"><div class="scada-label">CORRENTE</div><div class="scada-value">{corr}<span class="scada-unit">A</span></div></div>', unsafe_allow_html=True)
s4.markdown(f'<div class="scada-card"><div class="scada-label">ROTAÇÃO</div><div class="scada-value">{rpm}<span class="scada-unit">RPM</span></div></div>', unsafe_allow_html=True)
s5.markdown(f'<div class="scada-card"><div class="scada-label">RUÍDO</div><div class="scada-value">{ruida}<span class="scada-unit">dB</span></div></div>', unsafe_allow_html=True)

# Gráficos Dinâmicos
g1, g2 = st.columns(2)

fig_telemetria = go.Figure()
fig_telemetria.add_trace(go.Scatter(y=st.session_state.history_temp, name="Temperatura (°C)", line=dict(color="#00E5FF", width=2)))
fig_telemetria.add_trace(go.Scatter(y=st.session_state.history_vib, name="Vibração (mm/s)", line=dict(color="#F59E0B", width=2)))
fig_telemetria.update_layout(
    title="Tendência de Sensores em Tempo Real",
    paper_bgcolor="#151C28", plot_bgcolor="#151C28",
    font=dict(color="#94A3B8"), height=280, margin=dict(l=10, r=10, t=30, b=10)
)

with g1:
    st.plotly_chart(fig_telemetria, use_container_width=True)

probs = [85.0 if simular_falha else 5.0, 10.0, 3.0, 2.0]
fig_falhas = go.Figure(go.Bar(
    x=probs,
    y=["SUPERAQUECIMENTO", "DESEQUILÍBRIO", "FALHA ELÉTRICA", "ROLAMENTO"],
    orientation='h',
    marker_color=['#F43F5E' if p > 50 else '#3B82F6' for p in probs]
))
fig_falhas.update_layout(
    title="Probabilidade de Falhas (ML)",
    paper_bgcolor="#151C28", plot_bgcolor="#151C28",
    font=dict(color="#94A3B8"), height=280, margin=dict(l=10, r=10, t=30, b=10)
)

with g2:
    st.plotly_chart(fig_falhas, use_container_width=True)

# LOOP DE ATUALIZAÇÃO EM TEMPO REAL
time.sleep(intervalo)
st.rerun()
