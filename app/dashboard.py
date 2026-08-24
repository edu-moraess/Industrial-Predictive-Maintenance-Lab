import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# Configuração da Página
st.set_page_config(
    page_title="SCADA Industrial Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Dark Mode SCADA (Impede cortes de texto e formata cards)
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
        letter-spacing: 0.5px;
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
intervalo = st.sidebar.slider("Intervalo de Leitura (s)", 0.5, 5.0, 2.5)

# Cabeçalho Principal
st.markdown(f"## ⚡ SALA DE CONTROLE SCADA — LINHA 01")
st.caption(f"Telemetria em tempo real · Modelo ML ativo · Unidade **{machine_id}**")

# Simulação de dados para visualização
if simular_falha:
    temp, vib, corr, rpm, ruida = 84.2, 7.8, 27.5, 1610.0, 89.0
    health, rul, risk, anomaly = 38.5, 120.0, "CRÍTICO", "DETECTADA"
    status_color = "#F43F5E"
else:
    temp, vib, corr, rpm, ruida = 42.1, 1.8, 14.8, 1788.0, 53.8
    health, rul, risk, anomaly = 94.2, 580.0, "SAUDÁVEL", "NORMAL"
    status_color = "#10B981"

# KPI Top Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="scada-card">
        <div class="scada-label">Pontuação de Saúde</div>
        <div class="scada-value" style="color: {status_color};">{health}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="scada-card">
        <div class="scada-label">RUL Estimado</div>
        <div class="scada-value">{rul} <span class="scada-unit">h</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="scada-card">
        <div class="scada-label">Nível de Risco</div>
        <div class="scada-value" style="color: {status_color};">{risk}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="scada-card">
        <div class="scada-label">Status Anomalia</div>
        <div class="scada-value">{anomaly}</div>
    </div>
    """, unsafe_allow_html=True)

# Leituras dos Sensores de Campo (Sem texto cortado)
st.markdown("#### 📡 Leitura de Sensores de Campo")
s1, s2, s3, s4, s5 = st.columns(5)

s1.markdown(f'<div class="scada-card"><div class="scada-label">TEMP</div><div class="scada-value">{temp}<span class="scada-unit">°C</span></div></div>', unsafe_allow_html=True)
s2.markdown(f'<div class="scada-card"><div class="scada-label">VIBRAÇÃO</div><div class="scada-value">{vib}<span class="scada-unit">mm/s</span></div></div>', unsafe_allow_html=True)
s3.markdown(f'<div class="scada-card"><div class="scada-label">CORRENTE</div><div class="scada-value">{corr}<span class="scada-unit">A</span></div></div>', unsafe_allow_html=True)
s4.markdown(f'<div class="scada-card"><div class="scada-label">ROTAÇÃO</div><div class="scada-value">{rpm}<span class="scada-unit">RPM</span></div></div>', unsafe_allow_html=True)
s5.markdown(f'<div class="scada-card"><div class="scada-label">RUÍDO</div><div class="scada-value">{ruida}<span class="scada-unit">dB</span></div></div>', unsafe_allow_html=True)

# Gráficos em Plotly (Escuros e Profissionais)
g1, g2 = st.columns(2)

# Gráfico 1: Telemetria
time_stamps = [datetime.now().strftime("%H:%M:%S") for _ in range(10)]
fig_telemetria = go.Figure()
fig_telemetria.add_trace(go.Scatter(y=[temp + np.random.randn() for _ in range(10)], name="Temperatura (°C)", line=dict(color="#00E5FF", width=2)))
fig_telemetria.add_trace(go.Scatter(y=[vib * 10 + np.random.randn() for _ in range(10)], name="Vibração (x10 mm/s)", line=dict(color="#F59E0B", width=2)))
fig_telemetria.update_layout(
    title="Tendência de Processo em Tempo Real",
    paper_bgcolor="#151C28", plot_bgcolor="#151C28",
    font=dict(color="#94A3B8"), height=300, margin=dict(l=20, r=20, t=40, b=20)
)

with g1:
    st.plotly_chart(fig_telemetria, use_container_width=True)

# Gráfico 2: Matriz de Falhas
probs = [85.0 if simular_falha else 5.0, 10.0, 3.0, 2.0]
fig_falhas = go.Figure(go.Bar(
    x=probs,
    y=["SOPERAQUECIMENTO", "DESEQUILÍBRIO", "FALHA ELÉTRICA", "ROLAMENTO"],
    orientation='h',
    marker_color=['#F43F5E' if p > 50 else '#3B82F6' for p in probs]
))
fig_falhas.update_layout(
    title="Probabilidade de Falhas (Random Forest)",
    paper_bgcolor="#151C28", plot_bgcolor="#151C28",
    font=dict(color="#94A3B8"), height=300, margin=dict(l=20, r=20, t=40, b=20)
)

with g2:
    st.plotly_chart(fig_falhas, use_container_width=True)
