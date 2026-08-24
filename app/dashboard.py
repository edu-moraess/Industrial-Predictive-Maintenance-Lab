import sys
from pathlib import Path

# Injeta a raiz do repositório no PATH para garantir importações absolutas
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import streamlit as st
import pandas as pd
import plotly.express as px

from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector
from ml.health_score import HealthScoreCalculator
from ml.failure_classifier import FailureClassifier
from ml.rul import RULEstimator
from simulator.machine import VirtualMachine
from simulator.failures import MachineState, FailureMode

st.set_page_config(
    page_title="Industrial Predictive Maintenance Lab",
    page_icon="⚙️",
    layout="wide"
)

# Cache da pipeline de ML
@st.cache_resource
def load_ml_pipeline():
    repo = DatabaseRepository()
    detector = AnomalyDetector()
    classifier = FailureClassifier()
    return repo, detector, classifier

repo, anomaly_detector, classifier = load_ml_pipeline()

# Controle do estado da simulação no Streamlit
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "virtual_machine" not in st.session_state:
    st.session_state.virtual_machine = VirtualMachine("MACHINE_001")

st.sidebar.title("⚙️ Simulação em Tempo Real")

machine_id = st.sidebar.text_input("ID da Máquina", value="MACHINE_001")
if machine_id != st.session_state.virtual_machine.machine_id:
    st.session_state.virtual_machine = VirtualMachine(machine_id)

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Painel de Injeção de Falhas")

selected_state = st.sidebar.selectbox(
    "Estado da Máquina",
    options=[e.value for e in MachineState],
    index=0
)

selected_failure = st.sidebar.selectbox(
    "Modo de Falha a Injetar",
    options=[e.value for e in FailureMode],
    index=0
)

st.session_state.virtual_machine.set_condition(
    MachineState(selected_state),
    FailureMode(selected_failure)
)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Loop de Telemetria")

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)

if col_btn1.button("▶ Start"):
    st.session_state.simulation_running = True

if col_btn2.button("⏸ Pause"):
    st.session_state.simulation_running = False

if col_btn3.button("🔄 Reset"):
    st.session_state.simulation_running = False
    st.session_state.virtual_machine = VirtualMachine(machine_id)

refresh_interval = st.sidebar.slider("Intervalo de Atualização (s)", 0.5, 5.0, 1.0)
history_limit = st.sidebar.slider("Histórico Visível", 20, 200, 50)

# Geração de telemetria se o loop estiver ativo
if st.session_state.simulation_running:
    telemetry = st.session_state.virtual_machine.generate_telemetry()
    repo.upsert_machine(machine_id=machine_id, status=telemetry["state"])
    repo.save_sensor_reading(telemetry)

raw_readings = repo.get_historical_readings(machine_id, limit=history_limit)

if not raw_readings:
    st.warning("Aguardando dados da máquina. Clique em '▶ Start' para iniciar.")
    st.stop()

raw_readings = list(reversed(raw_readings))
df_features = FeatureEngineer.process_telemetry(raw_readings)

if len(df_features) >= 10:
    anomaly_detector.train(df_features)
    classifier.train(df_features)

df_analyzed = anomaly_detector.detect(df_features)

latest_data = df_analyzed.iloc[-1].to_dict()
health_score, risk_level = HealthScoreCalculator.calculate(latest_data)

if classifier.is_fitted:
    predictions, probabilities = classifier.predict(df_analyzed.tail(1))
    current_failure = predictions[0]
    failure_probs = probabilities[0]
else:
    current_failure = "NORMAL_OPERATION"
    failure_probs = {e.value: 0.0 for e in FailureMode}

rul_hours = RULEstimator.estimate(health_score, latest_data)
is_anomaly = bool(latest_data.get("is_anomaly", False))
anomaly_score = float(latest_data.get("anomaly_score", 0.0))

# Visualização de KPIs e Gráficos
st.title("🛠️ Industrial Predictive Maintenance Lab")
st.caption(f"Monitoramento Ativo: **{machine_id}** | Simulação: **{'EXECUTANDO 🟢' if st.session_state.simulation_running else 'PAUSADA 🟡'}**")

st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Health Score", f"{health_score}%")
c2.metric("Risco", risk_level)
c3.metric("Anomalia", "DETECTADA" if is_anomaly else "NORMAL", delta=f"{anomaly_score:.2f}", delta_color="inverse" if is_anomaly else "off")
c4.metric("Falha Prevista", current_failure)
c5.metric("RUL", f"{rul_hours} hrs")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Distribuição de Probabilidade de Falhas")
    df_probs = pd.DataFrame(list(failure_probs.items()), columns=["Modo de Falha", "Probabilidade (%)"])
    fig_prob = px.bar(df_probs, x="Probabilidade (%)", y="Modo de Falha", orientation="h", text_auto=".1f")
    fig_prob.update_layout(showlegend=False, height=280)
    st.plotly_chart(fig_prob, use_container_width=True)

with col_right:
    st.subheader("📈 Telemetria: Vibração & Temperatura")
    fig_telemetry = px.line(df_analyzed, x="timestamp", y=["vibration", "temperature"], title="Vibração (mm/s) vs Temperatura (°C)")
    fig_telemetry.update_layout(height=280)
    st.plotly_chart(fig_telemetry, use_container_width=True)

if st.session_state.simulation_running:
    time.sleep(refresh_interval)
    st.rerun()
