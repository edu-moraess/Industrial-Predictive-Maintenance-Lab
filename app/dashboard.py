import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector
from ml.health_score import HealthScoreCalculator
from ml.failure_classifier import FailureClassifier
from ml.rul import RULEstimator
from simulator.generator import DataGenerator

# Configuração da página Streamlit
st.set_page_config(
    page_title="Industrial Predictive Maintenance Lab",
    page_icon="⚙️",
    layout="wide"
)

# Inicialização de banco e modelos (com cache para performance)
@st.cache_resource
def load_ml_pipeline():
    repo = DatabaseRepository()
    generator = DataGenerator(repo)
    
    # Bootstrap de treino inicial
    generator.generate_historical_dataset("MACHINE_001", hours=12, frequency_minutes=5)
    history = repo.get_historical_readings("MACHINE_001", limit=100)
    
    df_features = FeatureEngineer.process_telemetry(history)
    
    detector = AnomalyDetector()
    detector.train(df_features)
    
    classifier = FailureClassifier()
    classifier.train(df_features)
    
    return repo, detector, classifier

repo, anomaly_detector, classifier = load_ml_pipeline()

# Barra Lateral (Sidebar)
st.sidebar.title("⚙️ Painel de Controle")
machines = repo.get_machines()
machine_list = [m["machine_id"] for m in machines] if machines else ["MACHINE_001"]
selected_machine = st.sidebar.selectbox("Selecione a Máquina", machine_list)

history_limit = st.sidebar.slider("Histórico de Registros", 20, 200, 50)

# Processamento dos Dados em Tempo Real
raw_readings = repo.get_historical_readings(selected_machine, limit=history_limit)

if not raw_readings:
    st.warning("Nenhum dado encontrado para a máquina selecionada.")
    st.stop()

# Pipeline de Engenharia e ML
df_features = FeatureEngineer.process_telemetry(raw_readings)
df_analyzed = anomaly_detector.detect(df_features)

latest_data = df_analyzed.iloc[-1].to_dict()
health_score, risk_level = HealthScoreCalculator.calculate(latest_data)
predictions, probabilities = classifier.predict(df_analyzed.tail(1))
rul_hours = RULEstimator.estimate(health_score, latest_data)

current_failure = predictions[0]
failure_probs = probabilities[0]
is_anomaly = bool(latest_data.get("is_anomaly", False))
anomaly_score = float(latest_data.get("anomaly_score", 0.0))

# Header Principal
st.title("🛠️ Industrial Predictive Maintenance Lab")
st.caption(f"Monitoramento Analítico da Unidade: **{selected_machine}** | Status dos Dados: Sintético / Virtual Lab")

st.markdown("---")

# Seção 1: Overview (KPIs)
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Health Score", f"{health_score}%", delta=f"{health_score - 100:.1f}%", delta_color="normal")
col2.metric("Nível de Risco", risk_level)
col3.metric("Status Anomalia", "DETECTADA" if is_anomaly else "NORMAL", delta=f"Score: {anomaly_score:.2f}", delta_color="inverse" if is_anomaly else "off")
col4.metric("Diagnóstico de Falha", current_failure)
col5.metric("RUL Estimada", f"{rul_hours} hrs")

st.markdown("---")

# Seção 2: Failure Probability Breakdown & Timeline
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Probabilidade de Falhas (ML Classifier)")
    df_probs = pd.DataFrame(list(failure_probs.items()), columns=["Modo de Falha", "Probabilidade (%)"])
    fig_prob = px.bar(
        df_probs, 
        x="Probabilidade (%)", 
        y="Modo de Falha", 
        orientation="h",
        color="Probabilidade (%)",
        color_continuous_scale="Blues",
        text_auto=".1f"
    )
    fig_prob.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=300)
    st.plotly_chart(fig_prob, use_container_width=True)

with col_right:
    st.subheader("📍 Linha de Evolução Operacional")
    st.markdown("""
    **Timeline do Estado Atual:**
    """)
    state_colors = {
        "HEALTHY": "🟢 **Operação Saudável**",
        "NORMAL": "🔵 **Operação Normal**",
        "WARNING": "🟡 **Alerta Inicial** (Atenção recomendada)",
        "CRITICAL": "🟠 **Degradação Crítica** (Ação requerida)",
        "FAILURE RISK": "🔴 **Risco Iminente de Falha** (Parada imediata)"
    }
    st.info(f"Estado Identificado: {state_colors.get(risk_level, risk_level)}")
    
    st.progress(int(health_score) / 100)
    st.caption("Barra de Integridade do Ativo (0% = Falha Total, 100% = Íntegro)")

st.markdown("---")

# Seção 3: Monitoramento dos Sensores (Séries Temporais)
st.subheader("📈 Monitoramento dos Sensores em Tempo Real")

tab_temp, tab_vib, tab_curr, tab_rpm = st.tabs(["Temperatura (°C)", "Vibração (mm/s)", "Corrente (A)", "RPM / Ruído"])

with tab_temp:
    fig_temp = px.line(df_analyzed, x="timestamp", y=["temperature", "temp_rolling_mean"], title="Evolução da Temperatura")
    st.plotly_chart(fig_temp, use_container_width=True)

with tab_vib:
    fig_vib = px.line(df_analyzed, x="timestamp", y=["vibration", "vib_rolling_mean"], title="Evolução da Vibração")
    st.plotly_chart(fig_vib, use_container_width=True)

with tab_curr:
    fig_curr = px.line(df_analyzed, x="timestamp", y=["current", "curr_rolling_mean"], title="Consumo Elétrico (Corrente)")
    st.plotly_chart(fig_curr, use_container_width=True)

with tab_rpm:
    fig_rpm = px.line(df_analyzed, x="timestamp", y=["rpm", "noise"], title="Velocidade e Ruído Acústico")
    st.plotly_chart(fig_rpm, use_container_width=True)

# Seção 4: Tabela de Anomalias Detectadas
st.markdown("---")
st.subheader("🚨 Registro Histórico de Anomalias")
anomalies_df = df_analyzed[df_analyzed["is_anomaly"] == True][["timestamp", "temperature", "vibration", "current", "anomaly_score"]]

if not anomalies_df.empty:
    st.dataframe(anomalies_df.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.success("Nenhuma anomalia crítica registrada no período analisado.")
