from fastapi import FastAPI, HTTPException, status
from typing import List, Dict, Any
import pandas as pd

from api.schemas import SensorInput, PredictResponse, MachineResponse
from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector
from ml.health_score import HealthScoreCalculator
from ml.failure_classifier import FailureClassifier
from ml.rul import RULEstimator
from simulator.generator import DataGenerator

app = FastAPI(
    title="Industrial Predictive Maintenance Lab API",
    version="1.0.0",
    description="API de telemetria, detecção de anomalias e manutenção preditiva industrial."
)

# Inicialização dos serviços e ML Engine
repo = DatabaseRepository()
anomaly_detector = AnomalyDetector()
classifier = FailureClassifier()

# Bootstrapping de treino dos modelos com histórico sintético inicial
def _bootstrap_models():
    generator = DataGenerator(repo)
    generator.generate_historical_dataset("INIT_TRAIN_MACHINE", hours=6, frequency_minutes=5)
    history = repo.get_historical_readings("INIT_TRAIN_MACHINE", limit=100)
    
    if history:
        df_features = FeatureEngineer.process_telemetry(history)
        anomaly_detector.train(df_features)
        classifier.train(df_features)

_bootstrap_models()

@app.get("/health", status_code=status.HTTP_200_OK)
def get_health_status() -> Dict[str, str]:
    """Endpoint de verificação de disponibilidade da API."""
    return {"status": "online", "service": "Industrial Predictive Maintenance Engine"}

@app.get("/machines", response_model=List[MachineResponse])
def get_machines():
    """Retorna a lista de máquinas registradas."""
    return repo.get_machines()

@app.get("/machines/{machine_id}")
def get_machine_details(machine_id: str):
    """Retorna os dados cadastrais de uma máquina específica."""
    machines = repo.get_machines()
    machine = next((m for m in machines if m["machine_id"] == machine_id), None)
    if not machine:
        raise HTTPException(status_code=404, detail="Máquina não encontrada.")
    return machine

@app.get("/machines/{machine_id}/latest")
def get_latest_telemetry(machine_id: str):
    """Retorna a última leitura de sensor enviada para a máquina."""
    reading = repo.get_latest_reading(machine_id)
    if not reading:
        raise HTTPException(status_code=404, detail="Nenhuma leitura encontrada para esta máquina.")
    return reading

@app.post("/predict", response_model=PredictResponse)
def predict_machine_health(telemetry: SensorInput):
    """Processa leituras brutas de sensores e calcula anomalias, saúde, tipo de falha e RUL."""
    # Recupera contexto histórico recente para calcular estatísticas de janela
    history = repo.get_historical_readings(telemetry.machine_id, limit=20)
    
    current_dict = telemetry.model_dump()
    raw_list = history + [current_dict]
    
    # Processa features
    df_features = FeatureEngineer.process_telemetry(raw_list)
    latest_row_df = df_features.tail(1)
    
    # Detecção de Anomalias
    df_anomaly = anomaly_detector.detect(latest_row_df)
    is_anomaly = bool(df_anomaly.iloc[0]['is_anomaly'])
    anomaly_score = float(df_anomaly.iloc[0]['anomaly_score'])
    
    # Cálculo do Health Score
    latest_feature_dict = df_anomaly.iloc[0].to_dict()
    health_score, risk_level = HealthScoreCalculator.calculate(latest_feature_dict)
    
    # Classificação da Causa Raiz de Falha
    predictions, probabilities = classifier.predict(df_anomaly)
    predicted_failure = predictions[0]
    failure_probs = probabilities[0]
    
    # Estimativa de RUL
    rul_hours = RULEstimator.estimate(health_score, latest_feature_dict)
    
    return PredictResponse(
        machine_id=telemetry.machine_id,
        health_score=health_score,
        risk_level=risk_level,
        anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        failure_type=predicted_failure,
        failure_probabilities=failure_probs,
        rul_hours=rul_hours
    )

@app.get("/anomalies")
def get_anomalies(machine_id: str, limit: int = 20):
    """Retorna o histórico de telemetria identificando pontos anômalos."""
    history = repo.get_historical_readings(machine_id, limit=limit)
    if not history:
        return []
        
    df_features = FeatureEngineer.process_telemetry(history)
    df_anomalies = anomaly_detector.detect(df_features)
    
    anomalies_only = df_anomalies[df_anomalies['is_anomaly'] == True]
    return anomalies_only.to_dict(orient="records")
