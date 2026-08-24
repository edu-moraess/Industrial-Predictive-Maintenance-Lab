import pytest
import pandas as pd
from ml.anomaly_detection import AnomalyDetector
from features.engineering import FeatureEngineer
from database.repository import DatabaseRepository
from simulator.generator import DataGenerator

def test_anomaly_detector_training_and_prediction():
    # 1. Cria histórico contendo dados normais e anômalo
    repo = DatabaseRepository()
    generator = DataGenerator(repo)
    machine_id = "MACHINE_ANOMALY_TEST"
    
    # Gera dados contendo degradação e anomalias
    generator.generate_historical_dataset(machine_id, hours=2, frequency_minutes=5)
    raw_data = repo.get_historical_readings(machine_id, limit=50)
    
    # 2. Extrai features
    df_features = FeatureEngineer.process_telemetry(raw_data)
    
    # 3. Executa detecção
    detector = AnomalyDetector(contamination=0.15)
    detector.train(df_features)
    result_df = detector.detect(df_features)
    
    assert 'is_anomaly' in result_df.columns
    assert 'anomaly_score' in result_df.columns
    assert result_df['anomaly_score'].min() >= 0.0
    assert result_df['anomaly_score'].max() <= 1.0
    assert result_df['is_anomaly'].dtype == bool

def test_spike_anomaly_detection():
    """Valida se um pico repentino de vibração/temperatura ativa a flag de anomalia."""
    normal_data = [
        {"vibration": 2.5, "temperature": 45.0, "current": 15.0, "rpm": 1800.0, "noise": 65.0}
        for _ in range(20)
    ]
    # Injeta uma anomalia discrepante no último ponto
    anomaly_point = {"vibration": 15.0, "temperature": 95.0, "current": 35.0, "rpm": 1100.0, "noise": 110.0}
    normal_data.append(anomaly_point)

    df_features = FeatureEngineer.process_telemetry(normal_data)
    detector = AnomalyDetector(contamination=0.05)
    result_df = detector.detect(df_features)

    # O último registro deve ser identificado como anomalia
    last_row = result_df.iloc[-1]
    assert last_row['is_anomaly'] == True
    assert last_row['anomaly_score'] > 0.6
