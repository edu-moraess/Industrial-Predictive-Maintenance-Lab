import pytest
import pandas as pd
from features.engineering import FeatureEngineer
from database.repository import DatabaseRepository
from simulator.generator import DataGenerator

def test_empty_telemetry():
    """Garante que a função não quebre caso receba uma lista vazia."""
    df = FeatureEngineer.process_telemetry([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_feature_generation():
    """Gera dados reais pelo banco e testa a extração de features."""
    repo = DatabaseRepository()
    generator = DataGenerator(repo)
    
    machine_id = "MACHINE_FEATURE_TEST"
    
    # Gera 30 minutos de dados (6 ciclos de 5 min)
    generator.generate_historical_dataset(machine_id, hours=0.5, frequency_minutes=5)
    
    # Recupera os dados
    raw_data = repo.get_historical_readings(machine_id, limit=10)
    
    # Processa as features
    df = FeatureEngineer.process_telemetry(raw_data, window_size=3)
    
    # Verifica a estrutura básica
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    
    # Verifica se as novas colunas cruciais foram criadas
    expected_columns = [
        'vib_rolling_mean', 'vib_rolling_std', 
        'temp_rolling_mean', 'temp_roc', 
        'vib_trend', 'vib_temp_ratio'
    ]
    
    for col in expected_columns:
        assert col in df.columns
        
    # Verifica se não há valores nulos que poderiam quebrar o Machine Learning
    assert df['vib_rolling_std'].isnull().sum() == 0
    assert df['temp_roc'].isnull().sum() == 0

def test_rate_of_change_logic():
    """Testa de forma isolada se a lógica matemática do ROC e Rolling está correta."""
    mock_data = [
        {"timestamp": "2026-01-01T10:00:00", "vibration": 2.0, "temperature": 40.0, "current": 10.0},
        {"timestamp": "2026-01-01T10:05:00", "vibration": 3.0, "temperature": 40.0, "current": 10.0},
        {"timestamp": "2026-01-01T10:10:00", "vibration": 7.0, "temperature": 40.0, "current": 10.0}
    ]
    
    df = FeatureEngineer.process_telemetry(mock_data, window_size=2)
    
    # O Rate of change (diff) da vibração no índice 1 deve ser 3.0 - 2.0 = 1.0
    assert df.loc[1, 'vib_roc'] == 1.0
    
    # O Rate of change da vibração no índice 2 deve ser 7.0 - 3.0 = 4.0
    assert df.loc[2, 'vib_roc'] == 4.0
    
    # A média móvel no índice 2 (janela 2) deve ser (3.0 + 7.0) / 2 = 5.0
    assert df.loc[2, 'vib_rolling_mean'] == 5.0
