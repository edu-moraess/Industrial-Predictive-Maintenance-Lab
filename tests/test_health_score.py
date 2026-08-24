import pytest
from ml.health_score import HealthScoreCalculator
from config.settings import settings

def test_healthy_machine():
    """Uma máquina operando nos valores nominais deve ser HEALTHY (90-100)."""
    data = {
        "vibration": settings.BASE_VIBRATION,
        "temperature": settings.BASE_TEMP,
        "anomaly_score": 0.01,
        "vib_trend": 0.0
    }
    
    score, risk = HealthScoreCalculator.calculate(data)
    
    assert 90 <= score <= 100
    assert risk == "HEALTHY"

def test_warning_machine():
    """Uma leve elevação e anomalia moderada devem gerar estado de WARNING ou NORMAL."""
    data = {
        "vibration": settings.BASE_VIBRATION * 1.5,  # 50% de aumento
        "temperature": settings.BASE_TEMP * 1.2,     # 20% de aumento
        "anomaly_score": 0.3,                        # Anomalia moderada
        "vib_trend": 0.1
    }
    
    score, risk = HealthScoreCalculator.calculate(data)
    
    # Perdas esperadas: 
    # Anomalia: ~12 pontos
    # Vibração: ~10 pontos
    # Temp: ~5 pontos
    # Esperado: ~73 (NORMAL) ou menos dependendo do ruído.
    assert 50 <= score <= 89
    assert risk in ["NORMAL", "WARNING"]

def test_critical_failure_risk_machine():
    """Alta degradação e anomalia clara devem acionar CRITICAL ou FAILURE RISK."""
    data = {
        "vibration": settings.BASE_VIBRATION * 3.0,  # 200% de aumento
        "temperature": settings.BASE_TEMP * 1.8,     # 80% de aumento
        "anomaly_score": 0.85,                       # Alta anomalia
        "vib_trend": 1.2                             # Tendência forte de alta
    }
    
    score, risk = HealthScoreCalculator.calculate(data)
    
    assert 0 <= score < 50
    assert risk in ["CRITICAL", "FAILURE RISK"]
