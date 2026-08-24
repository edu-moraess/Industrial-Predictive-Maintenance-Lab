import pytest
from ml.rul import RULEstimator

def test_healthy_machine_rul():
    """Máquina com 100% de saúde e sem degradação deve retornar RUL máxima."""
    health_score = 100.0
    telemetry = {"vib_trend": 0.0, "temp_trend": 0.0}
    
    rul = RULEstimator.estimate(health_score, telemetry)
    assert rul == 720.0

def test_degraded_machine_rul():
    """Máquina com perda de saúde e forte degradação deve ter a RUL penalizada."""
    health_score = 50.0
    telemetry = {"vib_trend": 1.2, "temp_trend": 0.8}
    
    rul = RULEstimator.estimate(health_score, telemetry)
    
    # Com 50% de saúde e tendência de alta, RUL deve cair bem abaixo da metade (360h)
    assert 0.0 < rul < 360.0

def test_critical_failure_rul():
    """Máquina com saúde zerada deve indicar RUL nula."""
    health_score = 0.0
    telemetry = {"vib_trend": 5.0}
    
    rul = RULEstimator.estimate(health_score, telemetry)
    assert rul == 0.0
