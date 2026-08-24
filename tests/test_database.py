import pytest
from database.repository import DatabaseRepository
from simulator.generator import DataGenerator

def test_database_and_repository():
    repo = DatabaseRepository()
    
    # Testa cadastro de máquina
    repo.upsert_machine("MACHINE_TEST", name="Test Unit", status="NORMAL")
    machines = repo.get_machines()
    assert any(m["machine_id"] == "MACHINE_TEST" for m in machines)
    
    # Testa salvamento e recuperação de leitura
    test_reading = {
        "timestamp": "2026-06-06T12:00:00+00:00",
        "machine_id": "MACHINE_TEST",
        "state": "NORMAL",
        "failure_mode": "NORMAL_OPERATION",
        "temperature": 45.5,
        "vibration": 2.1,
        "current": 15.0,
        "rpm": 1800.0,
        "noise": 65.0
    }
    
    repo.save_sensor_reading(test_reading)
    latest = repo.get_latest_reading("MACHINE_TEST")
    
    assert latest is not None
    assert latest["machine_id"] == "MACHINE_TEST"
    assert latest["temperature"] == 45.5

def test_historical_generator():
    repo = DatabaseRepository()
    generator = DataGenerator(repo)
    
    # Gera 1 hora de histórico (12 passos de 5 minutos)
    generator.generate_historical_dataset("MACHINE_HIST", hours=1, frequency_minutes=5)
    
    history = repo.get_historical_readings("MACHINE_HIST", limit=20)
    assert len(history) > 0
    assert history[0]["machine_id"] == "MACHINE_HIST"
