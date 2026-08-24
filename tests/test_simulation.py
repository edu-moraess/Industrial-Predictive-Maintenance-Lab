import pytest
from simulator.machine import VirtualMachine
from simulator.failures import MachineState, FailureMode
from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.anomaly_detection import AnomalyDetector

def test_realtime_stream_simulation():
    repo = DatabaseRepository()
    machine = VirtualMachine("STREAM_TEST_MACHINE")
    detector = AnomalyDetector()

    # Simula 5 ciclos de streaming continuo
    for _ in range(5):
        telemetry = machine.generate_telemetry()
        repo.save_sensor_reading(telemetry)

    readings = repo.get_historical_readings("STREAM_TEST_MACHINE", limit=5)
    assert len(readings) == 5

    df_features = FeatureEngineer.process_telemetry(readings)
    detector.train(df_features)
    df_result = detector.detect(df_features)

    assert "is_anomaly" in df_result.columns
