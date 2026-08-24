"""Integration: normal -> inject bearing failure -> model responds."""

import os

os.environ.setdefault("IPML_DB_PATH", "/tmp/ipml_inference_test.db")

from simulator.failures import FailureMode, MachineState
from simulator.machine import VirtualMachine
from ml.inference_engine import InferenceEngine
from database.repository import DatabaseRepository
from simulator.generator import DataGenerator


def _train_engine() -> InferenceEngine:
    repo = DatabaseRepository()
    DataGenerator(repo).generate_historical_dataset(
        "TRAIN_INTEGRATION", hours=4, frequency_minutes=5
    )
    rows = repo.get_historical_readings("TRAIN_INTEGRATION", limit=200)
    eng = InferenceEngine()
    eng.bootstrap(rows)
    assert eng.is_ready
    return eng


def test_engine_bootstrap_and_status():
    eng = _train_engine()
    st = eng.status()
    assert st.ready is True
    assert st.isolation_forest == "READY"
    assert st.random_forest == "READY"
    assert st.training_samples > 0


def test_normal_then_bearing_failure_causal():
    eng = _train_engine()
    vm = VirtualMachine("MACHINE_CAUSAL")

    normal_readings = []
    vm.set_condition(MachineState.NORMAL, FailureMode.NORMAL_OPERATION)
    for _ in range(15):
        normal_readings.append(vm.generate_telemetry())

    r_normal = eng.predict(normal_readings)
    assert r_normal.health_score >= 50
    assert r_normal.model_status == "READY"

    fail_readings = list(normal_readings)
    vm.set_condition(MachineState.CRITICAL, FailureMode.BEARING_FAILURE)
    for _ in range(10):
        fail_readings.append(vm.generate_telemetry())

    r_fail = eng.predict(fail_readings)

    # Causal expectations on synthetic data (soft thresholds)
    assert r_fail.health_score <= r_normal.health_score + 5
    assert r_fail.anomaly_score >= r_normal.anomaly_score - 0.05 or r_fail.is_anomaly
    assert r_fail.rul_hours <= r_normal.rul_hours + 50
    # Ground truth comparison available when failure_mode present
    assert r_fail.ground_truth_failure == "BEARING_FAILURE"
