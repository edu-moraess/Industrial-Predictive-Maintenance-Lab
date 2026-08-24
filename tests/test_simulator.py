import pytest
from simulator.machine import VirtualMachine
from simulator.failures import MachineState, FailureMode

def test_machine_initialization():
    machine = VirtualMachine("M001")
    assert machine.machine_id == "M001"
    assert machine.state == MachineState.NORMAL
    assert machine.failure_mode == FailureMode.NORMAL_OPERATION

def test_normal_telemetry():
    machine = VirtualMachine("M001")
    data = machine.generate_telemetry()
    
    assert "timestamp" in data
    assert data["machine_id"] == "M001"
    assert 40 < data["temperature"] < 50  # Temperatura base é 45
    assert 1.5 < data["vibration"] < 3.5  # Vibração base é 2.5

def test_bearing_failure_degradation():
    machine = VirtualMachine("M001")
    
    # Coleta baseline
    normal_data = machine.generate_telemetry()
    
    # Muda para falha e estado crítico
    machine.set_condition(MachineState.CRITICAL, FailureMode.BEARING_FAILURE)
    
    # Roda alguns ciclos para acumular degradação
    for _ in range(5):
        critical_data = machine.generate_telemetry()
        
    # Em falha de rolamento crítica, a vibração deve ser muito maior que o normal
    assert critical_data["vibration"] > normal_data["vibration"] * 1.5
