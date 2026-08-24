import time
from datetime import datetime
from typing import Dict, Any
from simulator.failures import MachineState, FailureMode
from simulator.sensors import SensorSimulator
from config.settings import settings

class VirtualMachine:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.state = MachineState.NORMAL
        self.failure_mode = FailureMode.NORMAL_OPERATION
        
        # Variáveis internas para simular a progressão do tempo e degradação
        self.cycle_count = 0
        self.degradation_factor = 0.0
        
    def set_condition(self, state: MachineState, failure_mode: FailureMode):
        """Muda o estado e o modo de falha da máquina manualmente (para testes/simulação)."""
        self.state = state
        self.failure_mode = failure_mode

    def _calculate_degradation(self):
        """Atualiza o fator de degradação com base no estado atual."""
        if self.state == MachineState.NORMAL:
            self.degradation_factor = 0.0
        elif self.state == MachineState.WARNING:
            self.degradation_factor += 0.05
        elif self.state == MachineState.DEGRADING:
            self.degradation_factor += 0.15
        elif self.state == MachineState.CRITICAL:
            self.degradation_factor += 0.5
        elif self.state == MachineState.FAILURE:
            self.degradation_factor += 2.0

    def generate_telemetry(self) -> Dict[str, Any]:
        """Gera um frame de dados dos sensores baseado no estado atual da máquina."""
        self.cycle_count += 1
        self._calculate_degradation()
        
        # Modificadores baseados no tipo de falha
        temp_mod = 1.0
        vib_mod = 1.0
        curr_mod = 1.0
        
        if self.failure_mode == FailureMode.OVERHEATING:
            temp_mod = 1.5 + self.degradation_factor
        elif self.failure_mode == FailureMode.BEARING_FAILURE:
            vib_mod = 2.0 + (self.degradation_factor * 1.2)
            temp_mod = 1.2 + (self.degradation_factor * 0.5)
        elif self.failure_mode == FailureMode.ELECTRICAL_FAULT:
            curr_mod = 1.8 + self.degradation_factor
        elif self.failure_mode == FailureMode.IMBALANCE:
            vib_mod = 1.5 + self.degradation_factor

        # Geração dos valores usando o simulador de sensores
        telemetry = {
            "timestamp": datetime.utcnow().isoformat(),
            "machine_id": self.machine_id,
            "state": self.state.value,
            "failure_mode": self.failure_mode.value,
            "temperature": SensorSimulator.generate_reading(
                settings.BASE_TEMP * temp_mod, noise_std=1.5
            ),
            "vibration": SensorSimulator.generate_reading(
                settings.BASE_VIBRATION * vib_mod, noise_std=0.2
            ),
            "current": SensorSimulator.generate_reading(
                settings.BASE_CURRENT * curr_mod, noise_std=0.5
            ),
            "rpm": SensorSimulator.generate_reading(
                settings.BASE_RPM, noise_std=15.0 # RPM costuma variar um pouco, mas cair drasticamente apenas em falha total
            ),
            "noise": SensorSimulator.generate_reading(
                settings.BASE_NOISE * (vib_mod * 0.8), noise_std=2.0
            )
        }
        
        return telemetry
