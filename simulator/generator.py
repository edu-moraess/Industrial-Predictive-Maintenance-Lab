from datetime import datetime, timedelta, timezone
from simulator.machine import VirtualMachine
from simulator.failures import MachineState, FailureMode
from database.repository import DatabaseRepository

class DataGenerator:
    """Generates historical datasets and full simulations."""
    
    def __init__(self, repository: DatabaseRepository):
        self.repo = repository
        
    def generate_historical_dataset(self, machine_id: str, hours: int = 24, frequency_minutes: int = 5):
        """Generate a historical dataset simulating progressive degradation."""
        machine = VirtualMachine(machine_id)
        self.repo.upsert_machine(machine_id, name=f"Industrial Machine {machine_id}")
        
        total_steps = int((hours * 60) / frequency_minutes)
        current_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        for step in range(total_steps):
            current_time += timedelta(minutes=frequency_minutes)
            
            progress_ratio = step / total_steps
            if progress_ratio > 0.8:
                machine.set_condition(MachineState.CRITICAL, FailureMode.BEARING_FAILURE)
            elif progress_ratio > 0.6:
                machine.set_condition(MachineState.DEGRADING, FailureMode.BEARING_FAILURE)
            elif progress_ratio > 0.4:
                machine.set_condition(MachineState.WARNING, FailureMode.BEARING_FAILURE)
            else:
                machine.set_condition(MachineState.NORMAL, FailureMode.NORMAL_OPERATION)
            
            reading = machine.generate_telemetry()
            reading["timestamp"] = current_time.isoformat()
            
            self.repo.save_sensor_reading(reading)
            
        print(f"[{machine_id}] {total_steps} historical records generated and saved to SQLite.")
