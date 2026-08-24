from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class SensorInput(BaseModel):
    machine_id: str = Field(..., example="MACHINE_001")
    temperature: float = Field(..., example=48.5)
    vibration: float = Field(..., example=3.2)
    current: float = Field(..., example=16.1)
    rpm: float = Field(..., example=1790.0)
    noise: float = Field(..., example=67.5)

class PredictResponse(BaseModel):
    machine_id: str
    health_score: float
    risk_level: str
    anomaly: bool
    anomaly_score: float
    failure_type: str
    failure_probabilities: Dict[str, float]
    rul_hours: float

class MachineResponse(BaseModel):
    machine_id: str
    name: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
