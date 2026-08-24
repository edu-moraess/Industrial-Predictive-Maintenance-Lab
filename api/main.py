"""
Industrial Predictive Maintenance Lab — FastAPI backend (V2).

All inference goes through ml.inference_engine.InferenceEngine.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.schemas import MachineResponse, PredictResponse, SensorInput
from database.repository import DatabaseRepository
from features.engineering import FeatureEngineer
from ml.inference_engine import get_engine, get_ready_engine

repo = DatabaseRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ready_engine()
    yield


app = FastAPI(
    title="Industrial Predictive Maintenance Lab API",
    version="2.0.0",
    description=(
        "Experimental API for synthetic industrial telemetry and unified ML inference."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health_status() -> Dict[str, Any]:
    eng = get_engine()
    st = eng.status()
    return {
        "status": "online",
        "service": "Industrial Predictive Maintenance Engine",
        "version": "2.0.0",
        "models_ready": st.ready,
        "ml": st.to_dict(),
    }


@app.get("/ml/status")
def ml_status() -> Dict[str, Any]:
    return get_engine().status().to_dict()


@app.get("/machines", response_model=List[MachineResponse])
def get_machines():
    return repo.get_machines()


@app.get("/machines/{machine_id}")
def get_machine_details(machine_id: str):
    machines = repo.get_machines()
    machine = next((m for m in machines if m["machine_id"] == machine_id), None)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return machine


@app.get("/machines/{machine_id}/latest")
def get_latest_telemetry(machine_id: str):
    reading = repo.get_latest_reading(machine_id)
    if not reading:
        raise HTTPException(status_code=404, detail="No readings found for this machine.")
    return reading


@app.post("/predict", response_model=PredictResponse)
def predict_machine_health(telemetry: SensorInput):
    eng = get_ready_engine()
    history = repo.get_historical_readings(telemetry.machine_id, limit=20)
    current_dict = telemetry.model_dump()
    # SensorInput has no failure_mode; preserve history labels only
    raw_list = history + [current_dict]
    result = eng.predict(raw_list, machine_id=telemetry.machine_id)

    return PredictResponse(
        machine_id=result.machine_id,
        health_score=result.health_score,
        risk_level=result.risk_level,
        anomaly=result.is_anomaly,
        anomaly_score=result.anomaly_score,
        failure_type=result.failure_mode,
        failure_probabilities=result.failure_probabilities,
        rul_hours=result.rul_hours,
    )


@app.get("/anomalies")
def get_anomalies(machine_id: str, limit: int = 20):
    eng = get_ready_engine()
    history = repo.get_historical_readings(machine_id, limit=limit)
    if not history:
        return []

    df_features = FeatureEngineer.process_telemetry(history)
    df_anomalies = eng.anomaly_detector.detect(df_features)
    anomalies_only = df_anomalies[df_anomalies["is_anomaly"] == True]
    return anomalies_only.to_dict(orient="records")
