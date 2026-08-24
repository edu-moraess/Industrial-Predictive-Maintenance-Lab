import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint():
    payload = {
        "machine_id": "MACHINE_API_TEST",
        "temperature": 85.0,
        "vibration": 8.5,
        "current": 28.0,
        "rpm": 1200.0,
        "noise": 95.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["machine_id"] == "MACHINE_API_TEST"
    assert "health_score" in data
    assert "risk_level" in data
    assert "anomaly" in data
    assert "failure_type" in data
    assert "rul_hours" in data

def test_get_machines():
    response = client.get("/machines")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
