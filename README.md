# Industrial Predictive Maintenance Lab

> **DISCLAIMER**  
> This project is strictly experimental and educational. All sensor data is **synthetic**.  
> Machine Learning models (Isolation Forest, Random Forest, Health Score, RUL) must **not** be used for real industrial diagnosis, safety decisions, or production prognosis without independent validation on calibrated physical data.

---

## Overview

**Industrial Predictive Maintenance Lab** is a modular virtual laboratory for Computer Engineering.  
It simulates industrial assets, generates sensor telemetry, engineers features, detects anomalies, classifies failure modes, estimates health and remaining useful life (RUL), and exposes results via FastAPI and a Streamlit control center.

### Conceptual pipeline

```
Virtual Machine
      ↓
Telemetry (Temperature, Vibration, Current, RPM, Noise)
      ↓
Feature Engineering
      ↓
Anomaly Detection (Isolation Forest)
      ↓
Failure Classification (Random Forest)
      ↓
Health Score (0–100) + Risk Level
      ↓
RUL Estimator (experimental)
      ↓
SQLite  ·  FastAPI  ·  Streamlit Dashboard
```

MQTT client/publisher modules are preserved for a future path to ESP32 / real sensors.

---

## Architecture layers

| Layer | Role |
|--------|------|
| **Simulator** | Official telemetry source (`simulator/`) — VirtualMachine, sensors, failure modes |
| **Feature Engineering** | Rolling stats, rate of change, trends (`features/`) |
| **ML** | Anomaly, classification, health, RUL (`ml/`) |
| **Database** | SQLite persistence (`database/`) |
| **API** | FastAPI REST (`api/`) |
| **UI** | Streamlit industrial dashboard (`app/`) — primary frontend |
| **MQTT** | Optional bridge for external / ESP32 telemetry (`mqtt/`) |

**Streamlit is the primary UI for this version.**  
A partial React prototype exists under `frontend/` but is **not** part of the main runtime path (no package.json / incomplete scaffold). Future direction: React → FastAPI → ML.

---

## Stack

- Python 3.10+
- NumPy, Pandas, scikit-learn
- FastAPI, Uvicorn, Pydantic
- Streamlit, Plotly
- SQLite
- paho-mqtt
- Pytest

---

## Quick start

### Prerequisites

- Python 3.10+
- pip, git

### Local install

```bash
git clone https://github.com/edu-moraess/Industrial-Predictive-Maintenance-Lab.git
cd Industrial-Predictive-Maintenance-Lab

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Streamlit dashboard

From the **repository root**:

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501

Sidebar: Machine ID, state, failure injection, refresh interval, history window.  
Controls: **START** / **PAUSE** / **RESET**.

### FastAPI

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs  
- Health: `GET /health`  
- Inference: `POST /predict` with sensor payload  

Models are bootstrapped on application **lifespan** (not at import time).

### Tests

```bash
pytest
```

`pytest.ini` sets `pythonpath = .` so package imports resolve from the repo root.

### Docker

```bash
docker compose up --build
```

- Dashboard: port **8501**  
- API: port **8000**

---

## Health Score thresholds

| Score | Risk level |
|-------|------------|
| 90–100 | HEALTHY |
| 70–89 | NORMAL |
| 50–69 | WARNING |
| 25–49 | CRITICAL |
| 0–24 | FAILURE RISK |

---

## RUL

**Estimated Remaining Useful Life** is an experimental mapping from health score and degradation trends on **synthetic** data.  
It is not a calibrated industrial prognosis. UI and API always treat it as experimental.

---

## Telemetry (simulated)

| Sensor | Nominal baseline | Unit (reference) |
|--------|------------------|------------------|
| Temperature | 45.0 | °C |
| Vibration | 2.5 | mm/s |
| Current | 15.0 | A |
| RPM | 1800 | RPM |
| Noise | 65.0 | dB |

Values are synthetic; no claim of physical sensor calibration.

---

## Design system (Streamlit)

- Background `#0F1115` · Surface `#171A21` · Border `#2A2F38`
- Text primary `#F2F2F2` · secondary `#9A9FA8`
- Accent `#D4A84F` (restrained) · Success / Warning / Critical status colors
- Typography: Arial / Helvetica / system-ui
- CSS tokens: `app/styles.py`
- Plotly theme: `app/theme.py` → `apply_industrial_plotly_theme(fig)`

Industrial control-room aesthetic: dark, minimal, technical — no neon, glassmorphism, or cyberpunk chrome.

---

## Project layout

```
Industrial-Predictive-Maintenance-Lab/
├── app/                 # Streamlit UI + design system
│   ├── dashboard.py
│   ├── styles.py
│   └── theme.py
├── api/                 # FastAPI
├── config/              # Settings / baselines
├── database/            # SQLite models + repository
├── features/            # Feature engineering
├── ml/                  # Anomaly, classifier, health, RUL
├── mqtt/                # MQTT listener + ESP32-style publisher sim
├── simulator/           # Official virtual machine + sensors
├── tests/
├── frontend/            # Incomplete React prototype (out of main path)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Known limitations / future work

1. **Dashboard scores** — Streamlit uses the official `VirtualMachine` for telemetry; health/anomaly/RUL in the UI remain **provisional** for responsiveness. Full ML stack is served by FastAPI `/predict`. Next iteration: consume API or shared service layer from the dashboard.
2. **Anomaly score** — Isolation Forest decision scores are min–max normalized per batch; scores can shift with batch composition. Future: fixed reference / rolling calibration.
3. **Classifier metrics** — Training accuracy on the same synthetic set is not production performance; do not present it as such.
4. **CORS** — `allow_origins` includes `*` for development; tighten via environment for deployment.
5. **React frontend** — Scaffold only; not wired as primary UI.
6. **MQTT** — Preserved for real-sensor path; not required for the lab loop.

---

## License / academic use

Intended for teaching and experimentation in Computer Engineering and related fields.  
Do not deploy as a safety-critical or production predictive-maintenance system without rigorous validation on real plant data.
