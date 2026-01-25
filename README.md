# Power Theft Detection in Smart Grids (Heuristic Demo, ML-Ready Architecture)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project is a **credible engineering demo** for power-theft monitoring in smart grids.

By default, the running dashboard/API operates in **simulated demo mode** and computes **heuristic risk scores** to demonstrate monitoring workflows in **near real-time (simulated)**.

The repository also contains ML/feature-engineering modules intended as a foundation for a future “real inference” pipeline, but **real-time ML inference is not wired into the running API by default**.

## Scope
- Heuristic anomaly risk scoring + simulated monitoring in the dashboard
- ML-ready architecture (models + preprocessing modules are present, but not connected to live inference)
- Research / prototype-level system

## Non-Goals
- Hardware / meter firmware integration
- Utility-scale deployment guarantees
- Full production IAM / authorization

### 🎯 Objectives

- **Demonstrate an intrusion-detection workflow** for smart-meter consumption monitoring
- **Handle data-related challenges** effectively (missing values, class imbalance)
- **Provide a clear upgrade path** to real ML inference
- **Provide near real-time (simulated) monitoring** and automated alert generation
- **Support energy security** and grid sustainability

### ⚡ Key Features

- **Near real-time (simulated) Monitoring**: Monitoring workflow demonstrated via dashboard + API
- **Heuristic Risk Scoring (Demo Mode)**: Produces risk-like outputs for UI/API demonstration
- **Dataset Support (Optional)**: Can load a CSV dataset if provided; otherwise simulates data
- **Interactive Dashboard**: Web-based monitoring interface
- **Automated Alerts**: Risk-based alert system (HIGH/MEDIUM/LOW)
- **CI + Tests**: Ruff + pytest run on every push via GitHub Actions

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Meter Data                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Intrusion Detection System (IDS)                    │
│  • Heuristic risk scoring (demo mode)                       │
│  • Risk Classification (HIGH/MEDIUM/LOW)                    │
│  • Anomaly Detection                                        │
│  • Alert Generation & Management                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Web Dashboard & Monitoring                     │
│  • Near real-time (simulated) Metrics                       │
│  • Alert Management                                         │
│  • Visualization & Reports                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚨 Intrusion Detection System

### Detection Process
1. **Data Collection**: Receive consumption reading
2. **Preprocessing**: Extract features
3. **Risk Scoring (Demo Mode)**: Compute a heuristic risk score
4. **Risk Classification**: Classify as HIGH/MEDIUM/LOW/NORMAL
5. **Alert Generation**: Create alerts for theft cases
6. **Logging**: Record all detections

## Upgrade Path to Real ML (Optional)

To convert this demo into a real ML inference system, you would typically add:
- A training script that saves a model artifact (e.g., `joblib` or Keras model)
- A saved scaler/normalizer and a stable feature list
- A deterministic preprocessing pipeline shared by training and inference
- API endpoints that load artifacts at startup and run real predictions on validated inputs

### Risk Levels
- **HIGH**: Probability ≥ 80% - Immediate investigation required
- **MEDIUM**: Probability ≥ 50% - Investigation recommended
- **LOW**: Probability ≥ 30% - Monitor closely
- **NORMAL**: Probability < 30% - No action needed

## 📱 Web Dashboard Features

- **Near real-time (simulated) Monitoring**: Live system status and metrics (simulated feed)
- **Detection Statistics**: Total detections and risk distribution (demo)
- **Alert Management**: View and acknowledge alerts
- **Risk Distribution**: Visualization of risk levels
- **Simulation Mode**: Test the system with random data
- **Export Functionality**: Download alerts and detection logs

## API (Demo)

All API routes are under `/api/*`.

If `API_KEY` is set, include:

- Header: `X-API-KEY: <your key>`

### `GET /api/available-years`
Response:

```json
{ "years": [2015, 2016, 2017] }
```

### `GET /api/year-detections/<year>`
Returns demo detection rows with a heuristic `risk_score`.

Response (array):

```json
[
  {
    "meter_id": "MTR-2024-00001",
    "risk_score": 0.83,
    "risk_level": "HIGH",
    "avg_consumption": 12.3,
    "expected_consumption": 22.0,
    "status": "Flagged",
    "detection_date": "2024-05-12",
    "year": 2024
  }
]
```

## 📁 Project Structure

```
power-theft-detection-system/
├── src/
│   ├── data_preprocessing.py    # Data preprocessing and feature engineering
│   ├── models.py                # Deep learning and ML models
│   ├── intrusion_detection.py  # IDS implementation
│   ├── risk_scoring.py          # Heuristic risk score helper used by the demo runtime
│   └── visualization.py         # Plotting and visualization
├── templates/
│   ├── index_with_timeline.html # Web dashboard template
│   └── login.html              # Login page
├── data/
│   ├── raw/                     # (Optional) place larger datasets here
│   └── sample_data.csv          # Small illustrative sample
├── results/
│   ├── MODEL_PERFORMANCE_SUMMARY.md  # Offline/experimental notes
│   └── training_results.json         # Offline/experimental notes
├── tests/
│   ├── test_api.py
│   └── test_risk_scoring.py
├── app.py                      # Flask web application
├── .github/workflows/ci.yml     # CI (ruff + pytest)
├── Dockerfile
├── requirements.txt            # Runtime/demo dependencies
├── requirements-ml.txt         # Optional ML/visualization dependencies
└── README.md                   # This file
```

## Experimental / Offline Modules

The `src/` directory contains exploratory and prototype modules (feature engineering, model architectures, visualization helpers).
These are **not used by the current demo runtime** (`app.py`) by default.
They exist to document an ML-ready architecture and provide a starting point for future work.

## Dependencies

- `requirements.txt` is the lightweight set needed to run the demo web app.
- `requirements-ml.txt` contains optional heavy dependencies (TensorFlow, scikit-learn, plotting) for offline experiments.

## 🔧 Configuration

Configuration is done via environment variables (see Quick Start).

## 📊 Results & Visualization

The running demo focuses on the dashboard experience and simulated monitoring.

The `results/` artifacts and the visualization modules under `src/` are intended for offline experimentation.

## 🎓 Research References

## Deployment (Docker)

```bash
docker build -t power-theft .
docker run -p 5000:5000 power-theft
```

## Security Note

This is a prototype. If `API_KEY` is set in the environment, `/api/*` routes require:

- Header: `X-API-KEY: <your key>`

If `API_KEY` is not set, API routes remain accessible after login.

## Testing

```bash
pytest -q
```

## CI

GitHub Actions runs on every push:

- `ruff check .`
- `pytest -q`

Based on literature survey including:

1. **Smart grids based on deep learning** (Noor Mahmoud Ibrahim et al., 2021)
2. **Electricity Theft Detection in Smart Grid Systems: A CNN-LSTM Based Approach** (Md. Nazmul Hasan et al., 2019)
3. **An Ensemble Deep Convolutional Neural Network Model for Electricity Theft Detection in Smart Grids** (Hossein Mohammadi Rouzbahani et al., 2021)
4. **An Intelligent Framework for Electricity Theft Detection in Smart Grid** (Yogesh Kulkarni et al., 2021)

