# ⚽ La Liga Deep Analytics Engine
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![v0.6.0](https://img.shields.io/badge/version-v0.6.0-gold.svg)](https://github.com/Sujoy-004/La-Liga-Score-Prediction)

A production-grade, full-stack machine learning system for La Liga match forecasting. This platform integrates automated data ingestion, calibrated ensemble modeling, and a professional interactive dashboard.

## 🏗️ Project Structure
```text
/src
  ├── api.py           # FastAPI Production Service
  ├── Dashboard.py     # Dash/Plotly Analytics UI
  ├── ml_logic.py      # Core Prediction Engine
  └── /pipeline
      ├── fetcher.py   # Automated API Data Ingestion
      └── retrain.py   # Automated Model Retraining
/data
  └── la_liga.db       # SQLite Analytics Engine
/models
  └── latest_model.pkl # Calibrated Ensemble Artifact
/notebooks
  └── predict_score_fr.ipynb
.github/workflows/      # Weekly CI/CD Pipeline
```

## 🚀 Key Features
- **Automated Pipeline**: Weekly Tuesday runs that fetch live results from API-Football and retrain the ensemble.
- **Deep Analytics UI**: Matte-dark dashboard with Radar Charts and Win Probability gauges.
- **Production API**: Scalable FastAPI backend with auto-generated Swagger docs at `/docs`.
- **Calibrated Ensemble**: RF + XGB + LGBM voting architecture optimized with Sigmoid calibration for 57% CV accuracy.

## 🛠️ Getting Started
1. Copy `sample.env` to `.env` and add your `FOOTBALL_API_KEY`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the API: `python -m src.api`.
4. Launch the Dashboard: `python -m src.Dashboard`.

---
*Maintained by Sujoy-004*
