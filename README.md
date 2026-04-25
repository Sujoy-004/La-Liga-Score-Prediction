# ⚽ La Liga Match Outcome Predictor

**Status:** ![Sovereignty](https://img.shields.io/badge/Protocol-Sovereign-orange) ![Version](https://img.shields.io/badge/Version-0.3.0-blue)
**Authority:** Rewired Senior Architect Protocol (Mythos-GSD-Graphify)

---

### 🏗️ Project Structure
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

## 🧠 ML Pipeline Logic (Phase 2)
- **Model**: Calibrated Stacked Ensemble (RF + XGB + LGBM).
- **Features**: 7-game rolling averages + Venue-specific form + H2H Dynamics.
- **Inference**: Dynamic probability thresholding (0.55 floor).

---

## 📈 Performance Tracking
| Metric | Value |
| :--- | :--- |
| **Cross-Val Accuracy** | 57.35% |
| **Brier Score** | 0.2370 |
| **Inference Time** | ~95ms |
| **Training Time** | ~6.2s |

---

**Built with 💡 by Antigravity | Senior Architect**
