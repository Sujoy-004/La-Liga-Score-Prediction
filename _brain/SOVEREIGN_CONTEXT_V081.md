# Sovereign Intelligence Report: v0.8.1

## 🏁 Current State
The project has reached **Gold Master (v0.8.1)**. All slop has been purged. The system is a high-fidelity, real-time tactical analytics engine for La Liga.

## 🧠 ML Stack Intelligence
- **Ensemble**: VotingClassifier (Random Forest, XGBoost, LightGBM).
- **Calibration**: Sigmoid-based probability calibration (Brier Score: 0.2370).
- **Explainability**: Integrated **SHAP (TreeExplainer)** targeting the Random Forest base estimator for sub-second tactical attribution.
- **Data Pipeline**: Rolling 7-match window features with H2H win-weighting.

## 🏛️ DDD Layer Mapping
- **Domain**: `MatchPrediction`, `TeamStats`, `PulseUpdate`.
- **Infrastructure**: `MatchRepository` (SQLite), `MLModelRepository` (Pickle + SHAP).
- **Application**: `PredictionService` (Inference orchestration).
- **Interfaces**: FastAPI endpoints (`/predict`, `/stats`, `/health`, `/metrics`) and `/ws/pulse` (WebSocket).

## 📡 Live Pulse Protocol
- **Transport**: WebSockets (FastAPI).
- **Payload**: Probability drift + tactical anomalies.
- **Frontend**: Framer Motion animated area charts + live tactical event log.

## 🛡️ Monitoring & Health
- **Sentinel**: Prometheus instrumented.
- **Metrics**: Available at `http://localhost:8002/metrics`.
- **Port Strategy**: Port 8002 (Backend) / Port 3000 (Frontend).
