# ⚽ La Liga Deep Analytics
**Status:** Gold Master (v0.8.1) | **Calibrated Win Probability Engine**

Professional machine learning pipeline and interactive dashboard for La Liga match result prediction. This project utilizes a **Calibrated Stacked Ensemble** (Random Forest, XGBoost, LightGBM) to deliver high-precision insights and tactical comparisons.

## 🚀 Quick Start (v0.8.1 Gold Master)

### 1. Tactical Engine (Backend)
The backend has been refactored into **Domain-Driven Design (DDD)** for enterprise scalability.
```bash
# From root
python -m uvicorn src.interfaces.api:app --host 0.0.0.0 --port 8002
```
*   **API Docs**: `http://localhost:8002/docs`
*   **Health**: `http://localhost:8002/health`
*   **Metrics**: `http://localhost:8002/metrics` (Prometheus)

### 2. Arctic HUD (Frontend)
Premium Next.js 14+ dashboard with Framer Motion animations.
```bash
cd frontend
npm run dev
```
*   **Predictor**: `http://localhost:3000/predict`
*   **Match Pulse**: `http://localhost:3000/pulse` (Real-time WebSockets)

## 🏛️ Architecture: The DDD Pivot
The platform follows a clean, layered architecture:
- **Domain**: Pure business logic and tactical entities (`src/domain/`).
- **Infrastructure**: Robust repositories for SQLite and ML inference (`src/infrastructure/`).
- **Application**: Orchestration services for tactical insights (`src/application/`).
- **Interfaces**: REST and WebSocket handlers (`src/interfaces/`).

## 🧠 ML Engine Features
- **Calibrated Engine**: Sigmoid (Platt) Scaling achieving a **Brier Score of 0.2370**.
- **Tactical Explainability**: Integrated **SHAP** attribution for every prediction.
- **Deep Insights**: Automated detection of "Bogey Team" energy and "Home Fortress" tactical windows.
- **Live Pulse**: Bidirectional WebSocket stream for real-time tactical anomalies.

## 🛡️ Monitoring & Observability
- **Sentinel Stack**: Integrated Prometheus exporter.
- **Dashboard**: Ready-to-use Grafana template in `monitoring/grafana_dashboard.json`.
