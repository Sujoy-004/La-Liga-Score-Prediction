# ⚽ La Liga Deep Analytics
**Status:** Gold Master (v0.6.0) | **Calibrated Win Probability Engine**

Professional machine learning pipeline and interactive dashboard for La Liga match result prediction. This project utilizes a **Calibrated Stacked Ensemble** (Random Forest, XGBoost, LightGBM) to deliver high-precision insights and tactical comparisons.

## 🚀 Key Features
- **Calibrated Engine**: Sigmoid (Platt) Scaling achieving a **Brier Score of 0.2370**.
- **Deep Insights**: Automated detection of "Bogey Team" energy and "Home Fortress" tactical windows.
- **Full-Stack Architecture**: Decoupled FastAPI backend with a Dash/Plotly frontend (Migration to Next.js in Roadmap).
- **MLOps Automation**: Weekly data fetch (API-Football), automated retraining, and Hugging Face Hub synchronization.

## 📊 Performance
- **CV Accuracy**: 57.35% (Baseline: 52.8%)
- **Model Integrity**: 5-Fold Time-Series Cross-Validation.

## 🛠️ Tech Stack
- **Languages**: Python (3.11+)
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Dash, Plotly, Bootstrap
- **Ops**: GitHub Actions, Docker, SQLite

## 🗺️ Strategic Roadmap
The project is currently transitioning to its next architectural evolution:
1.  **UI Pivot**: Migration to Next.js + Tailwind CSS.
2.  **Explainability**: SHAP/LIME integration for feature attribution.
3.  **Real-time Pulse**: Live Match-Day WebSockets.
4.  **DDD Refactor**: Architectural pivot to Domain-Driven Design.
5.  **Observability**: Prometheus/Grafana monitoring stack.

---
*Developed by Sujoy | Guided by Antigravity Senior Architect Protocol*
