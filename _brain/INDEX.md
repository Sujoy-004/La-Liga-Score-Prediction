# La Liga Deep Analytics: Sovereign Intelligence Index (v0.8.1)

## 🏛️ Core Architecture: Domain-Driven Design (DDD)
The platform is structured into four distinct tactical layers to ensure enterprise-grade scalability and modularity.

- [[DDD_TRANSITION|DDD Migration Report]]
- [[SOVEREIGN_CONTEXT_V081|Tactical Summary v0.8.1]]
- [[ML_LOGIC_V081|Model Ensemble & Explainability]]

## 🛠️ Components
- **Domain Layer**: Pure business entities (`src/domain/models.py`)
- **Infrastructure Layer**: Data & Model adapters (`src/infrastructure/repositories.py`)
- **Application Layer**: Orchestration logic (`src/application/services.py`)
- **Interfaces Layer**: FastAPI Controllers & WebSockets (`src/interfaces/`)

## 🚀 Deployment Stats
- **Backend**: FastAPI (Port 8002)
- **Frontend**: Next.js 14+ (Port 3000)
- **Monitoring**: Prometheus (Port 8002/metrics)
- **Database**: SQLite (data/la_liga.db)

## 📅 Roadmap Execution
- [x] Phase 1-5: Gold Master Baseline
- [x] Phase 7: UI Migration (Next.js)
- [x] Phase 8: Model Explainability (SHAP)
- [x] Phase 9: Real-time Live Pulse (WebSockets)
- [x] Phase 10: Architectural Pivot (DDD)
- [x] Phase 11: Observability (Sentinel Stack)
