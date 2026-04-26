# CHANGELOG: La Liga Deep Analytics

## [0.6.0] - 2026-04-26
### Added
- **Nyquist Verification Gates**: Added `scratch/test_upsert.py` for database integrity testing.
- **Visual Assets**: Generated high-res dashboard screenshot for Bogey Team insights.
- **Unique Constraint**: Added `idx_date_team_unique` to `la_liga.db` to prevent context rot during weekly fetches.

### Fixed
- **API Key Inconsistency**: Standardized `fetcher.py` to use `FOOTBALL_API_KEY` environment variable, aligning with CI/CD secrets.
- **Upsert Logic**: Corrected `INSERT ... ON CONFLICT` behavior by ensuring underlying unique index existence.

### Security
- Verified `.github/workflows/sync_to_hf.yml` for correct secret injection and LFS support.

## [0.7.0] - 2026-04-26
### Added
- **Next.js Frontend**: Scaffolded Next.js 14+ (App Router) in `frontend/` directory.
- **Arctic Design System**: Implemented "Quiet Luxury" UI with glassmorphism, Geist typography, and Arctic Blue palette.
- **Client Components**: Developed `Sidebar`, `Header`, and `Predictor` interactive components.
- **State Management**: Integrated `@tanstack/react-query` for real-time inference hydration.
- **Environment Config**: Added `.env.local` for dynamic backend port mapping.

## [0.7.1] - 2026-04-26
### Added
- **Model Explainability**: Integrated `SHAP` in `ml_logic.py` to calculate feature attribution for calibrated predictions.
- **Tactical Breakdown UI**: Added a horizontal bar chart in the Predictor page using `Recharts` to visualize SHAP weights.
- **Backend Attribution**: Updated FastAPI `/predict` endpoint to return mathematical attribution payloads.

### Fixed
- **UI Hygiene**: Deduplicated teams list in Predictor to resolve React key warnings.
- **Hydration Sync**: Optimized chart rendering sequence to prevent dev-mode overlay notifications.

## [0.7.2] - 2026-04-26
### Added
- **Live Pulse Engine**: Implemented `WebSocket` server using FastAPI for real-time tactical streaming.
- **Match Pulse UI**: Created `/pulse` page with dynamic area charts and real-time event logs.
- **Probability Drift Simulation**: Integrated an in-play simulation engine to demonstrate live probability variance.

### Changed
- **API CORS**: Enabled `CORSMiddleware` in `src/api.py` to allow cross-origin requests from the Next.js frontend.
- **Backend Port**: Shifted default local API port to **8001** to prevent workspace service conflicts.

## [To-Be-Done] - Future Strategic Pillars
- [x] **UI Migration**: Transition from Dash to **Next.js + Tailwind CSS** for premium aesthetic sovereignty.
- [x] **Explainability**: Integrated **SHAP** to provide "Tactical Attribution" charts for model decision-making.
- [x] **Live Pulse**: Implemented **WebSockets** for real-time match-day probability updates.
- [x] **Architecture Refactor**: Pivoted to **Domain-Driven Design (DDD)** for enterprise-grade scalability.
- [x] **Observability**: Deployed a **Prometheus/Grafana** sentinel stack for tactical monitoring.

## [0.8.1] - 2026-04-26
### Added
- **Metric Instrumentation**: Integrated `prometheus-fastapi-instrumentator` for telemetry.
- **Sentinel Stack**: Added `monitoring/` directory with `prometheus.yml` and `grafana_dashboard.json`.
- **Health Verification**: Validated `/metrics` endpoint for real-time scraping.

## [0.8.0] - 2026-04-26
### Added
- **Architectural Sovereignty**: Fully refactored codebase into DDD layers (`domain`, `application`, `infrastructure`, `interfaces`).
- **Dependency Injection**: Implemented repository patterns for database and ML model management.
- **Enhanced Modularity**: Decoupled core inference logic from the web transport layer.
