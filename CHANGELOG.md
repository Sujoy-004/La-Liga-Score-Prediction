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

## [To-Be-Done] - Future Strategic Pillars
- [ ] **UI Migration**: Transition from Dash to **Next.js + Tailwind CSS** for premium aesthetic sovereignty.
- [ ] **Explainability**: Integrate **SHAP/LIME** to provide "Deep Insights" into model decision-making.
- [ ] **Live Pulse**: Implement **WebSockets** for real-time match-day probability updates.
- [ ] **Architecture Refactor**: Pivot to **Domain-Driven Design (DDD)** for enterprise-grade scalability.
- [ ] **Observability**: Deploy a **Prometheus/Grafana** stack for model drift and API latency monitoring.
