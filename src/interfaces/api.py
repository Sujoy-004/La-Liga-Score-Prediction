from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from src.application.services import PredictionService
from src.infrastructure.repositories import MatchRepository, MLModelRepository
from src.interfaces.ws_handler import handle_pulse_stream
from prometheus_fastapi_instrumentator import Instrumentator
import os

app = FastAPI(title="La Liga DDD API", version="0.8.0")

# Instrument for Prometheus
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DI Initialization
DB_PATH = 'data/la_liga.db'
MODEL_PATH = 'models/latest_model.pkl'

match_repo = MatchRepository(DB_PATH)
model_repo = MLModelRepository(MODEL_PATH)
prediction_service = PredictionService(match_repo, model_repo)

@app.get("/health")
def health():
    return {"status": "sovereign", "arch": "DDD"}

@app.get("/predict")
def predict(home_team: str, away_team: str):
    try:
        result = prediction_service.predict_fixture(home_team, away_team)
        return {
            "fixture": f"{result.home_team} vs {result.away_team}",
            "prediction": result.prediction,
            "probability_home_win": f"{result.home_win_prob*100:.1f}%",
            "insights": result.insights,
            "attribution": result.attribution
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats/{team}")
def stats(team: str):
    try:
        s = prediction_service.get_team_stats(team)
        return {
            "team": s.team_name,
            "rolling_gf": s.rolling_gf,
            "rolling_ga": s.rolling_ga,
            "recent_results": s.recent_results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.websocket("/ws/pulse")
async def websocket_pulse(websocket: WebSocket):
    await handle_pulse_stream(websocket)
