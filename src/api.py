from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import joblib
import os
from src.ml_logic import predict_match

app = FastAPI(
    title="La Liga Deep Analytics API",
    description="Professional ML prediction engine for La Liga fixtures using a Calibrated Stacked Ensemble.",
    version="0.6.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables to store data/model
MODEL_PATH = 'models/latest_model.pkl'
DB_PATH = 'data/la_liga.db'

def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    payload = joblib.load(MODEL_PATH)
    conn = sqlite3.connect(DB_PATH)
    historical_data = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    
    return payload['model'], payload['features'], historical_data

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Calibrated Stacked Ensemble"}

@app.get("/predict")
def get_prediction(home_team: str, away_team: str):
    model, features, historical_data = load_resources()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained. Run retrain.py first.")
    
    try:
        prediction, (p_home, p_not_home), insights = predict_match(home_team, away_team, model, features, historical_data)
        return {
            "fixture": f"{home_team} vs {away_team}",
            "prediction": prediction,
            "probability_home_win": f"{p_home*100:.1f}%",
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats/{team}")
def get_team_stats(team: str):
    conn = sqlite3.connect(DB_PATH)
    # Get last 7 games
    query = "SELECT * FROM matches WHERE team = ? ORDER BY date DESC LIMIT 7"
    df = pd.read_sql(query, conn, params=(team,))
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Team not found in database.")
    
    return {
        "team": team,
        "rolling_gf": float(df['gf'].mean()),
        "rolling_ga": float(df['ga'].mean()),
        "recent_results": df['result'].tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
