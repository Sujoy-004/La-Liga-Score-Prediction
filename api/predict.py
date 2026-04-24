from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Optional
import os
import sys

# Add parent directory to path to import ml_logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_logic

app = FastAPI(title="La Liga Match Predictor API")

# Global state for the model and data
state = {
    "model": None,
    "model_features": None,
    "historical_data": None
}

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    prediction: str
    p_home: float
    p_not_home: float

def initialize_pipeline():
    """Lazy load data and train model"""
    if state["model"] is None:
        try:
            print("🚀 Initializing ML Pipeline (Lazy Load)...")
            historical_data, _ = ml_logic.get_historical_data()
            features_df = ml_logic.create_simplified_features(historical_data)
            model, model_features = ml_logic.train_simplified_model(features_df)
            
            state["model"] = model
            state["model_features"] = model_features
            state["historical_data"] = historical_data
            print("✅ Pipeline initialized successfully.")
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise e

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict a match result"""
    # Initialize pipeline if needed
    initialize_pipeline()
    
    try:
        prediction, (p_home, p_not_home), error = ml_logic.predict_match(
            request.home_team,
            request.away_team,
            state["model"],
            state["model_features"],
            state["historical_data"]
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
            
        return {
            "prediction": prediction,
            "p_home": float(p_home),
            "p_not_home": float(p_not_home)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}
