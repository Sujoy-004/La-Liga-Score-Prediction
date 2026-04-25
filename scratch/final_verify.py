import sqlite3
import pandas as pd
import joblib
from src.ml_logic import predict_match

def run_final_verification():
    print("=== FINAL SYSTEM VERIFICATION ===")
    
    # 1. Load Resources
    payload = joblib.load('models/latest_model.pkl')
    model = payload['model']
    features = payload['features']
    
    conn = sqlite3.connect('data/la_liga.db')
    historical_data = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    
    # 2. Run High-Profile Prediction
    home, away = "Real Madrid", "Barcelona"
    prediction, (p_home, p_not_home), insights = predict_match(home, away, model, features, historical_data)
    
    print(f"\nMatch: {home} vs {away}")
    print(f"Prediction: {prediction.upper()}")
    print(f"Win Probability ({home}): {p_home*100:.1f}%")
    print("\nMatch Insights:")
    for ins in insights:
        print(f" - {ins}")

if __name__ == "__main__":
    run_final_verification()
