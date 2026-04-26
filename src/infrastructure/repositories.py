import sqlite3
import pandas as pd
import joblib
import os
import shap
from typing import Tuple, Dict, Any, Optional, List

class MatchRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_historical_data(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM matches", conn)
        conn.close()
        return df

    def get_team_recent_matches(self, team: str, limit: int = 7) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM matches WHERE team = ? ORDER BY date DESC LIMIT ?"
        df = pd.read_sql(query, conn, params=(team, limit))
        conn.close()
        return df

class MLModelRepository:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._features = None

    def load(self) -> Tuple[Any, List[str]]:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        payload = joblib.load(self.model_path)
        self._model = payload['model']
        self._features = payload['features']
        return self._model, self._features

    def get_shap_attribution(self, feature_df: pd.DataFrame) -> Dict[str, float]:
        if self._model is None:
            self.load()
            
        try:
            # Access the Random Forest component (index 0) of the VotingClassifier ensemble
            base_ensemble = getattr(self._model.calibrated_classifiers_[0], "estimator", 
                                  getattr(self._model.calibrated_classifiers_[0], "base_estimator", None))
            rf_model = base_ensemble.estimators_[0]
            
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(feature_df)
            
            # Extract importance for the "Home Win" class (index 1)
            if isinstance(shap_values, list):
                attr = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                attr = shap_values[0, :, 1]
            else:
                attr = shap_values[0]
                
            return {feat: float(val) for feat, val in zip(self._features, attr)}
        except Exception as e:
            return {"error": str(e)}
