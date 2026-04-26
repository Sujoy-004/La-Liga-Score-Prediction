from src.domain.models import MatchPrediction, TeamStats, PulseUpdate
from src.infrastructure.repositories import MatchRepository, MLModelRepository
from src.ml_logic import prepare_match_features, get_tactical_insights
import pandas as pd
from typing import List, Tuple

class PredictionService:
    def __init__(self, match_repo: MatchRepository, model_repo: MLModelRepository):
        self.match_repo = match_repo
        self.model_repo = model_repo

    def predict_fixture(self, home_team: str, away_team: str) -> MatchPrediction:
        # 1. Load Resources
        model, features = self.model_repo.load()
        historical_data = self.match_repo.get_historical_data()
        
        # 2. Feature Engineering (Delegated to specialized logic)
        feature_df = prepare_match_features(home_team, away_team, historical_data, features)
        
        # 3. Inference
        probs = model.predict_proba(feature_df)[0]
        p_home = float(probs[1])
        prediction = "Home Win" if p_home > 0.5 else "Away Win"
        
        # 4. Insights & Attribution
        insights = get_tactical_insights(home_team, away_team, historical_data)
        attribution = self.model_repo.get_shap_attribution(feature_df)
        
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            prediction=prediction,
            home_win_prob=p_home,
            insights=insights,
            attribution=attribution
        )

    def get_team_stats(self, team: str) -> TeamStats:
        df = self.match_repo.get_team_recent_matches(team)
        if df.empty:
            raise ValueError(f"Team {team} not found")
            
        return TeamStats(
            team_name=team,
            rolling_gf=float(df['gf'].mean()),
            rolling_ga=float(df['ga'].mean()),
            recent_results=df['result'].tolist()
        )
