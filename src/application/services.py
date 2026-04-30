from src.domain.models import MatchPrediction, TeamStats, PulseUpdate
from src.infrastructure.repositories import MatchRepository, MLModelRepository
from src.ml_logic import prepare_match_features, get_tactical_insights
import pandas as pd
from typing import List, Tuple, Dict
import difflib

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
            
        gf_mean = float(df['gf'].mean())
        ga_mean = float(df['ga'].mean())
        
        # Calculate Tactical Stability (based on GF variance)
        # Higher variance = Lower stability
        gf_std = df['gf'].std() if len(df) > 1 else 0
        stability = "High" if gf_std < 0.8 else "Moderate" if gf_std < 1.5 else "Volatile"
        
        # Calculate Trend (Last 3 vs Overall Average)
        recent_3 = df.head(3)
        recent_gf = recent_3['gf'].mean() if not recent_3.empty else gf_mean
        trend_val = ((recent_gf - gf_mean) / gf_mean * 100) if gf_mean > 0 else 0
        trend_str = f"{'+' if trend_val >= 0 else ''}{trend_val:.1f}% vs avg"

        return TeamStats(
            team_name=team,
            rolling_gf=gf_mean,
            rolling_ga=ga_mean,
            recent_results=df['result'].tolist(),
            tactical_stability=stability,
            stability_trend=trend_str
        )

    def search_entities(self, query: str) -> List[Dict[str, str]]:
        query = query.strip()
        found_teams = self.match_repo.search_teams(query)
        results = []
        
        for team in found_teams:
            results.append({"type": "team", "name": team, "url": f"/stats/{team}"})
        
        # Robust fixture matching logic (vs, v, -, x)
        separators = [" vs ", " v ", " - ", " x "]
        q_lower = query.lower()
        
        for sep in separators:
            if sep in q_lower:
                parts = q_lower.split(sep)
                if len(parts) == 2:
                    h_search, a_search = parts[0].strip().lower(), parts[1].strip().lower()
                    
                    # Validate against actual teams using fuzzy matching (difflib)
                    all_teams = self.match_repo.get_all_teams()
                    
                    h_matches = difflib.get_close_matches(h_search, [t.lower() for t in all_teams], n=1, cutoff=0.6)
                    a_matches = difflib.get_close_matches(a_search, [t.lower() for t in all_teams], n=1, cutoff=0.6)
                    
                    if h_matches and a_matches:
                        # Find the original casing
                        h_match = next(t for t in all_teams if t.lower() == h_matches[0])
                        a_match = next(t for t in all_teams if t.lower() == a_matches[0])
                        
                        results.insert(0, {
                            "type": "fixture", 
                            "name": f"{h_match} vs {a_match}", 
                            "url": f"/predict?home={h_match}&away={a_match}"
                        })
                        break 
        
        return results
