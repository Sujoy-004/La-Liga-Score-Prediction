from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass(frozen=True)
class TeamStats:
    team_name: str
    rolling_gf: float
    rolling_ga: float
    recent_results: List[str]
    tactical_stability: str = "High"
    stability_trend: str = "Stable"

@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    prediction: str
    home_win_prob: float
    insights: List[str]
    attribution: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class PulseUpdate:
    match: str
    home_win_prob: float
    away_win_prob: float
    event: str
    timestamp: float
