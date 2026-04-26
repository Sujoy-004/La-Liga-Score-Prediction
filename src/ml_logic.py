import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import lightgbm as lgb
import shap
from typing import List, Tuple

def clean_cols(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df

def get_historical_data():
    # Priority: Local data for stability, Fallback: GitHub for fresh data
    try:
        historical_data = pd.read_excel("data/matches_full.xlsx")
        fixtures_data = pd.read_excel("data/la-liga-2025-UTC.xlsx")
    except FileNotFoundError:
        historical_data_url = "https://github.com/Sujoy-004/La-Liga-Score-Prediction/raw/main/matches_full.xlsx"
        fixtures_data_url = "https://github.com/Sujoy-004/La-Liga-Score-Prediction/raw/main/la-liga-2025-UTC.xlsx"
        historical_data = pd.read_excel(historical_data_url)
        fixtures_data = pd.read_excel(fixtures_data_url)
    
    historical_data = clean_cols(historical_data)
    fixtures_data = clean_cols(fixtures_data)
    
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    fixtures_data['date'] = pd.to_datetime(fixtures_data['date'])
    
    team_mapping = {
        'Athletic Club': 'Athletic Club', 'Atletico Madrid': 'Atletico Madrid', 'AtlÃ©tico Madrid': 'Atletico Madrid',
        'Barcelona': 'Barcelona', 'FC Barcelona': 'Barcelona', 'Real Madrid': 'Real Madrid',
        'Villarreal': 'Villarreal', 'Villarreal CF': 'Villarreal', 'Real Betis': 'Real Betis',
        'Betis': 'Real Betis', 'Rayo Vallecano': 'Rayo Vallecano', 'Mallorca': 'Mallorca',
        'RCD Mallorca': 'Mallorca', 'Real Sociedad': 'Real Sociedad', 'Celta Vigo': 'Celta Vigo',
        'Celta': 'Celta Vigo', 'Osasuna': 'Osasuna', 'CA Osasuna': 'Osasuna', 'Sevilla': 'Sevilla',
        'Sevilla FC': 'Sevilla', 'Girona': 'Girona', 'Girona FC': 'Girona', 'Getafe': 'Getafe',
        'Getafe CF': 'Getafe', 'Espanyol': 'Espanyol', 'RCD Espanyol de Barcelona': 'Espanyol',
        'Leganes': 'Leganes', 'LeganÃ©s': 'Leganes', 'Las Palmas': 'Las Palmas', 'Valencia': 'Valencia',
        'Valencia CF': 'Valencia', 'Alaves': 'Alaves', 'AlavÃ©s': 'Alaves', 'Deportivo AlavÃ©s': 'Alaves',
        'Valladolid': 'Valladolid', 'Cadiz': 'Cadiz', 'CÃ¡diz': 'Cadiz', 'Almeria': 'Almeria',
        'AlmerÃ­a': 'Almeria', 'Granada': 'Granada', 'Elche': 'Elche', 'Elche CF': 'Elche',
        'Levante': 'Levante', 'Levante UD': 'Levante', 'Huesca': 'Huesca', 'Eibar': 'Eibar',
        'Real Oviedo': 'Real Oviedo'
    }
    
    historical_data['team'] = historical_data['team'].map(team_mapping).fillna(historical_data['team'])
    historical_data['opponent'] = historical_data['opponent'].map(team_mapping).fillna(historical_data['opponent'])
    fixtures_data['home_team'] = fixtures_data['home_team'].map(team_mapping).fillna(fixtures_data['home_team'])
    fixtures_data['away_team'] = fixtures_data['away_team'].map(team_mapping).fillna(fixtures_data['away_team'])
    
    return historical_data, fixtures_data

def create_simplified_features(historical_data):
    """Optimized feature generation using vectorized pandas operations"""
    df = historical_data.sort_values(['team', 'date']).copy()
    
    # Pre-calculate points and wins
    df['points'] = df['result'].map({'W': 3, 'D': 1, 'L': 0})
    df['is_win'] = (df['result'] == 'W').astype(int)
    
    # Grouped rolling features (past games only via .shift(1))
    grouped = df.groupby('team')
    df['rolling_gf_7'] = grouped['gf'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_ga_7'] = grouped['ga'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_wins_3'] = grouped['is_win'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    df['rolling_points_3'] = grouped['points'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    
    # Venue-Specific Form (Last 5 home/away)
    df['venue_gf_5'] = df.groupby(['team', 'venue'])['gf'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['venue_ga_5'] = df.groupby(['team', 'venue'])['ga'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # Defaults for early season
    df[['rolling_gf_7', 'rolling_ga_7', 'venue_gf_5', 'venue_ga_5']] = df[['rolling_gf_7', 'rolling_ga_7', 'venue_gf_5', 'venue_ga_5']].fillna(1.5)
    df[['rolling_wins_3', 'rolling_points_3']] = df[['rolling_wins_3', 'rolling_points_3']].fillna(0)
    
    # Pivot to match format
    home_df = df[df['venue'] == 'Home'].copy()
    away_df = df[df['venue'] == 'Away'].copy()
    
    features_df = home_df.merge(
        away_df[['date', 'team', 'rolling_gf_7', 'rolling_ga_7', 'rolling_wins_3', 'rolling_points_3', 'venue_gf_5', 'venue_ga_5']],
        left_on=['date', 'opponent'],
        right_on=['date', 'team'],
        suffixes=('_home', '_away')
    ).rename(columns={
        'rolling_gf_7_home': 'home_goals_avg', 'rolling_ga_7_home': 'home_conceded_avg',
        'rolling_gf_7_away': 'away_goals_avg', 'rolling_ga_7_away': 'away_conceded_avg',
        'rolling_wins_3_home': 'home_recent_wins', 'rolling_points_3_home': 'home_recent_points',
        'rolling_wins_3_away': 'away_recent_wins', 'rolling_points_3_away': 'away_recent_points',
        'venue_gf_5_home': 'home_venue_goals_avg', 'venue_ga_5_away': 'away_venue_conceded_avg'
    })

    # H2H Calculation
    h2h_home, h2h_away = [], []
    for idx, row in features_df.iterrows():
        past_h2h = features_df[(features_df['date'] < row['date']) & 
                               (((features_df['team_home'] == row['team_home']) & (features_df['team_away'] == row['team_away'])) |
                                ((features_df['team_home'] == row['team_away']) & (features_df['team_away'] == row['team_home'])))]
        
        home_wins = ((past_h2h['team_home'] == row['team_home']) & (past_h2h['result'] == 'W')).sum() + \
                    ((past_h2h['team_away'] == row['team_home']) & (past_h2h['result'] == 'L')).sum()
        away_wins = ((past_h2h['team_home'] == row['team_away']) & (past_h2h['result'] == 'W')).sum() + \
                    ((past_h2h['team_away'] == row['team_away']) & (past_h2h['result'] == 'L')).sum()
        
        h2h_home.append(home_wins)
        h2h_away.append(away_wins)
        
    features_df['h2h_home_wins'] = h2h_home
    features_df['h2h_away_wins'] = h2h_away
    
    target_mapping = {'W': 1, 'D': 0, 'L': 0}
    features_df['target'] = features_df['result'].map(target_mapping)
    return features_df

def get_stacked_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    lgbm = lgb.LGBMClassifier(random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft'
    )
    return CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)

def train_simplified_model(features_df):
    model_features = ['season', 'home_goals_avg', 'home_conceded_avg', 'away_goals_avg', 'away_conceded_avg', 'home_recent_wins', 'home_recent_points', 'away_recent_wins', 'away_recent_points', 'h2h_home_wins', 'h2h_away_wins', 'home_venue_goals_avg', 'away_venue_conceded_avg']
    train_mask = features_df['season'] < 2024
    X_train, y_train = features_df.loc[train_mask, model_features].fillna(0), features_df.loc[train_mask, 'target']
    
    model = get_stacked_model()
    model.fit(X_train, y_train)
    return model, model_features

def calculate_team_stats(team_name, historical_data, is_home=True):
    team_matches = historical_data[historical_data['team'] == team_name].sort_values('date', ascending=False)
    last_7 = team_matches.head(7)
    venue_matches = team_matches[team_matches['venue'] == ('Home' if is_home else 'Away')].head(5)
    
    if len(last_7) == 0:
        return {'goals_avg': 1.5, 'conceded_avg': 1.5, 'recent_wins': 0, 'recent_points': 0, 'venue_stat': 1.5}
    
    goals_avg = last_7['gf'].mean()
    conceded_avg = last_7['ga'].mean()
    recent_3 = last_7.head(3)
    recent_wins = (recent_3['result'] == 'W').sum()
    recent_points = (recent_3['result'] == 'W').sum() * 3 + (recent_3['result'] == 'D').sum()
    
    venue_stat = venue_matches['gf'].mean() if is_home else venue_matches['ga'].mean()
    if pd.isna(venue_stat): venue_stat = 1.5
    
    return {'goals_avg': goals_avg, 'conceded_avg': conceded_avg, 'recent_wins': recent_wins, 'recent_points': recent_points, 'venue_stat': venue_stat}

def prepare_match_features(home_team, away_team, historical_data, model_features):
    home_stats = calculate_team_stats(home_team, historical_data, is_home=True)
    away_stats = calculate_team_stats(away_team, historical_data, is_home=False)
    
    h2h_matches = historical_data[((historical_data['team'] == home_team) & (historical_data['opponent'] == away_team)) | ((historical_data['team'] == away_team) & (historical_data['opponent'] == home_team))]
    h2h_home_wins = ((h2h_matches['team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
    h2h_away_wins = ((h2h_matches['team'] == away_team) & (h2h_matches['result'] == 'W')).sum()
    
    features = {
        'season': 2025, 'home_goals_avg': home_stats['goals_avg'], 'home_conceded_avg': home_stats['conceded_avg'],
        'away_goals_avg': away_stats['goals_avg'], 'away_conceded_avg': away_stats['conceded_avg'],
        'home_recent_wins': home_stats['recent_wins'], 'home_recent_points': home_stats['recent_points'],
        'away_recent_wins': away_stats['recent_wins'], 'away_recent_points': away_stats['recent_points'],
        'h2h_home_wins': h2h_home_wins, 'h2h_away_wins': h2h_away_wins,
        'home_venue_goals_avg': home_stats['venue_stat'], 'away_venue_conceded_avg': away_stats['venue_stat']
    }
    
    return pd.DataFrame([features])[model_features].fillna(0)

def get_tactical_insights(home_team, away_team, historical_data) -> List[str]:
    home_stats = calculate_team_stats(home_team, historical_data, is_home=True)
    away_stats = calculate_team_stats(away_team, historical_data, is_home=False)
    
    h2h_matches = historical_data[((historical_data['team'] == home_team) & (historical_data['opponent'] == away_team)) | ((historical_data['team'] == away_team) & (historical_data['opponent'] == home_team))]
    h2h_home_wins = ((h2h_matches['team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
    h2h_away_wins = ((h2h_matches['team'] == away_team) & (h2h_matches['result'] == 'W')).sum()

    insights = []
    if home_stats['venue_stat'] > 2.0:
        insights.append(f"FORTRESS: {home_team} is a Home Fortress (Avg {home_stats['venue_stat']:.1f} goals at home).")
    if h2h_away_wins > h2h_home_wins + 2:
        insights.append(f"BOGEY: {home_team} has 'Bogey Team' energy against {away_team} (H2H: {h2h_home_wins}-{h2h_away_wins}).")
    
    if not insights:
        insights.append("Tactical stalemate expected in the midfield transition phase.")
    return insights

def predict_match(home_team, away_team, model, model_features, historical_data):
    feature_df = prepare_match_features(home_team, away_team, historical_data, model_features)
    probabilities = model.predict_proba(feature_df)[0]
    p_home = float(probabilities[1])
    p_not_home = float(probabilities[0])
    
    if p_home >= 0.55:
        prediction = "Home Win"
    else:
        home_stats = calculate_team_stats(home_team, historical_data, is_home=True)
        away_stats = calculate_team_stats(away_team, historical_data, is_home=False)
        goal_diff = home_stats['goals_avg'] - away_stats['goals_avg']
        conceded_diff = away_stats['conceded_avg'] - home_stats['conceded_avg']
        score = 0.6 * goal_diff + 0.4 * conceded_diff
        prediction = "Away Win" if score < -0.15 else "Draw"
        
    insights = get_tactical_insights(home_team, away_team, historical_data)
        
    # SHAP Attribution (Explainable AI)
    try:
        base_ensemble = getattr(model.calibrated_classifiers_[0], "estimator", getattr(model.calibrated_classifiers_[0], "base_estimator", None))
        rf_model = base_ensemble.estimators_[0] 
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(feature_df)
        
        if isinstance(shap_values, list): 
            attr = shap_values[1][0]
        elif len(shap_values.shape) == 3: 
            attr = shap_values[0, :, 1]
        else:
            attr = shap_values[0]
            
        attribution = {feat: float(val) for feat, val in zip(model_features, attr)}
    except Exception as e:
        attribution = {"error": str(e)}

    return prediction, (p_home, p_not_home), insights, attribution
