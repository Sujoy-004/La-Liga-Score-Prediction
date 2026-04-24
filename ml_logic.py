import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def clean_cols(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df

def get_historical_data():
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
    matches_df = historical_data[historical_data['season'] >= 2020].copy()
    features_list = []
    
    for season in matches_df['season'].unique():
        season_matches = matches_df[matches_df['season'] == season].sort_values('date')
        for idx, match in season_matches.iterrows():
            if match['venue'] != 'Home': continue
            home_team, away_team, match_date, result = match['team'], match['opponent'], match['date'], match['result']
            
            home_history = matches_df[(matches_df['team'] == home_team) & (matches_df['date'] < match_date)].sort_values('date').tail(10)
            away_history = matches_df[(matches_df['team'] == away_team) & (matches_df['date'] < match_date)].sort_values('date').tail(10)
            
            home_goals_avg = home_history['gf'].mean() if len(home_history) > 0 else 1.5
            home_conceded_avg = home_history['ga'].mean() if len(home_history) > 0 else 1.5
            away_goals_avg = away_history['gf'].mean() if len(away_history) > 0 else 1.5
            away_conceded_avg = away_history['ga'].mean() if len(away_history) > 0 else 1.5
            
            home_recent, away_recent = home_history.tail(3), away_history.tail(3)
            home_recent_wins = (home_recent['result'] == 'W').sum()
            home_recent_points = (home_recent['result'] == 'W').sum() * 3 + (home_recent['result'] == 'D').sum()
            away_recent_wins = (away_recent['result'] == 'W').sum()
            away_recent_points = (away_recent['result'] == 'W').sum() * 3 + (away_recent['result'] == 'D').sum()
            
            h2h_matches = matches_df[((matches_df['team'] == home_team) & (matches_df['opponent'] == away_team)) | ((matches_df['team'] == away_team) & (matches_df['opponent'] == home_team))]
            h2h_home_wins = ((h2h_matches['team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
            h2h_away_wins = ((h2h_matches['team'] == away_team) & (h2h_matches['result'] == 'W')).sum()
            
            features = {
                'date': match_date, 'season': season, 'home_team': home_team, 'away_team': away_team,
                'home_goals_avg': home_goals_avg, 'home_conceded_avg': home_conceded_avg,
                'away_goals_avg': away_goals_avg, 'away_conceded_avg': away_conceded_avg,
                'home_recent_wins': home_recent_wins, 'home_recent_points': home_recent_points,
                'away_recent_wins': away_recent_wins, 'away_recent_points': away_recent_points,
                'h2h_home_wins': h2h_home_wins, 'h2h_away_wins': h2h_away_wins, 'result': result
            }
            features_list.append(features)
            
    features_df = pd.DataFrame(features_list)
    target_mapping = {'W': 1, 'D': 0, 'L': 0}
    features_df['target'] = features_df['result'].map(target_mapping)
    return features_df

def train_simplified_model(features_df):
    model_features = ['season', 'home_goals_avg', 'home_conceded_avg', 'away_goals_avg', 'away_conceded_avg', 'home_recent_wins', 'home_recent_points', 'away_recent_wins', 'away_recent_points', 'h2h_home_wins', 'h2h_away_wins']
    train_mask = features_df['season'] < 2024
    X_train, y_train = features_df.loc[train_mask, model_features].fillna(0), features_df.loc[train_mask, 'target']
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, model_features

def calculate_team_stats(team_name, historical_data, is_home=True):
    team_matches = historical_data[historical_data['team'] == team_name].sort_values('date', ascending=False)
    last_10 = team_matches.head(10)
    if len(last_10) == 0:
        return {'goals_avg': 1.5, 'conceded_avg': 1.5, 'recent_wins': 0, 'recent_points': 0}
    
    goals_avg = last_10['gf'].mean()
    conceded_avg = last_10['ga'].mean()
    recent_3 = last_10.head(3)
    recent_wins = (recent_3['result'] == 'W').sum()
    recent_points = (recent_3['result'] == 'W').sum() * 3 + (recent_3['result'] == 'D').sum()
    
    return {'goals_avg': goals_avg, 'conceded_avg': conceded_avg, 'recent_wins': recent_wins, 'recent_points': recent_points}

def predict_match(home_team, away_team, model, model_features, historical_data):
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
        'h2h_home_wins': h2h_home_wins, 'h2h_away_wins': h2h_away_wins
    }
    feature_df = pd.DataFrame([features])[model_features].fillna(0)
    probabilities = model.predict_proba(feature_df)[0]
    p_home, p_not_home = probabilities[1], probabilities[0]
    
    if p_home >= 0.65:
        prediction = "Home Win"
    else:
        goal_diff = home_stats['goals_avg'] - away_stats['goals_avg']
        conceded_diff = away_stats['conceded_avg'] - home_stats['conceded_avg']
        score = 0.6 * goal_diff + 0.4 * conceded_diff
        prediction = "Away Win" if score > 0.2 else "Draw"
        
    return prediction, (p_home, p_not_home), None
