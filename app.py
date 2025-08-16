import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="La Liga Score Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin: 1rem 0;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the data"""
    with st.spinner("Loading historical data..."):
        # Load the historical data with error handling
        historical_data_url = "https://github.com/Sujoy-004/La-Liga-Score-Prediction/raw/main/matches_full.xlsx"
        try:
            historical_data = pd.read_excel(historical_data_url)
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            # Fallback option if needed
            raise e
        
        # Load the future fixtures data with error handling
        fixtures_data_url = "https://github.com/Sujoy-004/La-Liga-Score-Prediction/raw/main/la-liga-2025-UTC.xlsx"
        try:
            fixtures_data = pd.read_excel(fixtures_data_url)
        except Exception as e:
            st.error(f"Error loading fixtures data: {str(e)}")
            # Fallback option if needed
            raise e
    
    # Clean column names
    def clean_cols(df):
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
        return df
    
    historical_data = clean_cols(historical_data)
    fixtures_data = clean_cols(fixtures_data)
    
    # Convert date columns
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    fixtures_data['date'] = pd.to_datetime(fixtures_data['date'])
    
    # Team mapping for consistency
    team_mapping = {
        'Athletic Club': 'Athletic Club',
        'Atletico Madrid': 'Atletico Madrid',
        'AtlÃ©tico Madrid': 'Atletico Madrid',
        'Barcelona': 'Barcelona',
        'FC Barcelona': 'Barcelona',
        'Real Madrid': 'Real Madrid',
        'Villarreal': 'Villarreal',
        'Villarreal CF': 'Villarreal',
        'Real Betis': 'Real Betis',
        'Betis': 'Real Betis',
        'Rayo Vallecano': 'Rayo Vallecano',
        'Mallorca': 'Mallorca',
        'RCD Mallorca': 'Mallorca',
        'Real Sociedad': 'Real Sociedad',
        'Celta Vigo': 'Celta Vigo',
        'Celta': 'Celta Vigo',
        'Osasuna': 'Osasuna',
        'CA Osasuna': 'Osasuna',
        'Sevilla': 'Sevilla',
        'Sevilla FC': 'Sevilla',
        'Girona': 'Girona',
        'Girona FC': 'Girona',
        'Getafe': 'Getafe',
        'Getafe CF': 'Getafe',
        'Espanyol': 'Espanyol',
        'RCD Espanyol de Barcelona': 'Espanyol',
        'Leganes': 'Leganes',
        'LeganÃ©s': 'Leganes',
        'Las Palmas': 'Las Palmas',
        'Valencia': 'Valencia',
        'Valencia CF': 'Valencia',
        'Alaves': 'Alaves',
        'AlavÃ©s': 'Alaves',
        'Deportivo AlavÃ©s': 'Alaves',
        'Valladolid': 'Valladolid',
        'Cadiz': 'Cadiz',
        'CÃ¡diz': 'Cadiz',
        'Almeria': 'Almeria',
        'AlmerÃ­a': 'Almeria',
        'Granada': 'Granada',
        'Elche': 'Elche',
        'Elche CF': 'Elche',
        'Levante': 'Levante',
        'Levante UD': 'Levante',
        'Huesca': 'Huesca',
        'Eibar': 'Eibar',
        'Real Oviedo': 'Real Oviedo'
    }
    
    # Apply team mapping
    historical_data['team'] = historical_data['team'].map(team_mapping).fillna(historical_data['team'])
    historical_data['opponent'] = historical_data['opponent'].map(team_mapping).fillna(historical_data['opponent'])
    fixtures_data['home_team'] = fixtures_data['home_team'].map(team_mapping).fillna(fixtures_data['home_team'])
    fixtures_data['away_team'] = fixtures_data['away_team'].map(team_mapping).fillna(fixtures_data['away_team'])
    
    return historical_data, fixtures_data, team_mapping

@st.cache_data
def engineer_features(historical_data):
    """Engineer features for the model"""
    with st.spinner("Engineering features..."):
        # Filter historical_data to include matches from the 2020 season onwards
        matches_df = historical_data[historical_data['season'] >= 2020].copy()
        
        # Calculate goals scored and conceded per match
        matches_df['home_goals_per_match'] = matches_df.groupby(['season', 'team'])['gf'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        matches_df['home_conceded_per_match'] = matches_df.groupby(['season', 'team'])['ga'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
        # Away matches stats
        away_matches = matches_df[matches_df['venue'] == 'Away'].copy()
        away_matches['away_goals_per_match'] = away_matches.groupby(['season', 'team'])['gf'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        away_matches['away_conceded_per_match'] = away_matches.groupby(['season', 'team'])['ga'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        
        # Merge away stats
        matches_df = matches_df.merge(
            away_matches[['season', 'date', 'team', 'away_goals_per_match', 'away_conceded_per_match']],
            on=['season', 'date', 'team'],
            how='left'
        )
        
        # Handle home matches - get away team stats
        home_matches = matches_df[matches_df['venue'] == 'Home'].copy()
        home_matches = home_matches.merge(
            away_matches[['season', 'date', 'team', 'away_goals_per_match', 'away_conceded_per_match']],
            left_on=['season', 'date', 'opponent'],
            right_on=['season', 'date', 'team'],
            how='left',
            suffixes=('', '_away_opponent')
        )
        
        matches_df.loc[matches_df['venue'] == 'Home', 'away_goals_per_match'] = home_matches['away_goals_per_match_away_opponent'].values
        matches_df.loc[matches_df['venue'] == 'Home', 'away_conceded_per_match'] = home_matches['away_conceded_per_match_away_opponent'].values
        
        matches_df['away_goals_per_match'] = matches_df['away_goals_per_match'].fillna(0)
        matches_df['away_conceded_per_match'] = matches_df['away_conceded_per_match'].fillna(0)
        
        # Calculate win rates
        def calculate_win_rate(df, venue_type):
            df['wins'] = (df['result'] == 'W').astype(int)
            df['draws'] = (df['result'] == 'D').astype(int)
            df['losses'] = (df['result'] == 'L').astype(int)
            df[f'{venue_type}_matches_played'] = df.groupby(['season', 'team']).cumcount()
            df[f'{venue_type}_wins_cumulative'] = df.groupby(['season', 'team'])['wins'].cumsum()
            df[f'{venue_type}_win_rate'] = (df[f'{venue_type}_wins_cumulative'].shift(1).fillna(0) / df[f'{venue_type}_matches_played'].shift(1).fillna(0)).fillna(0)
            df = df.drop(columns=[f'{venue_type}_wins_cumulative', f'{venue_type}_matches_played', 'wins', 'draws', 'losses'])
            return df
        
        # Home and away win rates
        home_matches_df = matches_df[matches_df['venue'] == 'Home'].copy()
        home_matches_df = calculate_win_rate(home_matches_df, 'home')
        
        away_matches_df = matches_df[matches_df['venue'] == 'Away'].copy()
        away_matches_df = calculate_win_rate(away_matches_df, 'away')
        
        matches_df = matches_df.merge(home_matches_df[['season', 'date', 'team', 'home_win_rate']], on=['season', 'date', 'team'], how='left')
        matches_df = matches_df.merge(away_matches_df[['season', 'date', 'team', 'away_win_rate']], on=['season', 'date', 'team'], how='left')
        
        # Get opponent win rates
        matches_df.loc[matches_df['venue'] == 'Home', 'away_win_rate'] = matches_df.loc[matches_df['venue'] == 'Home'].merge(
            away_matches_df[['season', 'date', 'team', 'away_win_rate']],
            left_on=['season', 'date', 'opponent'],
            right_on=['season', 'date', 'team'],
            how='left'
        )['away_win_rate_y'].values
        
        matches_df.loc[matches_df['venue'] == 'Away', 'home_win_rate'] = matches_df.loc[matches_df['venue'] == 'Away'].merge(
            home_matches_df[['season', 'date', 'team', 'home_win_rate']],
            left_on=['season', 'date', 'opponent'],
            right_on=['season', 'date', 'team'],
            how='left'
        )['home_win_rate_y'].values
        
        matches_df['home_win_rate'] = matches_df['home_win_rate'].fillna(0)
        matches_df['away_win_rate'] = matches_df['away_win_rate'].fillna(0)
        
        # Recent form calculation
        def calculate_recent_form(df, team_col, venue_type):
            df['points'] = df['result'].apply(lambda x: 3 if x == 'W' else (1 if x == 'D' else 0))
            df['is_win'] = (df['result'] == 'W').astype(int)
            df['is_draw'] = (df['result'] == 'D').astype(int)
            df['is_loss'] = (df['result'] == 'L').astype(int)
            
            df[f'{venue_type}_form_wins'] = df.groupby(['season', team_col])['is_win'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            df[f'{venue_type}_form_draws'] = df.groupby(['season', team_col])['is_draw'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            df[f'{venue_type}_form_losses'] = df.groupby(['season', team_col])['is_loss'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            df[f'{venue_type}_form_points'] = df.groupby(['season', team_col])['points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            df[f'{venue_type}_form_gf'] = df.groupby(['season', team_col])['gf'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            df[f'{venue_type}_form_ga'] = df.groupby(['season', team_col])['ga'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum().fillna(0))
            
            df = df.drop(columns=['is_win', 'is_draw', 'is_loss', 'points'])
            return df
        
        # Calculate recent form
        home_matches_form = matches_df[matches_df['venue'] == 'Home'].copy()
        home_matches_form = calculate_recent_form(home_matches_form, 'team', 'home')
        
        away_matches_form = matches_df[matches_df['venue'] == 'Away'].copy()
        away_matches_form = calculate_recent_form(away_matches_form, 'team', 'away')
        
        matches_df = matches_df.merge(home_matches_form[['season', 'date', 'team', 'home_form_wins', 'home_form_draws', 'home_form_losses', 'home_form_points', 'home_form_gf', 'home_form_ga']], on=['season', 'date', 'team'], how='left')
        matches_df = matches_df.merge(away_matches_form[['season', 'date', 'team', 'away_form_wins', 'away_form_draws', 'away_form_losses', 'away_form_points', 'away_form_gf', 'away_form_ga']], on=['season', 'date', 'team'], how='left')
        
        # Head-to-head records (simplified version for performance)
        matches_df['h2h_home_wins'] = 0
        matches_df['h2h_draws'] = 0  
        matches_df['h2h_away_wins'] = 0
        
        # Filter to home matches only
        matches_df = matches_df[matches_df['venue'] == 'Home'].copy()
        matches_df.fillna(0, inplace=True)
        
        # Create features dataframe
        features_df = matches_df[['date', 'team', 'opponent', 'result', 'season',
                                  'home_goals_per_match', 'home_conceded_per_match',
                                  'away_goals_per_match', 'away_conceded_per_match',
                                  'home_win_rate', 'away_win_rate',
                                  'home_form_wins', 'home_form_draws', 'home_form_losses',
                                  'home_form_points', 'home_form_gf', 'home_form_ga',
                                  'away_form_wins', 'away_form_draws', 'away_form_losses',
                                  'away_form_points', 'away_form_gf', 'away_form_ga',
                                  'h2h_home_wins', 'h2h_draws', 'h2h_away_wins']].copy()
        
        features_df = features_df.rename(columns={'team': 'home_team', 'opponent': 'away_team'})
        
        # Target encoding
        target_mapping = {'W': 2, 'D': 1, 'L': 0}
        features_df['target_encoded'] = features_df['result'].map(target_mapping)
        
        return features_df, matches_df

@st.cache_resource
def train_models(features_df):
    """Train the machine learning models"""
    with st.spinner("Training models..."):
        # Data splitting
        train_mask = features_df['season'] < 2024
        val_mask = features_df['season'] == 2024
        
        model_features = ['season', 'home_goals_per_match', 'home_conceded_per_match',
                         'away_goals_per_match', 'away_conceded_per_match',
                         'home_win_rate', 'away_win_rate',
                         'home_form_wins', 'home_form_draws', 'home_form_losses',
                         'home_form_points', 'home_form_gf', 'home_form_ga',
                         'away_form_wins', 'away_form_draws', 'away_form_losses',
                         'away_form_points', 'away_form_gf', 'away_form_ga',
                         'h2h_home_wins', 'h2h_draws', 'h2h_away_wins']
        
        X_train = features_df.loc[train_mask, model_features]
        y_train = features_df.loc[train_mask, 'target_encoded']
        X_val = features_df.loc[val_mask, model_features]
        y_val = features_df.loc[val_mask, 'target_encoded']
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42),
            'LightGBM': lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1)
        }
        
        val_accuracies = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred_val)
            val_accuracies[name] = accuracy
        
        # Select best model
        best_model_name = max(val_accuracies, key=val_accuracies.get)
        best_model = models[best_model_name]
        
        return models, val_accuracies, best_model, best_model_name, model_features

def predict_fixture_features(home_team, away_team, fixture_date, matches_df, model_features):
    """Calculate features for a specific fixture"""
    fixture_season = fixture_date.year
    
    # Filter historical data
    historical_subset = matches_df[(matches_df['date'] < fixture_date) & (matches_df['season'] <= fixture_season)].copy()
    
    # Calculate home team stats
    home_team_historical_home = historical_subset[(historical_subset['home_team'] == home_team) & (historical_subset['venue'] == 'Home')].copy()
    home_goals_per_match = home_team_historical_home['gf'].mean() if not home_team_historical_home.empty else 0
    home_conceded_per_match = home_team_historical_home['ga'].mean() if not home_team_historical_home.empty else 0
    
    # Calculate away team stats  
    away_team_historical_away = historical_subset[(historical_subset['away_team'] == away_team) & (historical_subset['venue'] == 'Away')].copy()
    away_goals_per_match = away_team_historical_away['gf'].mean() if not away_team_historical_away.empty else 0
    away_conceded_per_match = away_team_historical_away['ga'].mean() if not away_team_historical_away.empty else 0
    
    # Calculate win rates
    home_team_historical = historical_subset[(historical_subset['home_team'] == home_team) | (historical_subset['away_team'] == home_team)].copy()
    home_wins = home_team_historical[
        ((home_team_historical['home_team'] == home_team) & (home_team_historical['result'] == 'W')) |
        ((home_team_historical['away_team'] == home_team) & (home_team_historical['result'] == 'L'))
    ].shape[0]
    home_matches_played = home_team_historical.shape[0]
    home_win_rate = home_wins / home_matches_played if home_matches_played > 0 else 0
    
    away_team_historical = historical_subset[(historical_subset['home_team'] == away_team) | (historical_subset['away_team'] == away_team)].copy()
    away_wins = away_team_historical[
        ((away_team_historical['home_team'] == away_team) & (away_team_historical['result'] == 'W')) |
        ((away_team_historical['away_team'] == away_team) & (away_team_historical['result'] == 'L'))
    ].shape[0]
    away_matches_played = away_team_historical.shape[0]
    away_win_rate = away_wins / away_matches_played if away_matches_played > 0 else 0
    
    # Recent form (simplified)
    home_team_recent = home_team_historical.tail(5)
    home_form_wins = ((home_team_recent['home_team'] == home_team) & (home_team_recent['result'] == 'W')).sum() + \
                     ((home_team_recent['away_team'] == home_team) & (home_team_recent['result'] == 'L')).sum()
    home_form_draws = (home_team_recent['result'] == 'D').sum()
    home_form_losses = ((home_team_recent['home_team'] == home_team) & (home_team_recent['result'] == 'L')).sum() + \
                       ((home_team_recent['away_team'] == home_team) & (home_team_recent['result'] == 'W')).sum()
    home_form_points = home_form_wins * 3 + home_form_draws
    
    away_team_recent = away_team_historical.tail(5)
    away_form_wins = ((away_team_recent['home_team'] == away_team) & (away_team_recent['result'] == 'W')).sum() + \
                     ((away_team_recent['away_team'] == away_team) & (away_team_recent['result'] == 'L')).sum()
    away_form_draws = (away_team_recent['result'] == 'D').sum()
    away_form_losses = ((away_team_recent['home_team'] == away_team) & (away_team_recent['result'] == 'L')).sum() + \
                       ((away_team_recent['away_team'] == away_team) & (away_team_recent['result'] == 'W')).sum()
    away_form_points = away_form_wins * 3 + away_form_draws
    
    # Head to head (simplified)
    h2h_matches = historical_subset[
        ((historical_subset['home_team'] == home_team) & (historical_subset['away_team'] == away_team)) |
        ((historical_subset['home_team'] == away_team) & (historical_subset['away_team'] == home_team))
    ]
    
    h2h_home_wins = ((h2h_matches['home_team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
    h2h_draws = (h2h_matches['result'] == 'D').sum()
    h2h_away_wins = ((h2h_matches['away_team'] == home_team) & (h2h_matches['result'] == 'L')).sum()
    
    # Create feature dictionary
    features = {
        'season': fixture_season,
        'home_goals_per_match': home_goals_per_match,
        'home_conceded_per_match': home_conceded_per_match,
        'away_goals_per_match': away_goals_per_match,
        'away_conceded_per_match': away_conceded_per_match,
        'home_win_rate': home_win_rate,
        'away_win_rate': away_win_rate,
        'home_form_wins': home_form_wins,
        'home_form_draws': home_form_draws,
        'home_form_losses': home_form_losses,
        'home_form_points': home_form_points,
        'home_form_gf': 0,  # Simplified
        'home_form_ga': 0,  # Simplified
        'away_form_wins': away_form_wins,
        'away_form_draws': away_form_draws,
        'away_form_losses': away_form_losses,
        'away_form_points': away_form_points,
        'away_form_gf': 0,  # Simplified
        'away_form_ga': 0,  # Simplified
        'h2h_home_wins': h2h_home_wins,
        'h2h_draws': h2h_draws,
        'h2h_away_wins': h2h_away_wins
    }
    
    # Create DataFrame and ensure all model features are present
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=model_features, fill_value=0)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return feature_df

# Main Streamlit App
def main():
    # Header
    st.markdown('<div class="main-header">⚽ La Liga Score Prediction</div>', unsafe_allow_html=True)
    
    # Initialize session state for expensive computations
    if 'data_loaded' not in st.session_state:
        try:
            with st.spinner("🔄 Loading and preprocessing data..."):
                historical_data, fixtures_data, team_mapping = load_and_preprocess_data()
                st.session_state.historical_data = historical_data
                st.session_state.fixtures_data = fixtures_data
                st.session_state.team_mapping = team_mapping
                st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            st.stop()
    else:
        historical_data = st.session_state.historical_data
        fixtures_data = st.session_state.fixtures_data
        team_mapping = st.session_state.team_mapping
    
    # Initialize session state for feature engineering
    if 'features_ready' not in st.session_state:
        try:
            with st.spinner("🔧 Engineering features..."):
                features_df, matches_df = engineer_features(historical_data)
                st.session_state.features_df = features_df
                st.session_state.matches_df = matches_df
                st.session_state.features_ready = True
        except Exception as e:
            st.error(f"Failed to engineer features: {str(e)}")
            st.stop()
    else:
        features_df = st.session_state.features_df
        matches_df = st.session_state.matches_df
    
    # Initialize session state for model training
    if 'models_trained' not in st.session_state:
        try:
            with st.spinner("🤖 Training machine learning models..."):
                models, val_accuracies, best_model, best_model_name, model_features = train_models(features_df)
                st.session_state.models = models
                st.session_state.val_accuracies = val_accuracies
                st.session_state.best_model = best_model
                st.session_state.best_model_name = best_model_name
                st.session_state.model_features = model_features
                st.session_state.models_trained = True
        except Exception as e:
            st.error(f"Failed to train models: {str(e)}")
            st.stop()
    else:
        models = st.session_state.models
        val_accuracies = st.session_state.val_accuracies
        best_model = st.session_state.best_model
        best_model_name = st.session_state.best_model_name
        model_features = st.session_state.model_features
        
        # Sidebar
        st.sidebar.markdown("## 📊 App Status")
        if st.session_state.get('data_loaded', False):
            st.sidebar.success("✅ Data loaded")
        if st.session_state.get('features_ready', False):
            st.sidebar.success("✅ Features ready")
        if st.session_state.get('models_trained', False):
            st.sidebar.success("✅ Models trained")
        
        st.sidebar.markdown("## 🏆 Model Performance")
        for name, accuracy in val_accuracies.items():
            color = "🟢" if name == best_model_name else "🔵"
            st.sidebar.metric(f"{color} {name}", f"{accuracy:.3f}")
        
        st.sidebar.markdown(f"**🥇 Best Model:** {best_model_name}")
        
        # Add app info
        st.sidebar.markdown("## ℹ️ About")
        st.sidebar.info(f"""
        **Teams**: {len(team_options)}
        **Historical Matches**: {len(features_df):,}
        **Upcoming Fixtures**: {len(fixtures_data):,}
        **Features Used**: {len(model_features)}
        """)
        
        # Team options
        team_options = sorted(list(set(fixtures_data['home_team'].unique()) | set(fixtures_data['away_team'].unique())))
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["🔮 Predict Match", "📅 Season Fixtures", "📊 Model Insights"])
        
        with tab1:
            st.markdown('<div class="sub-header">Predict Individual Match</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                home_team = st.selectbox("🏠 Home Team", team_options, key="home")
                
            with col2:
                away_team = st.selectbox("✈️ Away Team", team_options, key="away")
            
            match_date = st.date_input("📅 Match Date", pd.Timestamp.now().date())
            
            if st.button("🔮 Predict Match Result", type="primary"):
                if home_team != away_team:
                    try:
                        with st.spinner("🔮 Making prediction..."):
                            # Convert date to pandas timestamp
                            match_datetime = pd.Timestamp(match_date)
                            
                            # Calculate features
                            fixture_features = predict_fixture_features(
                                home_team, away_team, match_datetime, matches_df, model_features
                            )
                            
                            # Make prediction
                            prediction_encoded = best_model.predict(fixture_features)[0]
                            prediction_proba = best_model.predict_proba(fixture_features)[0]
                            
                            # Convert prediction
                            result_mapping = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
                            predicted_result = result_mapping[prediction_encoded]
                        
                        # Display prediction (outside spinner)
                        st.success("✅ Prediction completed successfully!")
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🎯 Match Prediction</h3>
                            <h2 style="color: #FF6B35;">{home_team} vs {away_team}</h2>
                            <h1 style="color: #2E86AB;">{predicted_result}</h1>
                            <p><strong>Confidence:</strong> {max(prediction_proba):.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown
                        st.markdown("### 📈 Probability Breakdown")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🏠 Home Win", f"{prediction_proba[2]:.1%}")
                        with col2:
                            st.metric("🤝 Draw", f"{prediction_proba[1]:.1%}")
                        with col3:
                            st.metric("✈️ Away Win", f"{prediction_proba[0]:.1%}")
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                        st.info("💡 Please try with different teams or check back later. Make sure both teams have sufficient historical data.")
                else:
                    st.warning("Please select different teams for home and away.")
        
        with tab2:
            st.markdown('<div class="sub-header">Season Fixtures Predictions</div>', unsafe_allow_html=True)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                selected_teams = st.multiselect(
                    "🏆 Filter by Teams (leave empty for all)", 
                    team_options,
                    default=[]
                )
            with col2:
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(fixtures_data['date'].min().date(), fixtures_data['date'].max().date()),
                    min_value=fixtures_data['date'].min().date(),
                    max_value=fixtures_data['date'].max().date()
                )
            
            if st.button("🔮 Predict All Fixtures", type="primary"):
                try:
                    with st.spinner("🔄 Filtering fixtures..."):
                        # Filter fixtures
                        filtered_fixtures = fixtures_data.copy()
                        
                        if selected_teams:
                            filtered_fixtures = filtered_fixtures[
                                (filtered_fixtures['home_team'].isin(selected_teams)) | 
                                (filtered_fixtures['away_team'].isin(selected_teams))
                            ]
                        
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                            filtered_fixtures = filtered_fixtures[
                                (filtered_fixtures['date'].dt.date >= start_date) & 
                                (filtered_fixtures['date'].dt.date <= end_date)
                            ]
                        
                        if len(filtered_fixtures) == 0:
                            st.warning("⚠️ No fixtures found with the selected filters. Please adjust your selection.")
                            st.stop()
                    
                    # Make predictions for filtered fixtures
                    st.info(f"🎯 Predicting {len(filtered_fixtures)} fixtures...")
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_fixtures = len(filtered_fixtures)
                    
                    for idx, (_, fixture) in enumerate(filtered_fixtures.iterrows()):
                        try:
                            status_text.text(f'Processing fixture {idx + 1}/{total_fixtures}: {fixture["home_team"]} vs {fixture["away_team"]}')
                            
                            fixture_features = predict_fixture_features(
                                fixture['home_team'], 
                                fixture['away_team'], 
                                fixture['date'], 
                                matches_df, 
                                model_features
                            )
                            
                            pred_encoded = best_model.predict(fixture_features)[0]
                            pred_proba = best_model.predict_proba(fixture_features)[0]
                            
                            result_mapping = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
                            predictions.append(result_mapping[pred_encoded])
                            probabilities.append(max(pred_proba))
                            
                        except Exception as e:
                            st.warning(f"⚠️ Could not predict {fixture['home_team']} vs {fixture['away_team']}: {str(e)}")
                            predictions.append('Error')
                            probabilities.append(0.0)
                        
                        progress_bar.progress((idx + 1) / total_fixtures)
                    
                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Add predictions to dataframe
                    filtered_fixtures['predicted_result'] = predictions
                    filtered_fixtures['confidence'] = probabilities
                        
                        # Display results
                        st.success("✅ All predictions completed!")
                        st.markdown("### 🎯 Fixture Predictions")
                        
                        display_df = filtered_fixtures[['date', 'home_team', 'away_team', 'predicted_result', 'confidence']].copy()
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if x > 0 else "N/A")
                        display_df.columns = ['Date', 'Home Team', 'Away Team', 'Prediction', 'Confidence']
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        st.markdown("### 📊 Prediction Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_fixtures = len(filtered_fixtures)
                            st.metric("Total Fixtures", total_fixtures)
                        
                        with col2:
                            home_wins = (filtered_fixtures['predicted_result'] == 'Home Win').sum()
                            st.metric("Predicted Home Wins", home_wins)
                        
                        with col3:
                            draws = (filtered_fixtures['predicted_result'] == 'Draw').sum()
                            st.metric("Predicted Draws", draws)
                        
                        with col4:
                            away_wins = (filtered_fixtures['predicted_result'] == 'Away Win').sum()
                            st.metric("Predicted Away Wins", away_wins)
                        
                    except Exception as e:
                        st.error(f"❌ Error predicting fixtures: {str(e)}")
                        st.info("💡 Please try again or contact support if the issue persists.")
                
                except Exception as e:
                    st.error(f"❌ Error processing fixtures: {str(e)}")
                    st.info("💡 Please check your selections and try again.")
        
        with tab3:
            st.markdown('<div class="sub-header">Model Insights & Performance</div>', unsafe_allow_html=True)
            
            # Model comparison
            st.markdown("### 🏆 Model Performance Comparison")
            
            model_df = pd.DataFrame({
                'Model': list(val_accuracies.keys()),
                'Validation Accuracy': list(val_accuracies.values())
            }).sort_values('Validation Accuracy', ascending=False)
            
            st.bar_chart(model_df.set_index('Model'))
            
            # Feature importance (for tree-based models)
            if hasattr(best_model, 'feature_importances_'):
                st.markdown("### 🎯 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('Feature'))
            
            # Dataset information
            st.markdown("### 📋 Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Historical Data:**")
                st.info(f"• Total matches: {len(features_df):,}")
                st.info(f"• Date range: {features_df['date'].min().strftime('%Y-%m-%d')} to {features_df['date'].max().strftime('%Y-%m-%d')}")
                st.info(f"• Seasons: {features_df['season'].min()}-{features_df['season'].max()}")
            
            with col2:
                st.markdown("**Fixture Data:**")
                st.info(f"• Total fixtures: {len(fixtures_data):,}")
                st.info(f"• Date range: {fixtures_data['date'].min().strftime('%Y-%m-%d')} to {fixtures_data['date'].max().strftime('%Y-%m-%d')}")
                st.info(f"• Unique teams: {len(team_options)}")
            
            # Prediction explanation
            st.markdown("### ❓ How It Works")
            
            st.markdown("""
            **The model uses the following features to make predictions:**
            
            🏠 **Home Team Features:**
            - Historical goals per match (home venue)
            - Historical goals conceded per match (home venue)
            - Win rate in recent matches
            - Recent form (last 5 matches): wins, draws, losses, points
            
            ✈️ **Away Team Features:**
            - Historical goals per match (away venue)
            - Historical goals conceded per match (away venue)
            - Win rate in recent matches
            - Recent form (last 5 matches): wins, draws, losses, points
            
            🤝 **Head-to-Head:**
            - Historical results between the two teams
            
            **Prediction Categories:**
            - 🏠 **Home Win**: The home team is predicted to win
            - 🤝 **Draw**: The match is predicted to end in a draw
            - ✈️ **Away Win**: The away team is predicted to win
            """)
            
            # Model limitations
            st.markdown("### ⚠️ Important Notes")
            st.warning("""
            **Model Limitations:**
            - Predictions are based on historical data and statistical patterns
            - Cannot account for injuries, transfers, or other real-time factors
            - Performance may vary for teams with limited historical data
            - Weather, referee decisions, and other external factors are not considered
            - Use predictions as guidance, not absolute certainty
            """)
    
    except Exception as e:
        st.error(f"Error loading application: {str(e)}")
        st.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
