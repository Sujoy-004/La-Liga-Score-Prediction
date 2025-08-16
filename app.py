import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
            raise e
        
        # Load the future fixtures data with error handling
        fixtures_data_url = "https://github.com/Sujoy-004/La-Liga-Score-Prediction/raw/main/la-liga-2025-UTC.xlsx"
        try:
            fixtures_data = pd.read_excel(fixtures_data_url)
        except Exception as e:
            st.error(f"Error loading fixtures data: {str(e)}")
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
def create_simplified_features(historical_data):
    """Create simplified features for faster training"""
    with st.spinner("Creating features..."):
        # Filter to recent seasons only (2020 onwards for speed)
        matches_df = historical_data[historical_data['season'] >= 2020].copy()
        
        # Create match-level features
        features_list = []
        
        # Group by season and process matches
        for season in matches_df['season'].unique():
            season_matches = matches_df[matches_df['season'] == season].sort_values('date')
            
            for idx, match in season_matches.iterrows():
                if match['venue'] != 'Home':  # Only process home matches
                    continue
                
                home_team = match['team']
                away_team = match['opponent']
                match_date = match['date']
                result = match['result']
                
                # Get historical data for both teams up to this match date
                home_history = season_matches[
                    (season_matches['team'] == home_team) & 
                    (season_matches['date'] < match_date)
                ]
                away_history = season_matches[
                    (season_matches['team'] == away_team) & 
                    (season_matches['date'] < match_date)
                ]
                
                # Calculate simple features
                home_goals_avg = home_history['gf'].mean() if len(home_history) > 0 else 1.5
                home_conceded_avg = home_history['ga'].mean() if len(home_history) > 0 else 1.5
                away_goals_avg = away_history['gf'].mean() if len(away_history) > 0 else 1.5
                away_conceded_avg = away_history['ga'].mean() if len(away_history) > 0 else 1.5
                
                # Recent form (last 3 matches)
                home_recent = home_history.tail(3)
                away_recent = away_history.tail(3)
                
                home_recent_wins = (home_recent['result'] == 'W').sum()
                home_recent_points = (home_recent['result'] == 'W').sum() * 3 + (home_recent['result'] == 'D').sum()
                away_recent_wins = (away_recent['result'] == 'W').sum()
                away_recent_points = (away_recent['result'] == 'W').sum() * 3 + (away_recent['result'] == 'D').sum()
                
                # Head to head (simplified)
                h2h_matches = matches_df[
                    ((matches_df['team'] == home_team) & (matches_df['opponent'] == away_team)) |
                    ((matches_df['team'] == away_team) & (matches_df['opponent'] == home_team))
                ]
                h2h_home_wins = ((h2h_matches['team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
                h2h_away_wins = ((h2h_matches['team'] == away_team) & (h2h_matches['result'] == 'W')).sum()
                
                features = {
                    'date': match_date,
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals_avg': home_goals_avg,
                    'home_conceded_avg': home_conceded_avg,
                    'away_goals_avg': away_goals_avg,
                    'away_conceded_avg': away_conceded_avg,
                    'home_recent_wins': home_recent_wins,
                    'home_recent_points': home_recent_points,
                    'away_recent_wins': away_recent_wins,
                    'away_recent_points': away_recent_points,
                    'h2h_home_wins': h2h_home_wins,
                    'h2h_away_wins': h2h_away_wins,
                    'result': result
                }
                
                features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Target encoding
        target_mapping = {'W': 2, 'D': 1, 'L': 0}
        features_df['target'] = features_df['result'].map(target_mapping)
        
        return features_df

@st.cache_resource
def train_simplified_model(features_df):
    """Train a simplified model for faster performance"""
    with st.spinner("Training model..."):
        # Select features
        model_features = [
            'season', 'home_goals_avg', 'home_conceded_avg',
            'away_goals_avg', 'away_conceded_avg',
            'home_recent_wins', 'home_recent_points',
            'away_recent_wins', 'away_recent_points',
            'h2h_home_wins', 'h2h_away_wins'
        ]
        
        # Split data
        train_mask = features_df['season'] < 2024
        val_mask = features_df['season'] == 2024
        
        X_train = features_df.loc[train_mask, model_features].fillna(0)
        y_train = features_df.loc[train_mask, 'target']
        X_val = features_df.loc[val_mask, model_features].fillna(0)
        y_val = features_df.loc[val_mask, 'target']
        
        # Use only Random Forest for speed
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        if len(X_val) > 0:
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
        else:
            accuracy = 0.0
        
        return model, accuracy, model_features

def calculate_team_stats(team_name, historical_data, is_home=True):
    """Calculate current team statistics"""
    # Get recent matches (last 10 matches)
    team_matches = historical_data[
        (historical_data['team'] == team_name) | (historical_data['opponent'] == team_name)
    ].sort_values('date').tail(10)
    
    if len(team_matches) == 0:
        return {
            'goals_avg': 1.5,
            'conceded_avg': 1.5,
            'recent_wins': 1,
            'recent_points': 3
        }
    
    # Calculate stats based on team's perspective
    goals_scored = []
    goals_conceded = []
    results = []
    
    for _, match in team_matches.iterrows():
        if match['team'] == team_name:
            goals_scored.append(match['gf'])
            goals_conceded.append(match['ga'])
            results.append(match['result'])
        else:
            goals_scored.append(match['ga'])
            goals_conceded.append(match['gf'])
            # Flip the result
            if match['result'] == 'W':
                results.append('L')
            elif match['result'] == 'L':
                results.append('W')
            else:
                results.append('D')
    
    return {
        'goals_avg': np.mean(goals_scored) if goals_scored else 1.5,
        'conceded_avg': np.mean(goals_conceded) if goals_conceded else 1.5,
        'recent_wins': sum(1 for r in results[-3:] if r == 'W'),
        'recent_points': sum(3 if r == 'W' else (1 if r == 'D' else 0) for r in results[-3:])
    }

def predict_match(home_team, away_team, model, model_features, historical_data):
    """Predict match result without requiring date input"""
    try:
        # Calculate current stats for both teams
        home_stats = calculate_team_stats(home_team, historical_data, is_home=True)
        away_stats = calculate_team_stats(away_team, historical_data, is_home=False)
        
        # Calculate head-to-head
        h2h_matches = historical_data[
            ((historical_data['team'] == home_team) & (historical_data['opponent'] == away_team)) |
            ((historical_data['team'] == away_team) & (historical_data['opponent'] == home_team))
        ]
        
        h2h_home_wins = ((h2h_matches['team'] == home_team) & (h2h_matches['result'] == 'W')).sum()
        h2h_away_wins = ((h2h_matches['team'] == away_team) & (h2h_matches['result'] == 'W')).sum()
        
        # Create feature vector
        features = {
            'season': 2025,  # Current season
            'home_goals_avg': home_stats['goals_avg'],
            'home_conceded_avg': home_stats['conceded_avg'],
            'away_goals_avg': away_stats['goals_avg'],
            'away_conceded_avg': away_stats['conceded_avg'],
            'home_recent_wins': home_stats['recent_wins'],
            'home_recent_points': home_stats['recent_points'],
            'away_recent_wins': away_stats['recent_wins'],
            'away_recent_points': away_stats['recent_points'],
            'h2h_home_wins': h2h_home_wins,
            'h2h_away_wins': h2h_away_wins
        }
        
        # Create DataFrame
        feature_df = pd.DataFrame([features])[model_features].fillna(0)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]
        
        return prediction, probabilities, None
        
    except Exception as e:
        return None, None, str(e)

# Main Streamlit App
def main():
    # Header
    st.markdown('<div class="main-header">⚽ La Liga Score Prediction</div>', unsafe_allow_html=True)
    
    # Initialize session state for data loading
    if 'data_loaded' not in st.session_state:
        try:
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
    
    # Initialize session state for features
    if 'features_ready' not in st.session_state:
        try:
            features_df = create_simplified_features(historical_data)
            st.session_state.features_df = features_df
            st.session_state.features_ready = True
        except Exception as e:
            st.error(f"Failed to create features: {str(e)}")
            st.stop()
    else:
        features_df = st.session_state.features_df
    
    # Initialize session state for model
    if 'model_trained' not in st.session_state:
        try:
            model, accuracy, model_features = train_simplified_model(features_df)
            st.session_state.model = model
            st.session_state.accuracy = accuracy
            st.session_state.model_features = model_features
            st.session_state.model_trained = True
        except Exception as e:
            st.error(f"Failed to train model: {str(e)}")
            st.stop()
    else:
        model = st.session_state.model
        accuracy = st.session_state.accuracy
        model_features = st.session_state.model_features
    
    # Sidebar
    st.sidebar.markdown("## 📊 App Status")
    if st.session_state.get('data_loaded', False):
        st.sidebar.success("✅ Data loaded")
    if st.session_state.get('features_ready', False):
        st.sidebar.success("✅ Features ready")
    if st.session_state.get('model_trained', False):
        st.sidebar.success("✅ Model trained")
    
    st.sidebar.markdown("## 🏆 Model Performance")
    st.sidebar.metric("🟢 Model Accuracy", f"{accuracy:.3f}")
    
    # Get team options
    all_teams = set(fixtures_data['home_team'].unique()) | set(fixtures_data['away_team'].unique())
    team_options = sorted(list(all_teams))
    
    # Add app info
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.info(f"""
    **Teams**: {len(team_options)}
    **Historical Matches**: {len(features_df):,}
    **Upcoming Fixtures**: {len(fixtures_data):,}
    **Features Used**: {len(model_features)}
    """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Match", "📅 Season Fixtures", "📊 Model Insights"])
    
    with tab1:
        st.markdown('<div class="sub-header">Predict Individual Match</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("🏠 Home Team", team_options, key="home")
            
        with col2:
            away_team = st.selectbox("✈️ Away Team", team_options, key="away")
        
        if st.button("🔮 Predict Match Result", type="primary"):
            if home_team != away_team:
                with st.spinner("🔮 Making prediction..."):
                    prediction_encoded, prediction_proba, error = predict_match(
                        home_team, away_team, model, model_features, historical_data
                    )
                
                if error:
                    st.error(f"❌ Error making prediction: {error}")
                else:
                    # Convert prediction
                    result_mapping = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
                    predicted_result = result_mapping[prediction_encoded]
                    
                    # Display prediction
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
                # Filter fixtures
                filtered_fixtures = fixtures_data.copy()
                
                if selected_teams:
                    mask = (
                        filtered_fixtures['home_team'].isin(selected_teams) | 
                        filtered_fixtures['away_team'].isin(selected_teams)
                    )
                    filtered_fixtures = filtered_fixtures[mask]
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (
                        (filtered_fixtures['date'].dt.date >= start_date) & 
                        (filtered_fixtures['date'].dt.date <= end_date)
                    )
                    filtered_fixtures = filtered_fixtures[mask]
                
                if len(filtered_fixtures) == 0:
                    st.warning("⚠️ No fixtures found with the selected filters.")
                else:
                    # Make predictions
                    st.info(f"🎯 Predicting {len(filtered_fixtures)} fixtures...")
                    
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, (_, fixture) in enumerate(filtered_fixtures.iterrows()):
                        pred_encoded, pred_proba, error = predict_match(
                            fixture['home_team'], 
                            fixture['away_team'], 
                            model, 
                            model_features, 
                            historical_data
                        )
                        
                        if error:
                            predictions.append('Error')
                            probabilities.append(0.0)
                        else:
                            result_mapping = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
                            predictions.append(result_mapping[pred_encoded])
                            probabilities.append(max(pred_proba))
                        
                        progress_bar.progress((idx + 1) / len(filtered_fixtures))
                    
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
                        st.metric("Total Fixtures", len(filtered_fixtures))
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
                st.error(f"❌ Error processing fixtures: {str(e)}")
    
    with tab3:
        st.markdown('<div class="sub-header">Model Insights & Performance</div>', unsafe_allow_html=True)
        
        # Model performance
        st.markdown("### 🏆 Model Performance")
        st.metric("Model Accuracy", f"{accuracy:.3f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': model_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
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
        
        # How it works
        st.markdown("### ❓ How It Works")
        st.markdown("""
        **The model uses simplified features for faster predictions:**
        
        🏠 **Team Statistics:**
        - Average goals scored and conceded (last 10 matches)
        - Recent form: wins and points from last 3 matches
        
        🤝 **Head-to-Head:**
        - Historical wins between the two teams
        
        **Prediction Categories:**
        - 🏠 **Home Win**: The home team is predicted to win
        - 🤝 **Draw**: The match is predicted to end in a draw
        - ✈️ **Away Win**: The away team is predicted to win
        
        **Key Improvements:**
        - ⚡ Faster training with simplified features
        - 🎯 No date input required - uses current team form
        - 🔧 Fixed filtering issues in season fixtures
        """)

if __name__ == "__main__":
    main()
