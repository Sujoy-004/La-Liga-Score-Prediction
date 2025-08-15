import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="La Liga Score Predictor",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Try CSV first
        df = pd.read_csv('matches_full.csv')
        st.success(f"Loaded {len(df)} records from CSV")
        return df
    except:
        try:
            # Try Excel
            df = pd.read_excel('la-liga-2025-UTC.xlsx')
            st.success(f"Loaded {len(df)} records from Excel")
            return df
        except Exception as e:
            st.error(f"Could not load data: {e}")
            return None

def identify_columns(df):
    """Identify the correct column names in the dataset"""
    columns = df.columns.tolist()
    
    # Common variations for team columns
    home_team_options = ['HomeTeam', 'Home Team', 'home_team', 'home', 'Home', 'team1', 'Team1']
    away_team_options = ['AwayTeam', 'Away Team', 'away_team', 'away', 'Away', 'team2', 'Team2']
    
    # Common variations for score columns
    home_score_options = ['FTHG', 'FT_home_score', 'home_score', 'HomeScore', 'Home Score', 'score1', 'goals_home']
    away_score_options = ['FTAG', 'FT_away_score', 'away_score', 'AwayScore', 'Away Score', 'score2', 'goals_away']
    
    home_team_col = None
    away_team_col = None
    home_score_col = None
    away_score_col = None
    
    # Find matching columns
    for col in columns:
        if col in home_team_options or any(opt.lower() in col.lower() for opt in home_team_options):
            if home_team_col is None:
                home_team_col = col
        
        if col in away_team_options or any(opt.lower() in col.lower() for opt in away_team_options):
            if away_team_col is None:
                away_team_col = col
        
        if col in home_score_options or any(opt.lower() in col.lower() for opt in home_score_options):
            if home_score_col is None:
                home_score_col = col
        
        if col in away_score_options or any(opt.lower() in col.lower() for opt in away_score_options):
            if away_score_col is None:
                away_score_col = col
    
    return {
        'home_team': home_team_col,
        'away_team': away_team_col,
        'home_score': home_score_col,
        'away_score': away_score_col,
        'all_columns': columns
    }

@st.cache_resource
def train_model(df):
    """Train a prediction model"""
    if df is None:
        return None
    
    # Identify columns
    col_mapping = identify_columns(df)
    
    st.write("**Column Detection:**")
    st.write(f"- Home Team: {col_mapping['home_team']}")
    st.write(f"- Away Team: {col_mapping['away_team']}")
    st.write(f"- Home Score: {col_mapping['home_score']}")
    st.write(f"- Away Score: {col_mapping['away_score']}")
    st.write(f"- All columns: {col_mapping['all_columns']}")
    
    # Check if we found the necessary columns
    if not all([col_mapping['home_team'], col_mapping['away_team']]):
        st.error("Could not identify team columns. Please check your data format.")
        return None
    
    try:
        # Rename columns to standard format
        df_work = df.copy()
        df_work = df_work.rename(columns={
            col_mapping['home_team']: 'HomeTeam',
            col_mapping['away_team']: 'AwayTeam'
        })
        
        # Handle score columns if available
        if col_mapping['home_score'] and col_mapping['away_score']:
            df_work = df_work.rename(columns={
                col_mapping['home_score']: 'FTHG',
                col_mapping['away_score']: 'FTAG'
            })
        else:
            # If no score columns, create dummy scores for demonstration
            st.warning("No score columns found. Creating dummy scores for demonstration.")
            df_work['FTHG'] = np.random.randint(0, 4, len(df_work))
            df_work['FTAG'] = np.random.randint(0, 4, len(df_work))
        
        # Remove rows with missing team data
        df_work = df_work.dropna(subset=['HomeTeam', 'AwayTeam'])
        
        if len(df_work) == 0:
            st.error("No valid data found after cleaning")
            return None
        
        # Get unique teams
        home_teams = df_work['HomeTeam'].unique()
        away_teams = df_work['AwayTeam'].unique()
        teams = sorted(list(set(list(home_teams) + list(away_teams))))
        
        st.success(f"Found {len(teams)} unique teams: {teams[:10]}{'...' if len(teams) > 10 else ''}")
        
        # Encode teams
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        
        le_home.fit(teams)
        le_away.fit(teams)
        
        df_work['HomeTeam_encoded'] = le_home.transform(df_work['HomeTeam'])
        df_work['AwayTeam_encoded'] = le_away.transform(df_work['AwayTeam'])
        
        # Prepare features
        X = df_work[['HomeTeam_encoded', 'AwayTeam_encoded']].values
        y_home = df_work['FTHG'].values
        y_away = df_work['FTAG'].values
        
        # Train models
        model_home = RandomForestRegressor(n_estimators=100, random_state=42)
        model_away = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model_home.fit(X, y_home)
        model_away.fit(X, y_away)
        
        return {
            'model_home': model_home,
            'model_away': model_away,
            'le_home': le_home,
            'le_away': le_away,
            'teams': teams,
            'data': df_work,
            'col_mapping': col_mapping
        }
    
    except Exception as e:
        st.error(f"Model training error: {e}")
        st.exception(e)
        return None

def predict_score(model_data, home_team, away_team):
    """Predict match score"""
    try:
        home_encoded = model_data['le_home'].transform([home_team])[0]
        away_encoded = model_data['le_away'].transform([away_team])[0]
        
        features = np.array([[home_encoded, away_encoded]])
        
        home_score = max(0, int(round(model_data['model_home'].predict(features)[0])))
        away_score = max(0, int(round(model_data['model_away'].predict(features)[0])))
        
        return home_score, away_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app
def main():
    st.title("⚽ La Liga Score Predictor")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Show data info
    st.success(f"Data loaded successfully! Shape: {df.shape}")
    
    # Show first few rows and column info
    with st.expander("Data Preview & Column Info", expanded=True):
        st.subheader("Columns in dataset:")
        st.write(df.columns.tolist())
        
        st.subheader("First 5 rows:")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Data types:")
        st.write(df.dtypes.to_dict())
    
    # Train model
    with st.spinner("Training model..."):
        model_data = train_model(df)
    
    if model_data is None:
        st.stop()
    
    st.success("Model trained successfully!")
    
    # Show available teams
    teams = model_data['teams']
    st.sidebar.header("Match Prediction")
    st.sidebar.write(f"Available teams: {len(teams)}")
    
    # Team selection
    home_team = st.sidebar.selectbox("Home Team", teams, key="home")
    away_team = st.sidebar.selectbox("Away Team", teams, key="away")
    
    # Prediction
    if st.sidebar.button("Predict Score", type="primary"):
        if home_team != away_team:
            home_score, away_score = predict_score(model_data, home_team, away_team)
            
            if home_score is not None:
                st.success(f"**Predicted Score:** {home_team} {home_score} - {away_score} {away_team}")
                
                # Show some stats if score columns exist
                if model_data['col_mapping']['home_score'] and model_data['col_mapping']['away_score']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{home_team} Stats")
                        home_matches = model_data['data'][model_data['data']['HomeTeam'] == home_team]
                        if not home_matches.empty:
                            st.metric("Home Matches", len(home_matches))
                            st.metric("Avg Home Goals", f"{home_matches['FTHG'].mean():.2f}")
                    
                    with col2:
                        st.subheader(f"{away_team} Stats")
                        away_matches = model_data['data'][model_data['data']['AwayTeam'] == away_team]
                        if not away_matches.empty:
                            st.metric("Away Matches", len(away_matches))
                            st.metric("Avg Away Goals", f"{away_matches['FTAG'].mean():.2f}")
        else:
            st.sidebar.error("Please select different teams!")
    
    # Show recent matches
    st.subheader("Sample Data")
    display_df = model_data['data'][['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].head(10)
    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()
