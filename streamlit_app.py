import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd

# Debug data loading
try:
    df = pd.read_csv('matches_full.csv')
    st.write(f"Data loaded successfully: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    
    # Check for team columns
    if 'home_team' in df.columns:
        teams = df['home_team'].unique()
        st.write(f"Found {len(teams)} teams")
    elif 'Home' in df.columns:
        teams = df['Home'].unique()
        st.write(f"Found {len(teams)} teams")
    else:
        st.error("No team column found. Available columns:", df.columns.tolist())
        
except Exception as e:
    st.error(f"Error loading data: {e}")

# Page config
st.set_page_config(
    page_title="La Liga Score Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .team-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Load your CSV data
        df = pd.read_csv('matches_full.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load or train the prediction model"""
    try:
        # Try to load pre-trained model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If no saved model, train a new one
        df = load_data()
        if df is not None:
            model = train_model(df)
            return model
        return None

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Encode categorical variables
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    
    if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
        df['HomeTeam_encoded'] = le_home.fit_transform(df['HomeTeam'])
        df['AwayTeam_encoded'] = le_away.fit_transform(df['AwayTeam'])
    
    # Create additional features
    df['DayOfWeek'] = df['Date'].dt.dayofweek if 'Date' in df.columns else 0
    df['Month'] = df['Date'].dt.month if 'Date' in df.columns else 1
    
    return df, le_home, le_away

def train_model(df):
    """Train the prediction model"""
    df_processed, le_home, le_away = preprocess_data(df)
    
    # Features for training
    feature_cols = ['HomeTeam_encoded', 'AwayTeam_encoded', 'DayOfWeek', 'Month']
    
    # Check if target columns exist
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        X = df_processed[feature_cols].fillna(0)
        y_home = df_processed['FTHG'].fillna(0)
        y_away = df_processed['FTAG'].fillna(0)
        
        # Train separate models for home and away scores
        model_home = RandomForestRegressor(n_estimators=100, random_state=42)
        model_away = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model_home.fit(X, y_home)
        model_away.fit(X, y_away)
        
        return {
            'model_home': model_home,
            'model_away': model_away,
            'le_home': le_home,
            'le_away': le_away,
            'teams': list(df['HomeTeam'].unique()) if 'HomeTeam' in df.columns else []
        }
    
    return None

def predict_score(model, home_team, away_team):
    """Predict the score for a match"""
    if model is None:
        return None, None
    
    try:
        # Encode team names
        home_encoded = model['le_home'].transform([home_team])[0]
        away_encoded = model['le_away'].transform([away_team])[0]
        
        # Create feature vector
        features = np.array([[home_encoded, away_encoded, 5, 3]]).reshape(1, -1)  # Default values
        
        # Predict scores
        home_score = max(0, int(round(model['model_home'].predict(features)[0])))
        away_score = max(0, int(round(model['model_away'].predict(features)[0])))
        
        return home_score, away_score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app
def main():
    st.markdown('<div class="main-header">⚽ La Liga Score Predictor</div>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    if df is None:
        st.error("Could not load data. Please check if matches_full.csv exists.")
        return
    
    # Sidebar
    st.sidebar.header("Match Prediction")
    
    # Get available teams
    teams = []
    if 'HomeTeam' in df.columns:
        teams = sorted(df['HomeTeam'].unique())
    
    if not teams:
        st.error("No teams found in data.")
        return
    
    # Team selection
    home_team = st.sidebar.selectbox("Home Team", teams)
    away_team = st.sidebar.selectbox("Away Team", teams)
    
    # Predict button
    if st.sidebar.button("Predict Score", type="primary"):
        if home_team != away_team:
            if model is not None:
                home_score, away_score = predict_score(model, home_team, away_team)
                
                if home_score is not None and away_score is not None:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted Score</h2>
                            <h1>{home_team} {home_score} - {away_score} {away_team}</h1>
                            <p>Match Prediction</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show team statistics
                if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{home_team} (Home)")
                        home_stats = df[df['HomeTeam'] == home_team]
                        if not home_stats.empty and 'FTHG' in df.columns:
                            avg_goals = home_stats['FTHG'].mean()
                            st.metric("Average Goals Scored (Home)", f"{avg_goals:.2f}")
                    
                    with col2:
                        st.subheader(f"{away_team} (Away)")
                        away_stats = df[df['AwayTeam'] == away_team]
                        if not away_stats.empty and 'FTAG' in df.columns:
                            avg_goals = away_stats['FTAG'].mean()
                            st.metric("Average Goals Scored (Away)", f"{avg_goals:.2f}")
            else:
                st.error("Model not loaded. Please check your data.")
        else:
            st.sidebar.error("Please select different teams!")
    
    # Display recent matches
    if not df.empty:
        st.subheader("Recent Matches")
        recent_matches = df.tail(10)
        
        if 'Date' in df.columns and 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            display_cols = ['Date', 'HomeTeam', 'AwayTeam']
            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                display_cols.extend(['FTHG', 'FTAG'])
            
            st.dataframe(recent_matches[display_cols], use_container_width=True)
        else:
            st.dataframe(recent_matches.head(), use_container_width=True)
    
    # Model info
    with st.expander("About the Model"):
        st.write("""
        This La Liga score prediction model uses machine learning to predict match outcomes.
        The model considers factors such as:
        - Team historical performance
        - Home/Away advantage
        - Recent form
        - Match timing (day of week, month)
        
        **Note:** Predictions are for entertainment purposes only.
        """)

if __name__ == "__main__":
    main()
