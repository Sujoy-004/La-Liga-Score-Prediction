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

@st.cache_resource
def train_model(df):
    """Train a simple prediction model"""
    if df is None:
        return None
    
    # Simple preprocessing
    try:
        # Handle different column names
        if 'HomeTeam' not in df.columns:
            if 'home_team' in df.columns:
                df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
            else:
                st.error("Could not find team columns in data")
                return None
        
        if 'FTHG' not in df.columns:
            if 'home_score' in df.columns:
                df = df.rename(columns={'home_score': 'FTHG', 'away_score': 'FTAG'})
            else:
                st.error("Could not find score columns in data")
                return None
        
        # Remove rows with missing data
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        
        if len(df) == 0:
            st.error("No valid data found after cleaning")
            return None
        
        # Encode teams
        teams = sorted(list(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist())))
        
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        
        le_home.fit(teams)
        le_away.fit(teams)
        
        df['HomeTeam_encoded'] = le_home.transform(df['HomeTeam'])
        df['AwayTeam_encoded'] = le_away.transform(df['AwayTeam'])
        
        # Prepare features
        X = df[['HomeTeam_encoded', 'AwayTeam_encoded']].values
        y_home = df['FTHG'].values
        y_away = df['FTAG'].values
        
        # Train models
        model_home = RandomForestRegressor(n_estimators=50, random_state=42)
        model_away = RandomForestRegressor(n_estimators=50, random_state=42)
        
        model_home.fit(X, y_home)
        model_away.fit(X, y_away)
        
        return {
            'model_home': model_home,
            'model_away': model_away,
            'le_home': le_home,
            'le_away': le_away,
            'teams': teams,
            'data': df
        }
    
    except Exception as e:
        st.error(f"Model training error: {e}")
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
    col1, col2 = st.sidebar.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, key="home")
    with col2:
        away_team = st.selectbox("Away Team", teams, key="away")
    
    # Prediction
    if st.sidebar.button("Predict Score", type="primary"):
        if home_team != away_team:
            home_score, away_score = predict_score(model_data, home_team, away_team)
            
            if home_score is not None:
                st.success(f"**Predicted Score:** {home_team} {home_score} - {away_score} {away_team}")
                
                # Show some stats
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
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Debug info
    with st.expander("Debug Info"):
        st.write("Columns in dataset:", df.columns.tolist())
        st.write("Data types:", df.dtypes.to_dict())
        st.write("Dataset shape:", df.shape)

if __name__ == "__main__":
    main()
