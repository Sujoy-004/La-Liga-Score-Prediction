import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from model import LaLigaPredictor
import warnings
warnings.filterwarnings('ignore')

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
    .loading {
        text-align: center;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_predictor():
    """Initialize and train the predictor model"""
    predictor = LaLigaPredictor()
    
    # Try to load pre-trained model first
    if predictor.load_model('laliga_model.pkl'):
        st.success("✅ Pre-trained model loaded successfully!")
        return predictor
    
    # If no pre-trained model, train new one
    st.info("🔄 Training new model... This may take a moment.")
    
    if predictor.load_data():
        predictor.preprocess_data()
        predictor.train_models()
        # Save model for future use
        predictor.save_model('laliga_model.pkl')
        st.success("✅ Model trained and ready!")
        return predictor
    else:
        st.error("❌ Failed to load data")
        return None

def display_team_comparison(predictor, home_team, away_team):
    """Display team statistics comparison"""
    df = predictor.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {home_team} (Home)")
        home_home_games = df[df['HomeTeam'] == home_team]
        home_away_games = df[df['AwayTeam'] == home_team]
        
        if not home_home_games.empty:
            home_goals_avg = home_home_games['FTHG'].mean()
            home_conceded_avg = home_home_games['FTAG'].mean()
            st.metric("Avg Goals Scored (Home)", f"{home_goals_avg:.2f}")
            st.metric("Avg Goals Conceded (Home)", f"{home_conceded_avg:.2f}")
            st.metric("Home Games Played", len(home_home_games))
    
    with col2:
        st.subheader(f"📊 {away_team} (Away)")
        away_home_games = df[df['HomeTeam'] == away_team]
        away_away_games = df[df['AwayTeam'] == away_team]
        
        if not away_away_games.empty:
            away_goals_avg = away_away_games['FTAG'].mean()
            away_conceded_avg = away_away_games['FTHG'].mean()
            st.metric("Avg Goals Scored (Away)", f"{away_goals_avg:.2f}")
            st.metric("Avg Goals Conceded (Away)", f"{away_conceded_avg:.2f}")
            st.metric("Away Games Played", len(away_away_games))

def display_head_to_head(predictor, home_team, away_team):
    """Display head-to-head statistics"""
    df = predictor.df
    
    # Get head-to-head matches
    h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
    
    if not h2h.empty:
        st.subheader("🆚 Head-to-Head Record")
        
        # Calculate wins
        home_wins = len(h2h[((h2h['HomeTeam'] == home_team) & (h2h['FTHG'] > h2h['FTAG'])) |
                           ((h2h['AwayTeam'] == home_team) & (h2h['FTAG'] > h2h['FTHG']))])
        
        away_wins = len(h2h[((h2h['HomeTeam'] == away_team) & (h2h['FTHG'] > h2h['FTAG'])) |
                           ((h2h['AwayTeam'] == away_team) & (h2h['FTAG'] > h2h['FTHG']))])
        
        draws = len(h2h[h2h['FTHG'] == h2h['FTAG']])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Wins", home_wins)
        with col2:
            st.metric("Draws", draws)
        with col3:
            st.metric(f"{away_team} Wins", away_wins)
        
        # Show recent matches
        st.subheader("📅 Recent Matches")
        recent_h2h = h2h.tail(5)
        
        display_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        if 'Date' in recent_h2h.columns:
            display_cols = ['Date'] + display_cols
            
        st.dataframe(recent_h2h[display_cols], use_container_width=True)

def main():
    st.markdown('<div class="main-header">⚽ La Liga Score Predictor</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    with st.spinner("Loading predictor model..."):
        predictor = initialize_predictor()
    
    if predictor is None:
        st.error("Failed to initialize predictor. Please check your data files.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🎯 Match Prediction")
    
    # Team selection
    teams = predictor.teams
    
    if not teams:
        st.error("No teams found in data.")
        st.stop()
    
    # Add popular matchups for quick selection
    st.sidebar.subheader("🔥 Popular Matchups")
    popular_matches = [
        ("Real Madrid", "Barcelona"),
        ("Atletico Madrid", "Real Madrid"),
        ("Barcelona", "Atletico Madrid"),
        ("Real Madrid", "Sevilla"),
        ("Barcelona", "Valencia")
    ]
    
    selected_matchup = st.sidebar.selectbox(
        "Quick Select",
        ["Custom Selection"] + [f"{h} vs {a}" for h, a in popular_matches if h in teams and a in teams]
    )
    
    if selected_matchup != "Custom Selection":
        home_default, away_default = selected_matchup.split(" vs ")
    else:
        home_default, away_default = teams[0], teams[1] if len(teams) > 1 else teams[0]
    
    home_team = st.sidebar.selectbox(
        "🏠 Home Team", 
        teams, 
        index=teams.index(home_default) if home_default in teams else 0
    )
    
    away_team = st.sidebar.selectbox(
        "✈️ Away Team", 
        teams,
        index=teams.index(away_default) if away_default in teams else 0
    )
    
    # Predict button
    if st.sidebar.button("🎯 Predict Score", type="primary", use_container_width=True):
        if home_team == away_team:
            st.sidebar.error("Please select different teams!")
        else:
            try:
                with st.spinner("Calculating prediction..."):
                    home_score, away_score = predictor.predict_match(home_team, away_team)
                
                # Display prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Determine result
                    if home_score > away_score:
                        result = f"{home_team} Win"
                        result_color = "#28a745"
                    elif away_score > home_score:
                        result = f"{away_team} Win"
                        result_color = "#dc3545"
                    else:
                        result = "Draw"
                        result_color = "#ffc107"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Predicted Score</h2>
                        <h1 style="font-size: 3rem; margin: 1rem 0;">{home_team} {home_score} - {away_score} {away_team}</h1>
                        <h3 style="color: {result_color}; margin-top: 1rem;">{result}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display team comparisons
                st.markdown("---")
                display_team_comparison(predictor, home_team, away_team)
                
                # Display head-to-head
                st.markdown("---")
                display_head_to_head(predictor, home_team, away_team)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Make sure both teams exist in the training data.")
    
    # Display recent matches
    st.markdown("---")
    st.subheader("📊 Recent La Liga Matches")
    
    if hasattr(predictor, 'df') and not predictor.df.empty:
        recent_matches = predictor.df.tail(15).copy()
        
        # Add result column
        recent_matches['Result'] = recent_matches.apply(
            lambda row: f"{row['FTHG']}-{row['FTAG']}", axis=1
        )
        
        display_cols = ['HomeTeam', 'AwayTeam', 'Result']
        if 'Date' in recent_matches.columns:
            display_cols = ['Date'] + display_cols
            recent_matches['Date'] = recent_matches['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            recent_matches[display_cols].rename(columns={
                'HomeTeam': 'Home Team',
                'AwayTeam': 'Away Team'
            }),
            use_container_width=True
        )
    
    # Model information
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        ### 🤖 La Liga Score Prediction Model
        
        This advanced machine learning model uses **Random Forest Regression** to predict La Liga match scores.
        
        **📈 Features Used:**
        - Team historical performance
        - Home/Away advantage patterns
        - Rolling averages (last 5 games)
        - Seasonal timing factors
        - Head-to-head statistics
        
        **🎯 Model Performance:**
        - Trained on historical La Liga data
        - Uses ensemble learning for better accuracy
        - Considers team form and recent performance
        
        **⚠️ Disclaimer:**
        Predictions are for entertainment purposes only. Football is unpredictable! 
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "⚽ Made with ❤️ for La Liga fans | Predictions for entertainment only"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
