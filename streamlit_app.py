import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class LaLigaPredictor:
    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.le_home = LabelEncoder()
        self.le_away = LabelEncoder()
        self.teams = []
        
    def load_data(self, csv_path='matches_full.csv', excel_path='la-liga-2025-UTC.xlsx'):
        """Load data from CSV or Excel file"""
        try:
            # Try loading CSV first
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} records from CSV")
        except:
            try:
                # Try loading Excel
                self.df = pd.read_excel(excel_path)
                print(f"Loaded {len(self.df)} records from Excel")
            except Exception as e:
                print(f"Error loading data: {e}")
                return False
        
        return True
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        # Convert date column if exists
        date_cols = ['Date', 'date', 'DATE', 'Match Date', 'match_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df['Date'] = pd.to_datetime(self.df[col])
                    break
                except:
                    continue
            
        # Handle different column naming conventions - be more flexible
        home_col = None
        away_col = None
        home_goals_col = None
        away_goals_col = None
        
        # Check for home team column
        home_options = ['HomeTeam', 'home_team', 'Home', 'HOME', 'Home Team', 'home']
        for col in self.df.columns:
            if col in home_options or 'home' in col.lower():
                home_col = col
                break
                
        # Check for away team column  
        away_options = ['AwayTeam', 'away_team', 'Away', 'AWAY', 'Away Team', 'away']
        for col in self.df.columns:
            if col in away_options or 'away' in col.lower():
                away_col = col
                break
                
        # Check for home goals column
        home_goal_options = ['FTHG', 'home_score', 'HomeScore', 'Home Goals', 'home_goals', 'HG']
        for col in self.df.columns:
            if col in home_goal_options or ('home' in col.lower() and ('goal' in col.lower() or 'score' in col.lower())):
                home_goals_col = col
                break
                
        # Check for away goals column
        away_goal_options = ['FTAG', 'away_score', 'AwayScore', 'Away Goals', 'away_goals', 'AG']
        for col in self.df.columns:
            if col in away_goal_options or ('away' in col.lower() and ('goal' in col.lower() or 'score' in col.lower())):
                away_goals_col = col
                break
        
        # If columns not found, print available columns for debugging
        if not all([home_col, away_col, home_goals_col, away_goals_col]):
            print("Available columns:", self.df.columns.tolist())
            print(f"Found - Home: {home_col}, Away: {away_col}, Home Goals: {home_goals_col}, Away Goals: {away_goals_col}")
            
            # Try to use first few columns as fallback
            cols = self.df.columns.tolist()
            if len(cols) >= 4:
                home_col = home_col or cols[0]
                away_col = away_col or cols[1] 
                home_goals_col = home_goals_col or cols[2]
                away_goals_col = away_goals_col or cols[3]
                print(f"Using fallback columns: {home_col}, {away_col}, {home_goals_col}, {away_goals_col}")
        
        # Standardize column names
        column_mapping = {}
        if home_col:
            column_mapping[home_col] = 'HomeTeam'
        if away_col:
            column_mapping[away_col] = 'AwayTeam'
        if home_goals_col:
            column_mapping[home_goals_col] = 'FTHG'
        if away_goals_col:
            column_mapping[away_goals_col] = 'FTAG'
            
        self.df = self.df.rename(columns=column_mapping)
        
        # Check which essential columns actually exist after renaming
        essential_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        existing_cols = [col for col in essential_cols if col in self.df.columns]
        
        if len(existing_cols) < 4:
            print(f"Warning: Only found {len(existing_cols)} essential columns: {existing_cols}")
            # Don't fail here, just work with what we have
        
        # Remove rows with missing essential data only for columns that exist
        if existing_cols:
            self.df = self.df.dropna(subset=existing_cols)
        
        # Make sure we have team data
        if 'HomeTeam' in self.df.columns and 'AwayTeam' in self.df.columns:
            # Encode team names
            all_teams = list(set(list(self.df['HomeTeam'].unique()) + list(self.df['AwayTeam'].unique())))
            self.teams = sorted([str(team) for team in all_teams if pd.notna(team)])
            
            # Fit label encoders on all teams
            if self.teams:
                self.le_home.fit(self.teams)
                self.le_away.fit(self.teams)
                
                self.df['HomeTeam_encoded'] = self.le_home.transform(self.df['HomeTeam'])
                self.df['AwayTeam_encoded'] = self.le_away.transform(self.df['AwayTeam'])
            else:
                raise ValueError("No valid teams found in data")
        else:
            raise ValueError("Home and Away team columns not found")
        
        # Create additional features
        if 'Date' in self.df.columns:
            self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
            self.df['Month'] = self.df['Date'].dt.month
        else:
            self.df['DayOfWeek'] = 5  # Default to Friday
            self.df['Month'] = 3     # Default to March
        
        # Calculate team statistics (rolling averages) only if we have score data
        if 'FTHG' in self.df.columns and 'FTAG' in self.df.columns:
            self.calculate_team_stats()
        else:
            # Create dummy stats
            self.df['home_avg_goals'] = 1.5
            self.df['away_avg_goals'] = 1.0
            self.df['home_avg_conceded'] = 1.2
            self.df['away_avg_conceded'] = 1.3
        
        print("Data preprocessing completed")
        
    def calculate_team_stats(self):
        """Calculate team statistics for better predictions"""
        # Sort by date if available
        if 'Date' in self.df.columns:
            self.df = self.df.sort_values('Date')
        
        # Initialize stats columns
        self.df['home_avg_goals'] = 0.0
        self.df['away_avg_goals'] = 0.0
        self.df['home_avg_conceded'] = 0.0
        self.df['away_avg_conceded'] = 0.0
        
        # Calculate rolling averages (last 5 games)
        for i in range(len(self.df)):
            home_team = self.df.iloc[i]['HomeTeam']
            away_team = self.df.iloc[i]['AwayTeam']
            
            # Get recent home team performance
            recent_home_games = self.df.iloc[:i]
            home_home_games = recent_home_games[recent_home_games['HomeTeam'] == home_team].tail(5)
            home_away_games = recent_home_games[recent_home_games['AwayTeam'] == home_team].tail(5)
            
            # Calculate home team stats
            if not home_home_games.empty:
                self.df.iloc[i, self.df.columns.get_loc('home_avg_goals')] = home_home_games['FTHG'].mean()
                self.df.iloc[i, self.df.columns.get_loc('home_avg_conceded')] = home_home_games['FTAG'].mean()
            
            # Get recent away team performance  
            away_home_games = recent_home_games[recent_home_games['HomeTeam'] == away_team].tail(5)
            away_away_games = recent_home_games[recent_home_games['AwayTeam'] == away_team].tail(5)
            
            # Calculate away team stats
            if not away_away_games.empty:
                self.df.iloc[i, self.df.columns.get_loc('away_avg_goals')] = away_away_games['FTAG'].mean()
                self.df.iloc[i, self.df.columns.get_loc('away_avg_conceded')] = away_away_games['FTHG'].mean()
    
    def train_models(self):
        """Train the prediction models"""
        # Features for training
        feature_cols = [
            'HomeTeam_encoded', 'AwayTeam_encoded', 
            'DayOfWeek', 'Month',
            'home_avg_goals', 'away_avg_goals',
            'home_avg_conceded', 'away_avg_conceded'
        ]
        
        # Make sure we have target columns
        if 'FTHG' not in self.df.columns or 'FTAG' not in self.df.columns:
            print("Warning: Score columns not found. Using dummy data for training.")
            # Create dummy scores for demo purposes
            self.df['FTHG'] = np.random.poisson(1.5, len(self.df))
            self.df['FTAG'] = np.random.poisson(1.0, len(self.df))
        
        X = self.df[feature_cols].fillna(0)
        y_home = self.df['FTHG']
        y_away = self.df['FTAG']
        
        # Split data
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, y_home, test_size=0.2, random_state=42
        )
        _, _, y_away_train, y_away_test = train_test_split(
            X, y_away, test_size=0.2, random_state=42
        )
        
        # Train models
        self.model_home = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.model_away = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        self.model_home.fit(X_train, y_home_train)
        self.model_away.fit(X_train, y_away_train)
        
        # Evaluate models
        home_pred = self.model_home.predict(X_test)
        away_pred = self.model_away.predict(X_test)
        
        print(f"Home goals MSE: {mean_squared_error(y_home_test, home_pred):.3f}")
        print(f"Away goals MSE: {mean_squared_error(y_away_test, away_pred):.3f}")
        print(f"Home goals MAE: {mean_absolute_error(y_home_test, home_pred):.3f}")
        print(f"Away goals MAE: {mean_absolute_error(y_away_test, away_pred):.3f}")
        
        print("Model training completed")
    
    def predict_match(self, home_team, away_team):
        """Predict the score for a specific match"""
        if self.model_home is None or self.model_away is None:
            raise ValueError("Models not trained yet")
        
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError("Team not found in training data")
        
        # Encode teams
        home_encoded = self.le_home.transform([home_team])[0]
        away_encoded = self.le_away.transform([away_team])[0]
        
        # Get team stats (use overall averages as approximation)
        home_stats = self.df[self.df['HomeTeam'] == home_team]
        away_stats = self.df[self.df['AwayTeam'] == away_team]
        
        home_avg_goals = home_stats['FTHG'].mean() if not home_stats.empty and 'FTHG' in self.df.columns else 1.5
        away_avg_goals = away_stats['FTAG'].mean() if not away_stats.empty and 'FTAG' in self.df.columns else 1.0
        home_avg_conceded = home_stats['FTAG'].mean() if not home_stats.empty and 'FTAG' in self.df.columns else 1.2
        away_avg_conceded = away_stats['FTHG'].mean() if not away_stats.empty and 'FTHG' in self.df.columns else 1.3
        
        # Create feature vector
        features = np.array([[
            home_encoded, away_encoded, 5, 3,  # Saturday, March
            home_avg_goals, away_avg_goals,
            home_avg_conceded, away_avg_conceded
        ]])
        
        # Predict scores
        home_score = max(0, round(self.model_home.predict(features)[0]))
        away_score = max(0, round(self.model_away.predict(features)[0]))
        
        return int(home_score), int(away_score)
    
    def save_model(self, filename='laliga_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model_home': self.model_home,
            'model_away': self.model_away,
            'le_home': self.le_home,
            'le_away': self.le_away,
            'teams': self.teams
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='laliga_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_home = model_data['model_home']
            self.model_away = model_data['model_away']
            self.le_home = model_data['le_home']
            self.le_away = model_data['le_away']
            self.teams = model_data['teams']
            
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    predictor = LaLigaPredictor()
    
    if predictor.load_data():
        predictor.preprocess_data()
        predictor.train_models()
        predictor.save_model()
        
        # Test prediction
        try:
            if len(predictor.teams) >= 2:
                team1, team2 = predictor.teams[0], predictor.teams[1]
                home_score, away_score = predictor.predict_match(team1, team2)
                print(f"\nPredicted Score: {team1} {home_score} - {away_score} {team2}")
        except Exception as e:
            print(f"Prediction error: {e}")
