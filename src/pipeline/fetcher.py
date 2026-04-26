import requests
import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

# API Configuration
API_URL = "https://v3.football.api-sports.io/fixtures"
LEAGUE_ID = 140 # La Liga
SEASON = 2024   # Current active season
HEADERS = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': os.getenv('FOOTBALL_API_KEY')
}

def fetch_and_upsert():
    if not os.getenv('FOOTBALL_API_KEY'):
        print("Error: FOOTBALL_API_KEY not found in environment variables.")
        return

    # 1. Fetch Fixtures (Past 7 and Next 7 days)
    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    date_to = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        'league': LEAGUE_ID,
        'season': SEASON,
        'from': date_from,
        'to': date_to
    }

    print(f"Fetching fixtures from {date_from} to {date_to}...")
    response = requests.get(API_URL, headers=HEADERS, params=params)
    data = response.json()

    if data.get('errors'):
        print(f"API Error: {data['errors']}")
        return

    fixtures = data.get('response', [])
    print(f"Retrieved {len(fixtures)} fixtures.")

    # 2. Process and Upsert into SQLite
    conn = sqlite3.connect('data/la_liga.db')
    cursor = conn.cursor()

    for item in fixtures:
        f = item['fixture']
        teams = item['teams']
        goals = item['goals']
        
        # Structure for 'matches' table
        # Note: We need two entries per match (one for home, one for away) to maintain 'ml_logic' parity
        match_date = f['date'][:10]
        
        # Home Team Entry
        cursor.execute("""
            INSERT INTO matches (date, team, opponent, venue, gf, ga, result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, team) DO UPDATE SET
            gf = excluded.gf,
            ga = excluded.ga,
            result = excluded.result
        """, (
            match_date, 
            teams['home']['name'], 
            teams['away']['name'], 
            'Home', 
            goals['home'], 
            goals['away'],
            'W' if goals['home'] > goals['away'] else ('D' if goals['home'] == goals['away'] else 'L') if goals['home'] is not None else None
        ))

        # Away Team Entry
        cursor.execute("""
            INSERT INTO matches (date, team, opponent, venue, gf, ga, result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, team) DO UPDATE SET
            gf = excluded.gf,
            ga = excluded.ga,
            result = excluded.result
        """, (
            match_date, 
            teams['away']['name'], 
            teams['home']['name'], 
            'Away', 
            goals['away'], 
            goals['home'],
            'W' if goals['away'] > goals['home'] else ('D' if goals['away'] == goals['home'] else 'L') if goals['away'] is not None else None
        ))

    conn.commit()
    conn.close()
    print("Upsert completed successfully.")

if __name__ == "__main__":
    fetch_and_upsert()
