import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import argparse
from src.ml_logic import get_historical_data, calculate_team_stats

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Load Data on Startup (for dropdowns and radar)
historical_data, _ = get_historical_data()
all_teams = sorted(historical_data['home_team'].unique())

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS for glass-effect and "Quiet Luxury"
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>La Liga Deep Analytics</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #1a1a1a; color: #e0e0e0; font-family: 'Inter', sans-serif; }
            .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 25px; margin-bottom: 20px; }
            .header-accent { color: #00d2ff; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
            .dropdown-custom .Select-control { background-color: #2d2d2d !important; border: 1px solid #444 !important; color: white !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("⚽ LA LIGA DEEP ANALYTICS", className="header-accent text-center mt-5 mb-2"), width=12),
        dbc.Col(html.P("Phase 4: Contextual Awareness | Multi-Service Architecture", className="text-center text-muted mb-5"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Home Team", className="mb-2"),
                dcc.Dropdown(id='home-dropdown', options=[{'label': t, 'value': t} for t in all_teams], value='Real Madrid', className="dropdown-custom"),
                html.Label("Away Team", className="mt-3 mb-2"),
                dcc.Dropdown(id='away-dropdown', options=[{'label': t, 'value': t} for t in all_teams], value='Barcelona', className="dropdown-custom"),
                dbc.Button("ANALYZE FIXTURE", id='analyze-btn', color="primary", className="w-100 mt-4", style={'backgroundColor': '#00d2ff', 'border': 'none'})
            ], className="glass-card")
        ], width=4),
        
        dbc.Col([
            html.Div([
                dcc.Graph(id='radar-chart', config={'displayModeBar': False})
            ], className="glass-card")
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='gauge-chart', config={'displayModeBar': False}),
                html.Div(id='prediction-text', className="text-center mt-3", style={'fontSize': '1.2em', 'fontWeight': 'bold'})
            ], className="glass-card")
        ], width=6),
        
        dbc.Col([
            html.Div([
                html.H4("Model Methodology", className="header-accent mb-3"),
                html.P("Our Stacked Ensemble architecture utilizes a VotingClassifier combining Random Forest, XGBoost, and LightGBM."),
                html.P("Decoupled Architecture: This frontend communicates with a FastAPI backend via REST for high-performance inference."),
                html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
                html.Div(id='insight-text', className="text-muted small")
            ], className="glass-card h-100")
        ], width=6)
    ])
], fluid=True)

@app.callback(
    [Output('radar-chart', 'figure'),
     Output('gauge-chart', 'figure'),
     Output('prediction-text', 'children'),
     Output('insight-text', 'children')],
    [Input('analyze-btn', 'n_clicks')],
    [State('home-dropdown', 'value'),
     State('away-dropdown', 'value')]
)
def update_dashboard(n_clicks, home, away):
    if not home or not away:
        return go.Figure(), go.Figure(), "", ""
    
    # 1. Call FastAPI Backend
    try:
        response = requests.get(f"{API_BASE_URL}/predict", params={"home_team": home, "away_team": away})
        api_data = response.json()
        prediction = api_data['prediction']
        p_home = float(api_data['probability_home_win'].replace('%', '')) / 100
        insights = api_data['insights']
    except Exception as e:
        return go.Figure(), go.Figure(), f"API ERROR: {str(e)}", ""
    
    # 2. Radar Chart (Local Calc for UI speed)
    home_stats = calculate_team_stats(home, historical_data, is_home=True)
    away_stats = calculate_team_stats(away, historical_data, is_home=False)
    
    categories = ['Attack', 'Defense', 'Form', 'H2H']
    home_vals = [home_stats['goals_avg'], 3 - home_stats['conceded_avg'], home_stats['recent_points']/9*3, home_stats['venue_stat']]
    away_vals = [away_stats['goals_avg'], 3 - away_stats['conceded_avg'], away_stats['recent_points']/9*3, away_stats['venue_stat']]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=home_vals, theta=categories, fill='toself', name=home, line_color='#00d2ff'))
    fig_radar.add_trace(go.Scatterpolar(r=away_vals, theta=categories, fill='toself', name=away, line_color='#ffd700'))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 3], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=True, title=f"Tactical Comparison: {home} vs {away}", margin=dict(t=50, b=20, l=30, r=30)
    )
    
    # 3. Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = p_home * 100, title = {'text': f"{home} Win Probability", 'font': {'size': 18, 'color': '#00d2ff'}},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#00d2ff"}, 'steps': [{'range': [0, 55], 'color': 'rgba(255,255,255,0.05)'}, {'range': [55, 100], 'color': 'rgba(0, 210, 255, 0.2)'}], 'threshold': {'line': {'color': "gold", 'width': 4}, 'value': 55}}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Inter"})
    
    pred_text = f"ANALYSIS RESULT: {prediction.upper()}"
    insight_text = [html.P(f"• {ins}") for ins in insights]
    
    return fig_radar, fig_gauge, pred_text, insight_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    app.run_server(host='0.0.0.0', port=args.port, debug=False)
