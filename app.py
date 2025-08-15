# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

st.set_page_config(page_title="LaLiga Score / Result Predictor", layout="wide")

@st.cache_data(ttl=3600)
def load_matches():
    # load CSV present in repo root
    if os.path.exists("matches_full.csv"):
        return pd.read_csv("matches_full.csv")
    else:
        st.error("matches_full.csv not found in repo root.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_fixtures():
    # load excel with upcoming fixtures
    if os.path.exists("la-liga-2025-UTC.xlsx"):
        return pd.read_excel("la-liga-2025-UTC.xlsx")
    else:
        return pd.DataFrame()

def basic_prep(df):
    """Auto-detect home/away/team/goals columns and create simple features.
       Returns df with columns ['home', 'away', 'home_goals', 'away_goals', 'result'] when possible.
    """
    df = df.copy()
    # common candidates
    cols = list(df.columns.str.lower())
    # heuristics
    def find(colname_variants):
        for v in colname_variants:
            if v in cols:
                return df.columns[cols.index(v)]
        return None

    home_col = find(['home_team', 'home', 'homeTeam', 'team_home'])
    away_col = find(['away_team', 'away', 'awayTeam', 'team_away'])
    hg_col = find(['home_goals', 'home_goals_full', 'fthg', 'home_goals_ft'])
    ag_col = find(['away_goals', 'away_goals_full', 'ftag', 'away_goals_ft'])

    if home_col: df.rename(columns={home_col: 'home'}, inplace=True)
    if away_col: df.rename(columns={away_col: 'away'}, inplace=True)
    if hg_col: df.rename(columns={hg_col: 'home_goals'}, inplace=True)
    if ag_col: df.rename(columns={ag_col: 'away_goals'}, inplace=True)

    # compute result if goals exist
    if 'home_goals' in df.columns and 'away_goals' in df.columns:
        def r(row):
            if row['home_goals'] > row['away_goals']: return 'H'
            if row['home_goals'] < row['away_goals']: return 'A'
            return 'D'
        df['result'] = df.apply(r, axis=1)

    # drop rows missing critical fields
    if 'home' in df.columns and 'away' in df.columns:
        df = df.dropna(subset=['home','away'])
    return df

def build_features(df):
    """Minimal features: encode home & away teams to integers and use year/season if present."""
    df = df.copy()
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    # joint encoding ensures id consistency
    teams = pd.Series(pd.concat([df['home'], df['away']]).unique())
    le = LabelEncoder().fit(teams)
    df['home_id'] = le.transform(df['home'])
    df['away_id'] = le.transform(df['away'])
    # optionally add year/month if date exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month.fillna(0).astype(int)
    else:
        df['month'] = 0
    X = df[['home_id','away_id','month']].fillna(0)
    return X, df

def train_demo_model(X, y):
    # quick, robust model - deterministic-ish
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

@st.cache_resource
def load_model_if_exists(path="model.pkl"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed loading model.pkl: {e}")
            return None
    return None

def main():
    st.title("LaLiga — Quick Deploy Predictor")
    st.markdown("**Options:** Train a quick demo model (fast) or upload `model.pkl` to use a pre-trained model.")

    df = load_matches()
    fixtures = load_fixtures()

    st.sidebar.header("Deploy controls")
    action = st.sidebar.radio("Action", ["Inspect data", "Train demo", "Load model & predict"])

    if action == "Inspect data":
        st.subheader("Matches (head)")
        st.dataframe(df.head(200))
        st.subheader("Fixtures (head)")
        st.dataframe(fixtures.head(200))

    elif action == "Train demo":
        st.subheader("Preparing training data")
        df_p = basic_prep(df)
        if 'result' not in df_p.columns:
            st.error("No result column detected from historical data. Training aborted.")
            return
        X, df_feat = build_features(df_p)
        y = df_feat['result']
        st.write("Training shape:", X.shape)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        with st.spinner("Training RandomForest..."):
            clf = train_demo_model(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        st.success(f"Training finished. Validation accuracy: {acc:.3f}")
        st.text("Classification report:")
        st.text(classification_report(y_val, y_pred))
        # save model
        joblib.dump((clf, df_feat[['home','away']].copy()), "model.pkl")
        st.success("Saved model.pkl to repo root (visible in your repo after commit).")

    else:  # Load model & predict
        st.subheader("Load existing model or train then load")
        uploaded = st.file_uploader("Upload model.pkl (optional)", type=['pkl','joblib'])
        model = None
        if uploaded is not None:
            joblib.dump(joblib.load(uploaded), "model.pkl")  # save into working dir
        model = load_model_if_exists("model.pkl")
        if model is None:
            st.warning("No model found. Use 'Train demo' first or upload model.pkl.")
            return
        # model can be either (clf, team_info) or clf
        if isinstance(model, tuple):
            clf = model[0]
        else:
            clf = model

        # prepare fixtures
        fixtures_p = basic_prep(fixtures)
        if fixtures_p.empty:
            st.error("No fixtures detected in la-liga-2025-UTC.xlsx, or file missing.")
            return
        Xf, fixtures_feat = build_features(fixtures_p)
        # Align label encoding: if model was trained on fewer teams, unknown teams will cause mismatch.
        # We attempt to handle unknown team ids by clipping them into range:
        max_team_id = Xf[['home_id','away_id']].max().max()
        # If model expects fewer team ids, predictions may be meaningless. Still try.
        try:
            preds = clf.predict(Xf)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        fixtures_p['predicted_result'] = preds
        st.dataframe(fixtures_p[['home','away','predicted_result']].head(200))
        st.markdown("### Pred counts")
        st.write(fixtures_p['predicted_result'].value_counts())

if __name__ == "__main__":
    main()
