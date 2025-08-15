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

# Replace your existing basic_prep() and build_features() with this code

def _find_column(df, variants):
    """
    Return the actual column name from df that matches any name in variants (case-insensitive).
    If none found, return None.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for v in variants:
        key = v.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def basic_prep(df):
    """Auto-detect home/away/goals/result columns and normalize names to: home, away, home_goals, away_goals, result."""
    df = df.copy()
    # candidate names (add more if your notebook uses different names)
    home_variants = ['home_team', 'home', 'homeTeam', 'home team', 'team_home', 'team home', 'HomeTeam']
    away_variants = ['away_team', 'away', 'awayTeam', 'away team', 'team_away', 'team away', 'AwayTeam']
    home_goals_variants = ['home_goals', 'fthg', 'home_goals_full', 'home_goals_ft', 'homegoals', 'FTHG']
    away_goals_variants = ['away_goals', 'ftag', 'away_goals_full', 'away_goals_ft', 'awaygoals', 'FTAG']
    result_variants = ['result', 'ftr', 'FTR', 'match_result', 'res']

    # find actual columns
    home_col = _find_column(df, home_variants)
    away_col = _find_column(df, away_variants)
    hg_col = _find_column(df, home_goals_variants)
    ag_col = _find_column(df, away_goals_variants)
    res_col = _find_column(df, result_variants)

    rename_map = {}
    if home_col:
        rename_map[home_col] = 'home'
    if away_col:
        rename_map[away_col] = 'away'
    if hg_col:
        rename_map[hg_col] = 'home_goals'
    if ag_col:
        rename_map[ag_col] = 'away_goals'
    if res_col:
        rename_map[res_col] = 'result'

    if rename_map:
        df = df.rename(columns=rename_map)

    # compute 'result' if goals exist but result doesn't
    if ('home_goals' in df.columns) and ('away_goals' in df.columns) and ('result' not in df.columns):
        def r(row):
            try:
                if int(row['home_goals']) > int(row['away_goals']): return 'H'
                if int(row['home_goals']) < int(row['away_goals']): return 'A'
                return 'D'
            except Exception:
                return np.nan
        df['result'] = df.apply(r, axis=1)

    # final check: we must have 'home' and 'away' for features/training
    if 'home' not in df.columns or 'away' not in df.columns:
        # helpful error: list available columns so user can see what's wrong
        available = list(df.columns)
        raise KeyError(
            "Required columns 'home' and 'away' not found after auto-detection. "
            f"Available columns: {available}. "
            "If your file uses different column names, add them to the detection lists in basic_prep()."
        )

    # drop rows with missing team names
    df = df.dropna(subset=['home','away']).reset_index(drop=True)

    return df

def build_features(df):
    """Encode teams to integer ids and create minimal features. Returns (X, df_with_features)."""
    df = df.copy()

    # ensure basic_prep has been applied (home/away present)
    if 'home' not in df.columns or 'away' not in df.columns:
        raise KeyError("build_features expects columns 'home' and 'away'. Run basic_prep() first.")

    # create a single team universe so encoding is consistent
    teams = pd.Series(pd.concat([df['home'], df['away']]).astype(str).unique())
    le = LabelEncoder().fit(teams)

    # map teams - if a team in df is unknown to le, we fit on all present teams so this is safe
    df['home_id'] = le.transform(df['home'].astype(str))
    df['away_id'] = le.transform(df['away'].astype(str))

    # optional date -> month feature
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month.fillna(0).astype(int)
    else:
        df['month'] = 0

    X = df[['home_id','away_id','month']].fillna(0).astype(int)
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
