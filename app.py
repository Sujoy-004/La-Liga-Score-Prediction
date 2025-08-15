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

# ---- Replace basic_prep() and build_features() with the following ----
import numpy as np
from sklearn.preprocessing import LabelEncoder

def _find_column(df, variants):
    lower_map = {c.lower(): c for c in df.columns}
    for v in variants:
        key = v.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def _is_home_text(txt):
    if pd.isna(txt):
        return None
    s = str(txt).strip().lower()
    if 'home' in s:
        return True
    if 'away' in s:
        return False
    # neutral / unknown
    return None

def basic_prep(df):
    """
    Normalizes a dataframe to match-level rows with these columns:
    ['home','away','home_goals','away_goals','result'] when possible.
    Handles:
      - Match-centric: columns like 'home','away','home_goals','away_goals' or 'Home Team'/'Away Team'
      - Team-centric: columns like 'team','opponent','venue','gf','ga' (your matches_full.csv)
    Raises KeyError with available columns if it cannot auto-detect.
    """
    df = df.copy()
    cols = list(df.columns)

    # candidate names
    home_variants = ['home','home_team','home team','hometeam','Home Team','HomeTeam']
    away_variants = ['away','away_team','away team','awayteam','Away Team','AwayTeam']
    hg_variants = ['home_goals','fthg','home_goals_full','homegoals','hg']
    ag_variants = ['away_goals','ftag','away_goals_full','awaygoals','ag']
    # team-centric candidates (your CSV)
    team_variants = ['team','team_name','side','club']
    opponent_variants = ['opponent','opposing team','opponent_name','opponent team']
    venue_variants = ['venue','location','home/away','home_away']  # 'Home' / 'Away' values
    gf_variants = ['gf','goals_for','for']
    ga_variants = ['ga','goals_against','against']
    result_variants = ['result','ftr','FTR','match_result']

    # find columns (actual names)
    home_col = _find_column(df, home_variants)
    away_col = _find_column(df, away_variants)
    hg_col = _find_column(df, hg_variants)
    ag_col = _find_column(df, ag_variants)
    team_col = _find_column(df, team_variants)
    opponent_col = _find_column(df, opponent_variants)
    venue_col = _find_column(df, venue_variants)
    gf_col = _find_column(df, gf_variants)
    ga_col = _find_column(df, ga_variants)
    res_col = _find_column(df, result_variants)

    # CASE A: already match-centric (home/away present) OR Home Team / Away Team from Excel
    if home_col and away_col:
        df = df.rename(columns={home_col: 'home', away_col: 'away'})
        if hg_col:
            df = df.rename(columns={hg_col: 'home_goals'})
        if ag_col:
            df = df.rename(columns={ag_col: 'away_goals'})
        if res_col:
            df = df.rename(columns={res_col: 'result'})

    # CASE B: team-centric (your matches_full.csv): 'team', 'opponent', 'venue' and 'gf','ga'
    elif team_col and opponent_col and venue_col and (gf_col or ga_col):
        # ensure gf/ga present - use gf/ga names we found
        if not gf_col or not ga_col:
            raise KeyError("Found team/opponent/venue but missing gf/ga columns. Available columns: " + str(cols))

        # standardize column names
        df = df.rename(columns={
            team_col: 'team',
            opponent_col: 'opponent',
            venue_col: 'venue',
            gf_col: 'gf',
            ga_col: 'ga'
        })

        # create home/away and goals per match
        homes = []
        aways = []
        home_goals = []
        away_goals = []
        for i, row in df.iterrows():
            venue_flag = _is_home_text(row.get('venue'))
            # if venue text explicitly indicates Home -> team is home
            if venue_flag is True:
                home = row['team']
                away = row['opponent']
                hg = row['gf']
                ag = row['ga']
            elif venue_flag is False:
                # team is away
                home = row['opponent']
                away = row['team']
                # gf is goals for 'team' (away), so swap
                hg = row['ga']
                ag = row['gf']
            else:
                # unknown venue: assume 'team' is home (best-effort) but mark it
                home = row['team']
                away = row['opponent']
                hg = row['gf']
                ag = row['ga']

            homes.append(home)
            aways.append(away)
            # ensure numeric
            try:
                home_goals.append(float(hg) if not pd.isna(hg) else np.nan)
            except Exception:
                home_goals.append(np.nan)
            try:
                away_goals.append(float(ag) if not pd.isna(ag) else np.nan)
            except Exception:
                away_goals.append(np.nan)

        df['home'] = homes
        df['away'] = aways
        df['home_goals'] = home_goals
        df['away_goals'] = away_goals

        # compute 'result' from goals when possible
        def res_from_goals(r):
            try:
                if pd.isna(r['home_goals']) or pd.isna(r['away_goals']):
                    return np.nan
                if float(r['home_goals']) > float(r['away_goals']): return 'H'
                if float(r['home_goals']) < float(r['away_goals']): return 'A'
                return 'D'
            except Exception:
                return np.nan
        df['result'] = df.apply(res_from_goals, axis=1)

    else:
        # failed to auto-detect either format
        raise KeyError(
            "Auto-detection failed. Required match-level columns not found. "
            f"Available columns: {cols}. "
            "This app expects either match-centric columns like 'Home Team'/'Away Team' or team-centric 'team','opponent','venue','gf','ga'."
        )

    # final cleanup: ensure 'home' and 'away' exist and drop rows missing them
    if 'home' not in df.columns or 'away' not in df.columns:
        raise KeyError("After normalization, 'home' and 'away' are still missing. Available cols: " + str(list(df.columns)))

    df = df.dropna(subset=['home', 'away']).reset_index(drop=True)
    return df

def build_features(df):
    """
    Build minimal features and label-encode teams.
    Returns X (DataFrame of features) and df (dataframe with added ids).
    """
    df = df.copy()
    if 'home' not in df.columns or 'away' not in df.columns:
        raise KeyError("build_features expects 'home' and 'away' columns. Run basic_prep() first.")

    # unify teams universe
    teams = pd.Series(pd.concat([df['home'].astype(str), df['away'].astype(str)]).unique())
    le = LabelEncoder().fit(teams)

    # map team names -> ids
    df['home_id'] = le.transform(df['home'].astype(str))
    df['away_id'] = le.transform(df['away'].astype(str))

    # optional date -> month
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
