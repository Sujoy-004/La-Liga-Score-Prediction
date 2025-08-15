# app.py - Fixed and production-ready Streamlit app for LaLiga predictions
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="LaLiga Score / Result Predictor", layout="wide")

# ------------------ Data loaders ------------------
@st.cache_data(ttl=3600)
def load_matches(path="matches_full.csv"):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_fixtures(path="la-liga-2025-UTC.xlsx"):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            st.error(f"Failed to read {path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# ------------------ Normalizers / Feature builders ------------------

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
    # try simple heuristics for common notations
    if s in ['h']:
        return True
    if s in ['a']:
        return False
    return None


def basic_prep(df):
    """Normalize a DataFrame to match-level rows with columns: home, away, home_goals, away_goals, result (if available).
    Handles both match-centric (Home Team / Away Team) and team-centric (team/opponent/venue/gf/ga) formats.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    cols = list(df.columns)

    # candidate name lists
    home_variants = ['home','home_team','home team','hometeam','home team name','home team name']
    away_variants = ['away','away_team','away team','awayteam','away team name']
    hg_variants = ['home_goals','fthg','homegoals','gf','gf_home','hg']
    ag_variants = ['away_goals','ftag','awaygoals','ga','ga_away','ag']
    team_variants = ['team','team_name','side','club']
    opponent_variants = ['opponent','opponent_name','opposing team']
    venue_variants = ['venue','location','home/away','home_away']
    result_variants = ['result','ftr','match_result']

    # find actual cols
    home_col = _find_column(df, home_variants)
    away_col = _find_column(df, away_variants)
    hg_col = _find_column(df, hg_variants)
    ag_col = _find_column(df, ag_variants)
    team_col = _find_column(df, team_variants)
    opponent_col = _find_column(df, opponent_variants)
    venue_col = _find_column(df, venue_variants)
    res_col = _find_column(df, result_variants)

    # CASE 1: match-centric file (has home & away)
    if home_col and away_col:
        df = df.rename(columns={home_col: 'home', away_col: 'away'})
        if hg_col:
            df = df.rename(columns={hg_col: 'home_goals'})
        if ag_col:
            df = df.rename(columns={ag_col: 'away_goals'})
        if res_col:
            df = df.rename(columns={res_col: 'result'})

    # CASE 2: team-centric (team/opponent/venue/gf/ga)
    elif team_col and opponent_col and venue_col and (hg_col or ag_col):
        # need both gf and ga detection (try different combos)
        gf_col = hg_col or _find_column(df, ['gf','goals_for'])
        ga_col = ag_col or _find_column(df, ['ga','goals_against'])
        if not gf_col or not ga_col:
            raise KeyError("Found team/opponent/venue but missing goals columns. Columns: " + str(cols))

        df = df.rename(columns={team_col: 'team', opponent_col: 'opponent', venue_col: 'venue', gf_col: 'gf', ga_col: 'ga'})

        homes, aways, hg_vals, ag_vals = [], [], [], []
        for _, row in df.iterrows():
            vflag = _is_home_text(row.get('venue'))
            if vflag is True:
                h = row['team']; a = row['opponent']; hg = row['gf']; ag = row['ga']
            elif vflag is False:
                h = row['opponent']; a = row['team']; hg = row['ga']; ag = row['gf']
            else:
                # ambiguous: best-effort assume 'team' is home
                h = row['team']; a = row['opponent']; hg = row['gf']; ag = row['ga']

            homes.append(h); aways.append(a)
            try:
                hg_vals.append(float(hg) if not pd.isna(hg) else np.nan)
            except Exception:
                hg_vals.append(np.nan)
            try:
                ag_vals.append(float(ag) if not pd.isna(ag) else np.nan)
            except Exception:
                ag_vals.append(np.nan)

        df['home'] = homes
        df['away'] = aways
        df['home_goals'] = hg_vals
        df['away_goals'] = ag_vals

        def res_from_goals(r):
            try:
                if pd.isna(r['home_goals']) or pd.isna(r['away_goals']):
                    return np.nan
                if float(r['home_goals']) > float(r['away_goals']):
                    return 'H'
                if float(r['home_goals']) < float(r['away_goals']):
                    return 'A'
                return 'D'
            except Exception:
                return np.nan

        df['result'] = df.apply(res_from_goals, axis=1)

    else:
        raise KeyError(f"Auto-detection failed. Available columns: {cols}")

    # final cleanup
    if 'home' not in df.columns or 'away' not in df.columns:
        raise KeyError("After normalization, 'home' and 'away' missing. Columns: " + str(list(df.columns)))

    df = df.dropna(subset=['home', 'away']).reset_index(drop=True)
    return df


def build_features(df):
    """Return X, df_with_ids, and fitted LabelEncoder for teams."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    df = df.copy()
    if 'home' not in df.columns or 'away' not in df.columns:
        raise KeyError("build_features expects 'home' and 'away' columns")

    teams = pd.Series(pd.concat([df['home'].astype(str), df['away'].astype(str)])).unique()
    teams = [str(t) for t in teams]
    le = LabelEncoder().fit(teams)

    df['home_id'] = le.transform(df['home'].astype(str))
    df['away_id'] = le.transform(df['away'].astype(str))

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month.fillna(0).astype(int)
    else:
        df['month'] = 0

    X = df[['home_id', 'away_id', 'month']].fillna(0).astype(int)
    return X, df, le


# ------------------ Training helper ------------------

def train_demo_model(X, y, df_feat=None):
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    if df_feat is not None and 'home' in df_feat.columns and 'away' in df_feat.columns:
        teams = pd.Series(pd.concat([df_feat['home'].astype(str), df_feat['away'].astype(str)])).unique()
        teams = [str(t) for t in teams]
        le = LabelEncoder().fit(teams)
        return clf, le, list(teams)
    return clf, None, None


@st.cache_resource
def load_model_if_exists(path="model.pkl"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed loading {path}: {e}")
            return None
    return None


# ------------------ App UI & logic ------------------

def predict_fixtures_with_model(clf, le, team_list, fixtures_df):
    """Return fixtures_df with predicted_result and predicted_prob columns added."""
    fixtures_p = basic_prep(fixtures_df)
    if fixtures_p.empty:
        return fixtures_p

    # month feature
    if 'date' in fixtures_p.columns:
        fixtures_p['date'] = pd.to_datetime(fixtures_p['date'], errors='coerce')
        fixtures_p['month'] = fixtures_p['date'].dt.month.fillna(0).astype(int)
    else:
        fixtures_p['month'] = 0

    # build mapping that preserves training ids
    base_map = {}
    if le is not None:
        base_map = {str(t): int(i) for i, t in enumerate(le.classes_)}
    elif team_list is not None:
        base_map = {str(t): int(i) for i, t in enumerate(team_list)}

    next_id = max(base_map.values()) + 1 if len(base_map) > 0 else 0

    def map_team(t):
        nonlocal next_id
        s = str(t)
        if s in base_map:
            return base_map[s]
        else:
            base_map[s] = next_id
            next_id += 1
            return base_map[s]

    fixtures_p['home_id'] = fixtures_p['home'].astype(str).apply(map_team)
    fixtures_p['away_id'] = fixtures_p['away'].astype(str).apply(map_team)

    Xf = fixtures_p[['home_id', 'away_id', 'month']].fillna(0).astype(int)

    try:
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(Xf)
            pred_idx = probs.argmax(axis=1)
            preds = clf.classes_[pred_idx]
            max_probs = probs.max(axis=1)
        else:
            preds = clf.predict(Xf)
            max_probs = [None] * len(preds)
    except Exception as e:
        # last-resort fallback: try re-fitting a small encoder union preserving training indices
        raise RuntimeError(f"Prediction failed: {e}")

    fixtures_p['predicted_result'] = preds
    fixtures_p['predicted_prob'] = max_probs
    return fixtures_p


def main():
    st.title("LaLiga — Quick Deploy Predictor")
    st.markdown("**Options:** Inspect data, Train a demo model (fast) or Load model & predict.")

    df = load_matches()
    fixtures = load_fixtures()

    st.sidebar.header("Deploy controls")
    action = st.sidebar.radio("Action", ["Inspect data", "Train demo", "Load model & predict"])

    if st.sidebar.checkbox("Show file column names (debug)"):
        st.subheader("Columns in matches_full.csv (if present)")
        st.write(list(df.columns))
        st.subheader("Columns in la-liga-2025-UTC.xlsx (if present)")
        st.write(list(fixtures.columns))
        st.markdown("---")

    if action == "Inspect data":
        st.subheader("Matches (head)")
        st.dataframe(df.head(200))
        st.subheader("Fixtures (head)")
        st.dataframe(fixtures.head(200))

    elif action == "Train demo":
        st.subheader("Preparing training data")
        try:
            df_p = basic_prep(df)
        except Exception as e:
            st.error(f"Auto-detection / normalization failed: {e}")
            st.stop()

        if 'result' not in df_p.columns:
            st.error("No result column detected from historical data. Training aborted.")
            st.stop()

        X, df_feat, le = build_features(df_p)
        y = df_feat['result']
        st.write("Training shape:", X.shape)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        with st.spinner("Training RandomForest..."):
            clf, le_trained, team_list = train_demo_model(X_train, y_train, df_feat)

        # Use the trained encoder (le_trained) if provided; otherwise fall back to le from build_features
        le_final = le_trained if le_trained is not None else le
        team_list_final = team_list if team_list is not None else (list(le_final.classes_) if le_final is not None else None)

        # validation
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        st.success(f"Training finished. Validation accuracy: {acc:.3f}")
        st.text("Classification report:")
        st.text(classification_report(y_val, y_pred))

        # save model blob
        model_blob = {'model': clf, 'le': le_final, 'team_list': team_list_final}
        joblib.dump(model_blob, "model.pkl")
        st.success("Saved model.pkl to working dir.")
        with open("model.pkl", "rb") as f:
            st.download_button("Download model.pkl", data=f.read(), file_name="model.pkl", mime="application/octet-stream")

        # immediate prediction on fixtures
        if fixtures is None or fixtures.empty:
            st.warning("No fixtures file found to predict.")
        else:
            try:
                fixtures_pred = predict_fixtures_with_model(clf, le_final, team_list_final, fixtures)
            except Exception as e:
                st.error(f"Predicting fixtures failed: {e}")
                st.stop()

            if fixtures_pred.empty:
                st.warning("Fixtures normalization produced no rows.")
            else:
                st.markdown("### Predicted fixtures (sample)")
                # choose display columns intelligently
                display_cols = [c for c in ['Match Number', 'Round Number', 'Date', 'Location', 'home', 'away', 'predicted_result', 'predicted_prob'] if c in fixtures_pred.columns]
                if not display_cols:
                    display_cols = ['home', 'away', 'predicted_result', 'predicted_prob']
                st.dataframe(fixtures_pred[display_cols].head(200))

                st.markdown("### Predicted result distribution")
                counts = fixtures_pred['predicted_result'].value_counts().reindex(['H','D','A']).fillna(0)
                st.bar_chart(counts)

                # show probability breakdown if available
                if 'predicted_prob' in fixtures_pred.columns and fixtures_pred['predicted_prob'].notna().any():
                    st.markdown("### Probabilities sample")
                    st.dataframe(fixtures_pred[['home','away','predicted_prob']].head(50))

                out_cols = [c for c in ['Match Number','Round Number','Date','home','away','predicted_result','predicted_prob'] if c in fixtures_pred.columns]
                if not out_cols:
                    out_cols = ['home','away','predicted_result','predicted_prob']
                out_csv = fixtures_pred[out_cols].to_csv(index=False)
                st.download_button("Download fixture predictions CSV", data=out_csv, file_name="la-liga-fixtures-predictions.csv", mime="text/csv")

    else:  # Load model & predict
        st.subheader("Load existing model or upload a model.pkl")
        uploaded = st.file_uploader("Upload model.pkl (optional)", type=['pkl','joblib'])
        model_blob = None
        if uploaded is not None:
            # save uploaded file into working dir
            with open("model_uploaded.pkl","wb") as f:
                f.write(uploaded.read())
            model_blob = load_model_if_exists("model_uploaded.pkl")
        else:
            model_blob = load_model_if_exists("model.pkl")

        if model_blob is None:
            st.warning("No model found. Use 'Train demo' to create a model or upload a model.pkl.")
            st.stop()

        # unpack model_blob
        clf = None; le = None; team_list = None
        if isinstance(model_blob, dict):
            clf = model_blob.get('model')
            le = model_blob.get('le')
            team_list = model_blob.get('team_list')
        elif isinstance(model_blob, (list, tuple)):
            # legacy tuple format: (clf, df_teams)
            clf = model_blob[0]
            try:
                df_teams = model_blob[1]
                if isinstance(df_teams, (pd.DataFrame, pd.Series)):
                    team_list = list(pd.Series(pd.concat([df_teams.get('home', pd.Series([])), df_teams.get('away', pd.Series([]))])).unique())
            except Exception:
                team_list = None
        else:
            clf = model_blob

        if clf is None:
            st.error("Loaded object does not contain a valid model.")
            st.stop()

        # predict fixtures
        try:
            fixtures_pred = predict_fixtures_with_model(clf, le, team_list, fixtures)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        if fixtures_pred.empty:
            st.warning("No fixtures to predict (normalization empty).")
        else:
            st.markdown("### Predicted fixtures (loaded model)")
            display_cols = [c for c in ['Match Number', 'Round Number', 'Date', 'Location', 'home', 'away', 'predicted_result', 'predicted_prob'] if c in fixtures_pred.columns]
            if not display_cols:
                display_cols = ['home','away','predicted_result','predicted_prob']
            st.dataframe(fixtures_pred[display_cols].head(200))

            st.markdown("### Predicted result distribution")
            counts = fixtures_pred['predicted_result'].value_counts().reindex(['H','D','A']).fillna(0)
            st.bar_chart(counts)

            if 'predicted_prob' in fixtures_pred.columns and fixtures_pred['predicted_prob'].notna().any():
                st.markdown("### Probabilities sample")
                st.dataframe(fixtures_pred[['home','away','predicted_prob']].head(50))

            out_cols = [c for c in ['Match Number','Round Number','Date','home','away','predicted_result','predicted_prob'] if c in fixtures_pred.columns]
            if not out_cols:
                out_cols = ['home','away','predicted_result','predicted_prob']
            out_csv = fixtures_pred[out_cols].to_csv(index=False)
            st.download_button("Download fixture predictions CSV", data=out_csv, file_name="la-liga-fixtures-predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
