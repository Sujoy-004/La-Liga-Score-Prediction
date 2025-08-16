import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_training_data():
    return pd.read_excel("matches_full.xlsx")

@st.cache_data
def load_fixtures_data():
    return pd.read_excel("la-liga-2025-UTC.xlsx")

# -----------------------
# Preprocessing
# -----------------------
def preprocess_training(df):
    # Drop useless cols
    drop_cols = ["Unnamed: 0", "date", "time", "round", "day",
                 "attendance", "captain", "formation", "opp formation",
                 "referee", "match report", "notes"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode categorical
    cat_cols = ["venue", "opponent", "team", "comp"]
    df_encoded = df.copy()
    le_dict = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    # Target = result
    target_le = LabelEncoder()
    df_encoded["result"] = target_le.fit_transform(df["result"])

    return df_encoded, target_le, le_dict

def preprocess_fixtures(df, le_dict):
    df_encoded = df.copy()
    for col in ["Home Team", "Away Team", "Location"]:
        if col in df_encoded.columns:
            le = le_dict.get("opponent", None)
            if le:
                df_encoded[col] = df_encoded[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df_encoded[col] = df_encoded[col].astype("category").cat.codes
    return df_encoded

# -----------------------
# Model Training
# -----------------------
def train_model(df_encoded):
    X = df_encoded.drop("result", axis=1)
    y = df_encoded["result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, classification_report(y_test, y_pred)

# -----------------------
# Streamlit UI
# -----------------------
st.title("⚽ La Liga Match Outcome Predictor")

# Load & preprocess
train_df = load_training_data()
fixtures_df = load_fixtures_data()

st.subheader("Training on Historical Matches")
st.write(f"Training data shape: {train_df.shape}")

df_encoded, target_le, le_dict = preprocess_training(train_df)
model, acc, report = train_model(df_encoded)

st.success(f"Model trained! Accuracy: {acc:.2f}")
st.text("Classification Report:")
st.text(report)

# Predict fixtures
st.subheader("Predicting Upcoming Matches")
fixtures_encoded = preprocess_fixtures(fixtures_df, le_dict)

if not fixtures_encoded.empty:
    preds = model.predict(fixtures_encoded.select_dtypes(include=[np.number]))
    preds_labels = target_le.inverse_transform(preds)
    fixtures_df["Predicted Result"] = preds_labels
    st.dataframe(fixtures_df[["Date", "Home Team", "Away Team", "Predicted Result"]])
else:
    st.warning("No fixtures available for prediction.")
