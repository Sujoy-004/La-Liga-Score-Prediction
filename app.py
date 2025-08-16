import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for other models
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------
# STREAMLIT APP CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="LaLiga Score Predictor", layout="wide")

st.title("⚽ LaLiga Match Outcome Predictor")
st.write("Predict match results using ML models retrained every time.")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Sujoy-004/La-Liga-Score-Prediction/main/matches_full.xlsx"
    df = pd.read_excel(url, engine="openpyxl")  # explicitly specify engine for Streamlit Cloud
    return df

df = load_data()
st.subheader("Sample of Dataset")
st.dataframe(df.head())

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
st.subheader("Preprocessing Data")

# Encode categorical columns
label_encoders = {}
df_processed = df.copy()

for col in df_processed.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

st.write("Categorical columns encoded successfully.")

# ---------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------
X = df_processed.drop("FTR", axis=1, errors="ignore")  # FTR = Full Time Result
y = df_processed["FTR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# MODEL SELECTION + TRAINING
# ---------------------------------------------------
st.subheader("Training Model...")

model_choice = st.selectbox("Choose a model:", ["RandomForest", "XGBoost", "LightGBM"])

if model_choice == "RandomForest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)
elif model_choice == "XGBoost":
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
else:
    model = lgb.LGBMClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ {model_choice} trained successfully! Accuracy: {acc:.2f}")

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------
st.subheader("Evaluation Metrics")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------------------------------------------
# PREDICTION INPUT
# ---------------------------------------------------
st.subheader("🔮 Predict New Match")

input_data = {}
for col in X.columns:
    val = st.number_input(f"Enter value for {col}:", value=float(X[col].mean()))
    input_data[col] = val

if st.button("Predict Match Result"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    # Decode result back
    result = label_encoders["FTR"].inverse_transform([pred])[0]
    st.success(f"Predicted Match Result: **{result}**")
