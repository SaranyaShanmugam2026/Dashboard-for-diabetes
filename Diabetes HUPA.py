# app.py
# ===========================
# GLUCOAI - ADVANCED VERSION
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ===========================
# CONFIG
# ===========================

st.set_page_config(
    page_title="GlucoAI Advanced Platform",
    page_icon="🩺",
    layout="wide"
)

# ===========================
# NAVIGATION MENU
# ===========================

menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Home",
        "🧹 Data Cleaning",
        "📊 Exploration",
        "🍽️ Meals & Insulin",
        "🏃 Activity",
        "🌙 Night Risk",
        "🤖 AI Prediction",
        "💊 Prescriptive AI",
        "📥 Export Data"
    ]
)

# ===========================
# LOAD DATA
# ===========================

@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb")
    demo = pd.read_csv("cleaned_demographics.csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# ===========================
# BASIC CLEANING PIPELINE (LIVE VIEW)
# ===========================

def clean_data(df):

    df = df.dropna(subset=["glucose", "time", "patient_id"])
    df = df.sort_values(["patient_id", "time"])

    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df["heart_rate"] = pd.to_numeric(df.get("heart_rate", 0), errors="coerce")

    df["glucose"] = df.groupby("patient_id")["glucose"].transform(
        lambda x: x.interpolate(limit=2)
    )

    df["glucose"] = df["glucose"].clip(40, 500)

    return df

df = clean_data(df)

# ===========================
# FEATURE ENGINEERING
# ===========================

df["hour"] = df["time"].dt.hour
df["is_night"] = df["hour"].between(0, 6).astype(int)

df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["rolling_mean"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12).mean()
    .reset_index(level=0, drop=True)
)

df["is_in_range"] = df["glucose"].between(70, 180).astype(int)

# ===========================
# HOME
# ===========================

if menu == "🏠 Home":

    st.title("🩺 GlucoAI Advanced Clinical Intelligence Platform")

    st.markdown("""
    ### 🚀 System Overview
    - CGM + Insulin + Activity + Sleep + Demographics
    - Real-time glucose intelligence
    - Predictive + Prescriptive AI engine
    - Clinical decision support system
    """)

    st.success("Ready for clinical AI decision-making 🚀")

# ===========================
# DATA CLEANING PAGE (IMPORTANT ADDITION)
# ===========================

elif menu == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning Pipeline Dashboard")

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    st.write("### Duplicate Check")
    st.write(df.duplicated().sum())

    st.write("### Glucose Distribution After Cleaning")
    fig = px.histogram(df, x="glucose", nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Outlier View")
    fig = px.box(df, y="glucose")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Data is cleaned and ready for AI modeling")

# ===========================
# EXPLORATION
# ===========================

elif menu == "📊 Exploration":

    st.title("📊 Clinical Data Exploration")

    patient = st.selectbox("Select Patient", df["patient_id"].unique())

    temp = df[df["patient_id"] == patient]

    fig = px.line(temp, x="time", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Avg Glucose", round(temp["glucose"].mean(), 2))

    with col2:
        st.metric("TIR", round(temp["is_in_range"].mean()*100, 2))

# ===========================
# MEALS
# ===========================

elif menu == "🍽️ Meals & Insulin":

    st.title("🍽️ Meal Impact Analysis")

    df_meal = df[df["carb_input"] > 0].copy()

    df_meal["spike"] = df_meal.groupby("patient_id")["glucose"].shift(-12) - df_meal["glucose"]

    fig = px.scatter(df_meal, x="carb_input", y="spike")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# ACTIVITY
# ===========================

elif menu == "🏃 Activity":

    st.title("🏃 Activity vs Glucose Stability")

    fig = px.scatter(df, x="steps", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# NIGHT RISK
# ===========================

elif menu == "🌙 Night Risk":

    st.title("🌙 Nocturnal Hypoglycemia Risk")

    night = df[df["is_night"] == 1]

    fig = px.histogram(night, x="glucose", color="is_in_range")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# AI MODEL
# ===========================

elif menu == "🤖 AI Prediction":

    st.title("🤖 Hypoglycemia Prediction Model")

    model_df = df.copy()

    model_df["target"] = (model_df["glucose"].shift(-6) < 70).astype(int)

    features = ["glucose", "glucose_roc", "hour"]

    model_df = model_df.dropna()

    X = model_df[features]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    st.metric("Accuracy", accuracy_score(y_test, pred))
    st.metric("ROC-AUC", roc_auc_score(y_test, prob))

# ===========================
# PRESCRIPTIVE AI
# ===========================

elif menu == "💊 Prescriptive AI":

    st.title("💊 Clinical Recommendation Engine")

    df["risk_score"] = (
        df["glucose"].rolling(12).mean() > 180
    ).astype(int)

    st.dataframe(df.groupby("patient_id")["risk_score"].mean())

# ===========================
# EXPORT
# ===========================

elif menu == "📥 Export Data":

    st.title("📥 Export Clean Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Clean Dataset",
        csv,
        "glucoai_clean.csv",
        "text/csv"
    )
