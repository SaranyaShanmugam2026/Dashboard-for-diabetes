# ======================================================
# 🩺 DIABETES AI INTELLIGENCE DASHBOARD (STABLE VERSION)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# UI STYLE
# =========================
st.markdown("""
<style>
body {background-color:#f5f7fb;}

.kpi {
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes AI Intelligence Dashboard")
st.caption("Clinical AI • Risk Prediction • Monitoring • Insights Engine")

# =========================
# DATA LOADING (SAFE)
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb", engine="pyxlsb")

    try:
        demo = pd.read_csv("cleaned_demographics.csv")
        df = df.merge(demo, on="patient_id", how="left")
    except:
        st.warning("⚠ Demographics file not found (running without it)")

    return df

df = load_data()

# =========================
# SAFE CLEANING (FIX FOR YOUR ERROR)
# =========================

if "time" not in df.columns:
    st.error("❌ Missing 'time' column in dataset")
    st.stop()

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time", "patient_id", "glucose"])

df = df.sort_values(["patient_id", "time"])

# SAFE TIME FEATURES
df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date

# =========================
# FEATURE ENGINEERING
# =========================
df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["tir"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["hypo"] = (df["glucose"] < 70).astype(int)
df["hyper"] = (df["glucose"] > 180).astype(int)

df["risk_score"] = (
    df["hyper"] * 2 +
    df["hypo"] * 3 +
    df["glucose_roc"].abs().fillna(0)
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🧭 Controls")

patients = df["patient_id"].dropna().unique()

selected_patients = st.sidebar.multiselect(
    "Select Patients",
    patients,
    default=patients[:3]
)

dfv = df[df["patient_id"].isin(selected_patients)]

if dfv.empty:
    st.warning("No data for selected patients")
    st.stop()

# =========================
# KPI SECTION
# =========================
st.subheader("📊 Key Health Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='kpi'><h3>TIR</h3><h2>{dfv['tir'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><h3>Hypo</h3><h2>{dfv['hypo'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><h3>Hyper</h3><h2>{dfv['hyper'].mean()*100:.1f}%</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><h3>Avg Glucose</h3><h2>{dfv['glucose'].mean():.1f}</h2></div>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tabs = st.tabs([
    "📊 Overview",
    "🍽 Meals",
    "🏃 Activity",
    "🌙 Night Risk",
    "🧠 AI Model",
    "💊 Score",
    "📌 Insights",
    "🚨 Risk",
    "🧬 Clusters"
])

# =========================
# OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Glucose Trend")

    fig = px.line(dfv, x="time", y="glucose", color="patient_id")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution")
    fig = px.box(dfv, x="patient_id", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# MEALS
# =========================
with tabs[1]:
    st.subheader("Carb Impact")

    temp = dfv.copy()
    temp["future_glucose"] = temp.groupby("patient_id")["glucose"].shift(-24)

    fig = px.scatter(
        temp,
        x="carb_input",
        y=temp["future_glucose"] - temp["glucose"],
        color="glucose"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# ACTIVITY
# =========================
with tabs[2]:
    st.subheader("Activity vs Glucose")

    fig = px.scatter(dfv, x="steps", y="glucose", color="patient_id")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# NIGHT RISK
# =========================
with tabs[3]:
    st.subheader("Night Glucose Pattern")

    night = dfv[dfv["hour"].between(0, 5)]

    fig = px.line(night, x="time", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# AI MODEL
# =========================
with tabs[4]:
    st.subheader("Hypoglycemia Prediction Model")

    model_df = dfv.dropna()

    features = ["glucose", "glucose_roc"]

    if len(model_df) > 100:

        X = model_df[features]
        y = model_df["hypo"]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        st.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")

# =========================
# SCORE
# =========================
with tabs[5]:
    st.subheader("Patient Score")

    score = dfv.groupby("patient_id").agg({
        "tir": "mean",
        "glucose": "std",
        "steps": "mean",
        "risk_score": "mean"
    }).reset_index()

    st.dataframe(score)

# =========================
# INSIGHTS
# =========================
with tabs[6]:
    st.markdown("""
    ✔ Higher activity improves glucose stability  
    ✔ Nighttime instability increases risk  
    ✔ Carbohydrates strongly affect post-meal spikes  
    """)

# =========================
# RISK
# =========================
with tabs[7]:
    st.subheader("High Risk Alerts")

    threshold = dfv["risk_score"].quantile(0.95)
    alerts = dfv[dfv["risk_score"] > threshold]

    st.dataframe(alerts[["patient_id", "glucose", "risk_score"]])

# =========================
# CLUSTERS (SAFE PCA)
# =========================
with tabs[8]:
    st.subheader("Patient Clustering")

    cluster = dfv.groupby("patient_id")[["glucose", "steps"]].mean().dropna()

    if len(cluster) < 2:
        st.warning("Need more patients for clustering")
    else:
        scaled = StandardScaler().fit_transform(cluster)

        pca = PCA(n_components=2)
        comp = pca.fit_transform(scaled)

        cluster["PC1"] = comp[:, 0]
        cluster["PC2"] = comp[:, 1]

        fig = px.scatter(cluster, x="PC1", y="PC2", text=cluster.index)
        st.plotly_chart(fig, use_container_width=True)
