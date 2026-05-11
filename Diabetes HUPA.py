# ======================================================
# 🩺 DIABETES AI INTELLIGENCE DASHBOARD (CLEAN VERSION)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes AI Intelligence System",
    layout="wide"
)

st.title("🩺 Diabetes AI Intelligence System")
st.caption("Clinical-grade analytics • Predictive AI • Risk intelligence • Patient clustering")

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb", engine="pyxlsb")
    demo = pd.read_csv("cleaned_demographics.csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "patient_id" in demo.columns:
        df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# =========================
# CLEANING
# =========================
df = df.dropna(subset=["time", "patient_id", "glucose"])
df = df.sort_values(["patient_id", "time"])

bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in df.columns else "bolus"

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date

# =========================
# FEATURES
# =========================
df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["rolling_std"] = df.groupby("patient_id")["glucose"].rolling(12).std().reset_index(0, drop=True)

df["tir"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["hypo"] = (df["glucose"] < 70).astype(int)
df["hyper"] = (df["glucose"] > 180).astype(int)

# =========================
# RISK ENGINE (IMPORTANT)
# =========================
df["risk_score"] = (
    df["hyper"] * 2 +
    df["hypo"] * 3 +
    df["glucose_roc"].abs().fillna(0)
)

# =========================
# SIDEBAR
# =========================
patients = df["patient_id"].dropna().unique()
selected = st.sidebar.multiselect("Patients", patients, default=patients[:5])

dfv = df[df["patient_id"].isin(selected)]

if dfv.empty:
    st.stop()

# =========================
# KPI ROW
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("TIR %", f"{dfv['tir'].mean()*100:.1f}")
c2.metric("Hypo %", f"{dfv['hypo'].mean()*100:.1f}")
c3.metric("Hyper %", f"{dfv['hyper'].mean()*100:.1f}")
c4.metric("Avg Glucose", f"{dfv['glucose'].mean():.1f}")

# =========================
# TABS
# =========================
tabs = st.tabs([
    "Overview",
    "Meals",
    "Activity",
    "Night Risk",
    "Predictive AI",
    "Prescriptive Score",
    "Insights",
    "Risk Engine",
    "Patient Clusters"
])

# ======================================================
# 1. OVERVIEW
# ======================================================
with tabs[0]:
    st.subheader("Glucose Trend")

    fig = px.line(dfv, x="time", y="glucose", color="patient_id")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution")
    fig = px.box(dfv, x="patient_id", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Correlation Heatmap")

    corr = dfv[["glucose", "glucose_roc", "steps", "heart_rate"]].corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 2. MEALS
# ======================================================
with tabs[1]:
    st.subheader("Carb vs Glucose Impact")

    temp = dfv.copy()
    temp["future_glucose"] = temp.groupby("patient_id")["glucose"].shift(-24)

    fig = px.scatter(
        temp,
        x="carb_input",
        y=temp["future_glucose"] - temp["glucose"],
        color="glucose",
        title="Meal Impact Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 3. ACTIVITY
# ======================================================
with tabs[2]:
    st.subheader("Activity Impact")

    fig = px.scatter(dfv, x="steps", y="glucose", color="patient_id")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 4. NIGHT RISK
# ======================================================
with tabs[3]:
    night = dfv[dfv["hour"].between(0, 5)]

    fig = px.line(night, x="time", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# 5. PREDICTIVE AI
# ======================================================
with tabs[4]:
    st.subheader("Hypoglycemia Prediction")

    model_df = dfv.dropna()

    features = ["glucose", "glucose_roc", "rolling_std"]

    model_df = model_df.dropna(subset=features)

    if len(model_df) > 100:

        X = model_df[features]
        y = model_df["hypo"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        st.metric("Accuracy", accuracy_score(y_test, pred))
        st.metric("ROC-AUC", roc_auc_score(y_test, prob))

# ======================================================
# 6. PRESCRIPTIVE SCORE
# ======================================================
with tabs[5]:
    score = dfv.groupby("patient_id").agg({
        "tir": "mean",
        "glucose": "std",
        "steps": "mean",
        "risk_score": "mean"
    }).reset_index()

    score["score"] = (
        score["tir"] * 40 +
        (1 - score["glucose"].rank(pct=True)) * 30 +
        score["steps"].rank(pct=True) * 30
    )

    st.dataframe(score)

# ======================================================
# 7. INSIGHTS
# ======================================================
with tabs[6]:
    st.markdown("""
    - High glucose variability strongly correlates with risk score  
    - Activity improves time-in-range  
    - Nighttime instability increases hypoglycemia risk  
    """)

# ======================================================
# 8. RISK ENGINE
# ======================================================
with tabs[7]:
    st.subheader("High Risk Alerts")

    threshold = dfv["risk_score"].quantile(0.95)
    alerts = dfv[dfv["risk_score"] > threshold]

    st.dataframe(alerts[["patient_id", "glucose", "risk_score"]])

# ======================================================
# 9. PATIENT CLUSTERS (SAFE PCA)
# ======================================================
with tabs[8]:
    st.subheader("Patient Clustering (PCA)")

    cluster = dfv.groupby("patient_id")[["glucose", "steps", "heart_rate"]].mean()

    cluster = cluster.dropna()

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
