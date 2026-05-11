# ======================================================
# 🩺 CLEAN DIABETES AI DASHBOARD (HACKATHON READY UI)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
<style>

body {
    background-color: #f4f6f9;
}

/* KPI CARDS */
.kpi-card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

/* SIDEBAR TITLE */
.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* ALERT BOX */
.alert-box {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #ffcc00;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🩺 Diabetes AI Intelligence Dashboard")
st.caption("Clinical AI • Risk Prediction • Patient Monitoring • Insights Engine")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb", engine="pyxlsb")

    try:
        demo = pd.read_csv("cleaned_demographics.csv")
        df = df.merge(demo, on="patient_id", how="left")
    except FileNotFoundError:
        st.warning("Demographics file not found. Running without demographic features.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df

# =========================
# FEATURES
# =========================
df["hour"] = df["time"].dt.hour
df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["tir"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["hypo"] = (df["glucose"] < 70).astype(int)
df["hyper"] = (df["glucose"] > 180).astype(int)

df["risk_score"] = df["hyper"]*2 + df["hypo"]*3 + df["glucose_roc"].abs().fillna(0)

# =========================
# SIDEBAR (FIXED & ORGANIZED)
# =========================
st.sidebar.markdown("## 🧭 Control Panel")

patients = df["patient_id"].dropna().unique()

selected_patients = st.sidebar.multiselect(
    "👤 Select Patients",
    patients,
    default=patients[:3]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Dashboard Sections")
st.sidebar.info("Use tabs above for navigation")

dfv = df[df["patient_id"].isin(selected_patients)]

if dfv.empty:
    st.warning("No data available for selected patients")
    st.stop()

# =========================
# KPI SECTION (ENHANCED CARDS)
# =========================
st.markdown("## 📊 Patient Health Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Time in Range</h3>
    <h2>{dfv['tir'].mean()*100:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Hypoglycemia</h3>
    <h2>{dfv['hypo'].mean()*100:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Hyperglycemia</h3>
    <h2>{dfv['hyper'].mean()*100:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
    <h3>Avg Glucose</h3>
    <h2>{dfv['glucose'].mean():.1f}</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================
# TABS (CLEAN NAMING)
# =========================
tabs = st.tabs([
    "📊 Overview",
    "🍽️ Meals",
    "🏃 Activity",
    "🌙 Night Risk",
    "🧠 AI Prediction",
    "💊 Patient Score",
    "📌 Insights",
    "🚨 Risk Alerts",
    "🧬 Clustering"
])

# =========================
# OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Glucose Trend Overview")

    fig = px.line(dfv, x="time", y="glucose", color="patient_id")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution")
    fig = px.box(dfv, x="patient_id", y="glucose")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# MEALS
# =========================
with tabs[1]:
    st.subheader("Carb Impact Analysis")

    temp = dfv.copy()
    temp["future"] = temp.groupby("patient_id")["glucose"].shift(-24)

    fig = px.scatter(temp, x="carb_input", y=temp["future"] - temp["glucose"])
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
    night = dfv[dfv["hour"].between(0, 5)]

    fig = px.line(night, x="time", y="glucose")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# AI PREDICTION
# =========================
with tabs[4]:
    st.subheader("AI Hypoglycemia Prediction")

    model_df = dfv.dropna()

    features = ["glucose", "glucose_roc"]

    if len(model_df) > 100:

        X = model_df[features]
        y = model_df["hypo"]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        st.success("Model trained successfully")

# =========================
# RISK ALERTS (IMPORTANT VISIBILITY FIX)
# =========================
with tabs[7]:
    st.subheader("🚨 High Risk Patients")

    threshold = dfv["risk_score"].quantile(0.95)
    alerts = dfv[dfv["risk_score"] > threshold]

    st.markdown('<div class="alert-box">High-risk glucose patterns detected</div>', unsafe_allow_html=True)

    st.dataframe(alerts[["patient_id", "glucose", "risk_score"]])

# =========================
# CLUSTERING (SAFE)
# =========================
with tabs[8]:
    st.subheader("Patient Clusters")

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
