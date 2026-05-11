# app.py
# 🩺 Diabetes AI Intelligence Dashboard (Hackathon Enhanced Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes AI Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.5rem;}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes AI Intelligence Dashboard")
st.caption("Predictive • Prescriptive • Explainable Clinical Analytics")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent (1).xlsb", engine="pyxlsb")
    demo = pd.read_csv("cleaned_demographics (1).csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# =========================
# CLEANING + FEATURES
# =========================
df = df.dropna(subset=["time", "patient_id", "glucose"])
df = df.sort_values(["patient_id", "time"])

bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in df.columns else "bolus"

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date
df["is_night"] = df["hour"].between(0, 5).astype(int)

df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["glucose_std"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12).std().reset_index(level=0, drop=True)
)

df["glucose_mean"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12).mean().reset_index(level=0, drop=True)
)

df["tir_flag"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["hypo_flag"] = (df["glucose"] < 70).astype(int)
df["hyper_flag"] = (df["glucose"] > 180).astype(int)

# =========================
# RISK SCORE (GLOBAL)
# =========================
df["risk_score"] = (
    df["hyper_flag"] * 2 +
    df["hypo_flag"] * 3 +
    (df["glucose_roc"].abs() > 30).astype(int)
)

# =========================
# SIDEBAR
# =========================
patients = sorted(df["patient_id"].unique())
selected_patients = st.sidebar.multiselect(
    "Select Patients",
    patients,
    default=patients[:5]
)

df_view = df[df["patient_id"].isin(selected_patients)]

if df_view.empty:
    st.warning("Select patients to continue")
    st.stop()

# =========================
# KPIs
# =========================
tir = df_view["tir_flag"].mean() * 100
hypo = df_view["hypo_flag"].mean() * 100
hyper = df_view["hyper_flag"].mean() * 100

c1, c2, c3 = st.columns(3)
c1.metric("TIR", f"{tir:.1f}%")
c2.metric("Hypoglycemia", f"{hypo:.1f}%")
c3.metric("Hyperglycemia", f"{hyper:.1f}%")

# =========================
# TABS
# =========================
tabs = st.tabs([
    "📊 Overview",
    "🚨 Risk Stratification",
    "🍽 Meal Intelligence",
    "🏃 Activity Intelligence",
    "🌙 Night Risk",
    "🧠 Predictive AI",
    "💊 Prescriptive Score",
    "📡 Real-Time Monitor",
    "📌 Insights"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tabs[0]:
    fig = px.line(df_view, x="time", y="glucose", color="patient_id")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=180, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: RISK
# =========================
with tabs[1]:
    risk = df_view.groupby("patient_id")["risk_score"].mean().reset_index()

    fig = px.bar(
        risk,
        x="patient_id",
        y="risk_score",
        color="risk_score",
        title="Patient Risk Stratification"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: MEAL
# =========================
with tabs[2]:
    dfm = df_view.copy()
    dfm["glucose_next"] = dfm.groupby("patient_id")["glucose"].shift(-24)
    dfm["spike"] = dfm["glucose_next"] - dfm["glucose"]

    meal = dfm[dfm["carb_input"] > 0].dropna()

    fig = px.scatter(
        meal.sample(min(5000, len(meal))),
        x="carb_input",
        y="spike",
        color="glucose",
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4: ACTIVITY
# =========================
with tabs[3]:
    df_act = df_view.copy()
    df_act["activity"] = pd.cut(df_act["steps"],
                                bins=[-1, 0, 100, 500, 100000],
                                labels=["None", "Low", "Med", "High"])

    summary = df_act.groupby("activity")["glucose_std"].mean().reset_index()

    fig = px.bar(summary, x="activity", y="glucose_std")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 5: NIGHT
# =========================
with tabs[4]:
    night = df_view[df_view["is_night"] == 1]

    fig = px.scatter(
        night,
        x="basal_rate",
        y="glucose",
        trendline="lowess"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 6: PREDICTIVE AI
# =========================
with tabs[5]:

    model_df = df_view.copy()
    model_df["target"] = (model_df["glucose"] > 180).astype(int)

    features = ["glucose", "glucose_roc", "steps", "heart_rate", "hour"]

    model_df = model_df.dropna()

    X = model_df[features]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier(n_estimators=80)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    st.metric("Accuracy", accuracy_score(y_test, pred))
    st.metric("AUC", roc_auc_score(y_test, prob))

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    st.session_state["importance"] = importance

    fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig)

# =========================
# TAB 7: PRESCRIPTIVE
# =========================
with tabs[6]:
    score = df_view.groupby("patient_id").agg({
        "tir_flag": "mean",
        "glucose": "std",
        "hypo_flag": "mean"
    }).reset_index()

    score["final_score"] = (
        score["tir_flag"] * 50 +
        (1 - score["hypo_flag"]) * 30 +
        (1 - score["glucose"].rank(pct=True)) * 20
    )

    st.dataframe(score)

# =========================
# TAB 8: REAL TIME
# =========================
with tabs[7]:
    latest = df_view.groupby("patient_id").tail(1)

    latest["status"] = np.where(
        latest["glucose"] < 70, "LOW",
        np.where(latest["glucose"] > 180, "HIGH", "NORMAL")
    )

    fig = px.scatter(latest, x="patient_id", y="glucose", color="status")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 9: INSIGHTS
# =========================
with tabs[8]:

    st.subheader("🧠 AI Clinical Summary")

    st.info(f"""
    • Avg glucose: {df_view['glucose'].mean():.1f}  
    • Variability: {df_view['glucose'].std():.1f}  
    • High-risk episodes: {df_view['hyper_flag'].mean()*100:.1f}%  
    • Hypoglycemia risk: {df_view['hypo_flag'].mean()*100:.1f}%  
    """)

    if "importance" in st.session_state:
        st.subheader("Explainable AI Drivers")

        fig = px.pie(
            st.session_state["importance"],
            names="Feature",
            values="Importance"
        )
        st.plotly_chart(fig)

st.success("🚀 Hackathon-Ready AI Diabetes Intelligence System Loaded")
