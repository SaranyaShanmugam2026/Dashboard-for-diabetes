import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Clinical AI Diabetes System",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    demo = pd.read_csv("cleaned_demographics.csv")

    diabetes = pd.read_excel(
        "cleaned_hupa_diabetes_recent.xlsb",
        engine="pyxlsb"
    )

    if "patient_id" in demo.columns and "patient_id" in diabetes.columns:
        df = diabetes.merge(demo, on="patient_id", how="left")
    else:
        df = diabetes.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

    # Basic feature engineering
    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_std"] = df["glucose"].rolling(12).std().fillna(0)

    return df

df = load_data()

# =========================================================
# SIDEBAR - INTERACTIVE CONTROLS
# =========================================================
st.sidebar.title("🧠 Clinical AI Controls")

patient_list = ["All"] + list(df["patient_id"].unique()) if "patient_id" in df.columns else ["All"]

patient = st.sidebar.selectbox("Select Patient", patient_list)

hypo_threshold = st.sidebar.slider("Hypoglycemia Threshold", 50, 80, 70)
hyper_threshold = st.sidebar.slider("Hyperglycemia Threshold", 150, 250, 180)
time_window = st.sidebar.slider("Time Window (hours)", 6, 72, 24)

compare_mode = st.sidebar.multiselect(
    "Compare Patients",
    df["patient_id"].unique() if "patient_id" in df.columns else []
)

insulin_sim = st.sidebar.slider("Simulated Insulin Effect (%)", 0, 50, 0)

# =========================================================
# FILTER DATA
# =========================================================
if patient != "All":
    df = df[df["patient_id"] == patient]

if "time" in df.columns:
    max_time = df["time"].max()
    min_time = max_time - pd.Timedelta(hours=time_window)
    df = df[df["time"] >= min_time]

# =========================================================
# TITLE
# =========================================================
st.title("🩺 Clinical AI Diabetes Intelligence System")

# =========================================================
# KPI SECTION (DYNAMIC)
# =========================================================
col1, col2, col3, col4 = st.columns(4)

if "glucose" in df.columns:

    col1.metric("Avg Glucose", round(df["glucose"].mean(), 2))
    col2.metric("Max Glucose", round(df["glucose"].max(), 2))
    col3.metric("Min Glucose", round(df["glucose"].min(), 2))

    tir = df["glucose"].between(hypo_threshold, hyper_threshold).mean() * 100
    col4.metric("Time in Range %", round(tir, 2))

# =========================================================
# RISK SCORE (DYNAMIC AI LOGIC)
# =========================================================
if "glucose" in df.columns:

    df["risk_score"] = (
        (df["glucose"] > hyper_threshold).astype(int) * 0.6 +
        (df["glucose"] < hypo_threshold).astype(int) * 0.6 +
        df["glucose_std"]
    )

    # =====================================================
    # SIMULATION (WHAT-IF INSULIN EFFECT)
    # =====================================================
    df["simulated_glucose"] = df["glucose"] - (df.get("insulin", 0) * insulin_sim * 0.1)

# =========================================================
# MAIN CHARTS (INTERACTIVE)
# =========================================================
st.subheader("📈 Glucose Trend (Interactive)")

if "time" in df.columns and "glucose" in df.columns:

    fig = px.line(
        df,
        x="time",
        y="glucose",
        color="patient_id" if "patient_id" in df.columns else None,
        title="Real-Time Glucose Trend"
    )

    fig.add_hline(y=hypo_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=hyper_threshold, line_dash="dash", line_color="orange")

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RISK VISUALIZATION
# =========================================================
st.subheader("🚨 Risk Analysis Engine")

if "glucose" in df.columns:

    fig2 = px.scatter(
        df,
        x="time",
        y="glucose",
        color="risk_score",
        title="Glucose Risk Heatmap"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# GLUCOSE VARIABILITY
# =========================================================
st.subheader("📊 Glucose Variability")

if "glucose_std" in df.columns:

    fig3 = px.line(
        df,
        y="glucose_std",
        title="Rolling Glucose Variability"
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# WHAT-IF SIMULATION
# =========================================================
st.subheader("🧪 What-If Simulation")

if "insulin" in df.columns:

    fig4 = px.line(
        df,
        x="time",
        y=["glucose", "simulated_glucose"],
        title="Actual vs Simulated Glucose Response"
    )

    st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# PATIENT COMPARISON
# =========================================================
if compare_mode:

    st.subheader("👥 Patient Comparison")

    compare_df = df[df["patient_id"].isin(compare_mode)]

    fig5 = px.line(
        compare_df,
        x="time",
        y="glucose",
        color="patient_id",
        title="Patient Comparison - Glucose Trends"
    )

    st.plotly_chart(fig5, use_container_width=True)

# =========================================================
# AI INSIGHTS PANEL
# =========================================================
st.subheader("🧠 AI Insights Engine")

if "glucose" in df.columns:

    avg_glucose = df["glucose"].mean()

    if avg_glucose > hyper_threshold:
        st.error("High glucose instability detected → risk of hyperglycemia")
    elif avg_glucose < hypo_threshold:
        st.warning("Low glucose trend → hypoglycemia risk")
    else:
        st.success("Metabolic pattern stable within target range")

    if df["risk_score"].mean() > 1:
        st.info("Elevated variability detected → monitor insulin response")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Clinical AI System | Interactive Diabetes Intelligence Dashboard")
