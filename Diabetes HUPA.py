import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Diabetes Intelligence Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD DATA (FIXED FOR YOUR FILES)
# ======================================================
@st.cache_data
def load_data():

    # ✔ YOUR FILES
    demo = pd.read_csv("cleaned_demographics(1).csv")
    df = pd.read_excel(
        "cleaned_hupa_diabetes_recent(1).xlsb",
        engine="pyxlsb"
    )

    # merge if possible
    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    # datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df["hour"] = df["time"].dt.hour

    # ======================================================
    # SAFE FEATURE ENGINEERING (IMPORTANT FIX)
    # ======================================================

    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_rolling_std"] = df["glucose"].rolling(12).std().fillna(0)

    if "heart_rate" in df.columns:
        df["hr_clean"] = df["heart_rate"]

    if "steps" not in df.columns:
        df["steps"] = 0

    if "calories" not in df.columns:
        df["calories"] = 0

    return df


df = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 AI Diabetes Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Descriptive Analytics", "Predictive Analytics", "Prescriptive Analytics"]
)

patient_list = ["All Patients"] + list(df["patient_id"].unique())
selected_patient = st.sidebar.selectbox("Select Patient", patient_list)

if selected_patient != "All Patients":
    df = df[df["patient_id"] == selected_patient]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# ======================================================
# KPI METRICS
# ======================================================
avg_glucose = df["glucose"].mean()
max_glucose = df["glucose"].max()
min_glucose = df["glucose"].min()

tir = df["glucose"].between(70, 180).mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Avg Glucose", f"{avg_glucose:.1f}")
col2.metric("🔺 Max Glucose", f"{max_glucose:.1f}")
col3.metric("🔻 Min Glucose", f"{min_glucose:.1f}")
col4.metric("🎯 Time in Range", f"{tir:.1f}%")

# ======================================================
# OVERVIEW
# ======================================================
if menu == "Overview":

    st.subheader("📘 Dataset Overview")

    fig = px.line(
        df,
        x="time",
        y="glucose",
        color="patient_id",
        title="Glucose Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="glucose", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# DESCRIPTIVE ANALYTICS
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Analytics")

    fig = px.scatter(
        df,
        x="heart_rate",
        y="glucose",
        color="steps"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df,
        x="time",
        y="glucose_rolling_std",
        title="Glucose Variability"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# PREDICTIVE ANALYTICS
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction")

    df["risk_score"] = (
        df["glucose_roc"].fillna(0).abs() * 0.5 +
        df["glucose_rolling_std"].fillna(0) * 0.5
    )

    fig = px.line(df, x="time", y="risk_score")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PRESCRIPTIVE ANALYTICS
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decisions")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(
        df,
        x="time",
        y="glucose",
        color="risk_level"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# INSIGHTS
# ======================================================
st.markdown("---")
st.subheader("📌 Clinical Insights")

st.info(f"""
✔ Average glucose: {avg_glucose:.1f} mg/dL  
✔ Time in Range: {tir:.1f}%  

👉 Interpretation:
{'Stable glucose control' if tir > 70 else 'Unstable glucose pattern detected'}
""")
