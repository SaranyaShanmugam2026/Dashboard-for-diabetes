import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Diabetes Clinical Intelligence System",
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
        df["hour"] = df["time"].dt.hour

    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_std"] = df["glucose"].rolling(12).std().fillna(0)

        # Time in Range
        df["tir"] = df["glucose"].between(70, 180).astype(int)

        # Risk score (simple prototype AI)
        df["risk_score"] = (
            abs(df["glucose_roc"]) * 0.4 +
            df["glucose_std"] * 0.4 +
            (df["glucose"] > 180).astype(int) * 0.2
        )

    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🧠 Clinical AI System")

menu = st.sidebar.radio(
    "Modules",
    [
        "Patient Overview",
        "Glucose Dynamics",
        "Risk & Events",
        "Insulin Intelligence",
        "Meal & Activity",
        "Predictive AI"
    ]
)

if "patient_id" in df.columns:
    patient = st.sidebar.selectbox(
        "Select Patient",
        ["All"] + list(df["patient_id"].unique())
    )

    if patient != "All":
        df = df[df["patient_id"] == patient]

# =========================================================
# TITLE
# =========================================================
st.title("🩺 Clinical Diabetes AI Intelligence System")

# =========================================================
# 1. PATIENT OVERVIEW
# =========================================================
if menu == "Patient Overview":

    st.subheader("📊 Patient Summary KPIs")

    col1, col2, col3, col4 = st.columns(4)

    if "glucose" in df.columns:
        col1.metric("Avg Glucose", round(df["glucose"].mean(), 2))
        col2.metric("Max Glucose", round(df["glucose"].max(), 2))
        col3.metric("Min Glucose", round(df["glucose"].min(), 2))
        col4.metric("TIR %", round(df["tir"].mean() * 100, 2))

    if "time" in df.columns and "glucose" in df.columns:
        fig = px.line(df, x="time", y="glucose", title="24-Hour Glucose Trend")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2. GLUCOSE DYNAMICS
# =========================================================
elif menu == "Glucose Dynamics":

    st.subheader("📈 Glucose Dynamics Engine")

    col1, col2 = st.columns(2)

    if "glucose_roc" in df.columns:
        fig1 = px.line(df, y="glucose_roc", title="Glucose Rate of Change (ROC)")
        col1.plotly_chart(fig1, use_container_width=True)

    if "glucose_std" in df.columns:
        fig2 = px.line(df, y="glucose_std", title="Glucose Variability")
        col2.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 3. RISK & EVENTS
# =========================================================
elif menu == "Risk & Events":

    st.subheader("🚨 Risk Detection System")

    if "glucose" in df.columns:

        high_risk = df[df["glucose"] > 200]
        low_risk = df[df["glucose"] < 70]

        st.metric("Hyperglycemia Events", len(high_risk))
        st.metric("Hypoglycemia Events", len(low_risk))

        fig = px.scatter(
            df,
            x="time",
            y="glucose",
            color="risk_score",
            title="Risk Heatmap Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 4. INSULIN INTELLIGENCE
# =========================================================
elif menu == "Insulin Intelligence":

    st.subheader("🧠 Insulin Effectiveness System")

    if "glucose" in df.columns and "insulin" in df.columns:

        # Simple effectiveness score (prototype)
        df["insulin_effectiveness"] = (
            100 - (df["risk_score"] * 2)
        ).clip(0, 100)

        st.metric(
            "Avg Insulin Effectiveness Score",
            round(df["insulin_effectiveness"].mean(), 2)
        )

        fig = px.scatter(
            df,
            x="insulin",
            y="glucose",
            color="insulin_effectiveness",
            title="Insulin vs Glucose Response"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 5. MEAL & ACTIVITY
# =========================================================
elif menu == "Meal & Activity":

    st.subheader("🍽️ Nutrition & Activity Impact")

    if "carb_input" in df.columns and "glucose" in df.columns:

        fig1 = px.scatter(
            df,
            x="carb_input",
            y="glucose",
            title="Carbs vs Glucose Spike"
        )

        st.plotly_chart(fig1, use_container_width=True)

    if "steps" in df.columns and "glucose" in df.columns:

        fig2 = px.scatter(
            df,
            x="steps",
            y="glucose",
            title="Activity vs Glucose Control"
        )

        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 6. PREDICTIVE AI
# =========================================================
elif menu == "Predictive AI":

    st.subheader("🤖 Predictive Intelligence Engine")

    if "risk_score" in df.columns:

        df["predicted_risk"] = df["risk_score"].shift(-1).fillna(0)

        fig = px.line(
            df,
            y="predicted_risk",
            title="Predicted Glucose Risk Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### AI Alerts")

        if df["risk_score"].mean() > 50:
            st.error("High instability predicted → intervention needed")

        elif df["risk_score"].mean() > 30:
            st.warning("Moderate risk detected")

        else:
            st.success("Stable metabolic pattern")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Clinical AI Diabetes System | Streamlit + Predictive Analytics")
