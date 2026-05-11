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
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():

    # ✔ EXACT FILE NAMES FROM YOUR GITHUB
    demo = pd.read_csv("cleaned_demographics(1).csv")

    df = pd.read_excel(
        "cleaned_hupa_diabetes_recent(1).xlsb",
        engine="pyxlsb"
    )

    # merge if possible
    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    # datetime handling
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

    # feature engineering
    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_rolling_std"] = df["glucose"].rolling(12).std().fillna(0)

    return df

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 Clinical AI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Descriptive Analytics",
        "Predictive Analytics",
        "Prescriptive Analytics",
        "Clinical AI Engine"
    ]
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
# KPIs
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

st.markdown("---")

# ======================================================
# OVERVIEW
# ======================================================
if menu == "Overview":

    st.subheader("📘 Glucose Overview")

    fig = px.line(df, x="time", y="glucose", color="patient_id")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="glucose", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# DESCRIPTIVE
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Clinical Patterns")

    fig1 = px.scatter(
        df,
        x="heart_rate",
        y="glucose",
        color="steps",
        title="Heart Rate vs Glucose"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(
        df,
        x="time",
        y="glucose_rolling_std",
        title="Glucose Variability"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(
        df,
        y="glucose",
        title="Glucose Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# PREDICTIVE
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction Engine")

    df["risk_score"] = (
        df["glucose_roc"].fillna(0).abs() * 0.6 +
        df["glucose_rolling_std"] * 0.4
    )

    fig = px.line(df, x="time", y="risk_score")
    st.plotly_chart(fig, use_container_width=True)

    st.info("⚠ Higher risk score indicates unstable glucose trends")

# ======================================================
# PRESCRIPTIVE
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decision Support")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(
        df,
        x="time",
        y="glucose",
        color="risk_level"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📌 Clinical Actions")

    if tir < 70:
        st.warning("Adjust insulin strategy recommended")
    if df["glucose"].max() > 250:
        st.error("Severe hyperglycemia detected")
    if df["glucose"].min() < 60:
        st.error("Hypoglycemia risk detected")

# ======================================================
# CLINICAL AI ENGINE (IMPORTANT PART)
# ======================================================
elif menu == "Clinical AI Engine":

    st.subheader("🧠 Clinical Question Engine")

    question = st.selectbox(
        "Ask Clinical AI",
        [
            "What is Time in Range meaning?",
            "Is patient high risk?",
            "What causes glucose spikes?",
            "Insulin effectiveness?",
            "Meal impact on glucose?",
            "Sleep impact on stability?",
            "Exercise effect on glucose?"
        ]
    )

    if question == "What is Time in Range meaning?":
        st.info("TIR = % of glucose readings between 70–180 mg/dL")

    elif question == "Is patient high risk?":
        if tir < 60:
            st.error("High risk patient detected")
        else:
            st.success("Stable risk profile")

    elif question == "What causes glucose spikes?":
        st.info("Carbohydrate intake, low activity, and insulin delay")

    elif question == "Insulin effectiveness?":
        st.info("Measured using glucose reduction rate and stability improvement")

    elif question == "Meal impact on glucose?":
        st.info("High carb meals cause sharp post-meal glucose spikes")

    elif question == "Sleep impact on stability?":
        st.info("Poor sleep increases glucose variability")

    elif question == "Exercise effect on glucose?":
        st.info("Exercise reduces glucose levels and improves insulin sensitivity")

# ======================================================
# FOOTER INSIGHTS
# ======================================================
st.markdown("---")

st.subheader("📌 Clinical Summary")

st.info(f"""
✔ Average Glucose: {avg_glucose:.1f} mg/dL  
✔ Time in Range: {tir:.1f}%  

👉 Overall Interpretation:
{'Stable metabolic control' if tir > 70 else 'Unstable glucose regulation detected'}
""")
