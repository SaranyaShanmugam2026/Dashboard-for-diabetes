import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Clinical Diabetes Intelligence System",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# UI THEME
# ======================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #050B18, #0A1F3D, #0D2A52);
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

h1, h2, h3 {
    color: #EAF3FF;
    font-weight: 700;
}

.insight-box {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    border-left: 5px solid #4FC3F7;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    demo = pd.read_csv("cleaned_demographics.csv")
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb", engine="pyxlsb")

    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for col in ["glucose", "heart_rate", "steps", "calories"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    df["glucose_roc"] = df["glucose"].diff()
    df["glucose_rolling_std"] = df["glucose"].rolling(12, min_periods=1).std()

    return df


df = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 Clinical AI System")

menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "Descriptive Analytics", "Predictive Analytics", "Prescriptive Analytics"]
)

patients = ["All Patients"] + list(df["patient_id"].dropna().unique())
patient = st.sidebar.selectbox("Select Patient", patients)

if patient != "All Patients":
    df = df[df["patient_id"] == patient]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# ======================================================
# KPI
# ======================================================
col1, col2, col3, col4 = st.columns(4)

tir = ((df['glucose'].between(70, 180)).mean()) * 100

with col1:
    st.metric("Average Glucose", f"{round(df['glucose'].mean(), 1)} mg/dL")

with col2:
    st.metric("Time In Range", f"{round(tir, 1)}%")

with col3:
    st.metric("Average Heart Rate", f"{round(df['heart_rate'].mean(), 1)} bpm")

with col4:
    st.metric("Average Steps", f"{int(df['steps'].mean())}")

# ======================================================
# OVERVIEW
# ======================================================
if menu == "Dataset Overview":

    st.subheader("📘 Glucose Trend (Clinical View)")

    df["glucose_smooth"] = df["glucose"].rolling(12, min_periods=1).mean()

    fig_glucose = go.Figure()

    fig_glucose.add_trace(go.Scatter(
        x=df["time"],
        y=df["glucose"],
        mode="lines",
        name="Raw",
        line=dict(width=1, color="rgba(255,255,255,0.3)")
    ))

    fig_glucose.add_trace(go.Scatter(
        x=df["time"],
        y=df["glucose_smooth"],
        mode="lines",
        name="Smoothed",
        line=dict(width=3, color="#4FC3F7")
    ))

    fig_glucose.add_hline(y=70, line_dash="dash", line_color="red")
    fig_glucose.add_hline(y=180, line_dash="dash", line_color="orange")

    fig_glucose.update_layout(
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)

    # Distribution
    st.subheader("📊 Glucose Distribution")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["glucose"],
        nbinsx=40,
        marker_color="#4FC3F7"
    ))

    fig_hist.update_layout(template="plotly_dark", height=350)

    st.plotly_chart(fig_hist, use_container_width=True)

    # Activity
    st.subheader("🚶 Physical Activity Trend")

    fig_steps = px.area(
        df,
        x="time",
        y="steps",
        template="plotly_dark"
    )

    st.plotly_chart(fig_steps, use_container_width=True)

# ======================================================
# DESCRIPTIVE
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Heart Rate vs Glucose")

    fig_hr = px.scatter(
        df,
        x="heart_rate",
        y="glucose",
        color="glucose",
        trendline="lowess",
        template="plotly_dark",
        color_continuous_scale="RdYlBu_r"
    )

    st.plotly_chart(fig_hr, use_container_width=True)

    st.subheader("📦 Hourly Glucose Pattern")

    fig_hour = px.box(
        df,
        x="hour",
        y="glucose",
        points="all",
        template="plotly_dark"
    )

    st.plotly_chart(fig_hour, use_container_width=True)

    corr = df[["glucose", "heart_rate", "steps", "calories"]].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

# ======================================================
# PRESCRIPTIVE
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Risk Monitoring")

    df["risk_score"] = df.get("risk_score", df["glucose"].rolling(3).mean())

    df["Risk_Level"] = np.where(df["risk_score"] > 50, "High Risk", "Stable")

    fig = px.scatter(
        df,
        x="time",
        y="glucose",
        color="Risk_Level",
        size="steps",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 AI Recommendations")

    if df["risk_score"].mean() > 40:
        st.error("High instability detected → adjust insulin strategy")

    if df["glucose"].max() > 250:
        st.warning("Severe hyperglycemia detected")

    if df["glucose"].min() < 60:
        st.info("Hypoglycemia risk detected")
# ======================================================
# PREDICTIVE
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction")

    df["risk_score"] = (
        abs(df["glucose_roc"]) * 0.4 +
        df["glucose_rolling_std"] * 0.4 +
        abs(df["heart_rate"]) * 0.2
    )

    df["risk_smooth"] = df["risk_score"].rolling(10, min_periods=1).mean()

    fig_risk = go.Figure()

    fig_risk.add_trace(go.Scatter(
        x=df["time"],
        y=df["risk_score"],
        line=dict(width=1, color="rgba(255,0,0,0.3)"),
        name="Raw"
    ))

    fig_risk.add_trace(go.Scatter(
        x=df["time"],
        y=df["risk_smooth"],
        line=dict(width=3, color="#FF5252"),
        name="Smoothed"
    ))

    fig_risk.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("📉 Variability vs Glucose")

    fig_pred = px.scatter(
        df,
        x="glucose_rolling_std",
        y="glucose",
        color="risk_score",
        template="plotly_dark",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig_pred, use_container_width=True)



# ======================================================
# INSIGHTS
# ======================================================
st.markdown("---")

st.subheader("📌 Clinical Insights")

st.markdown("""
<div class='insight-box'>
<b>Key Insights:</b><br><br>

• Smoothed glucose trends improve clinical interpretation<br>
• Variability is a strong predictor of risk<br>
• Physical activity stabilizes glucose response<br>
• AI-based risk scoring enables early intervention<br>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Clinical Diabetes Intelligence System | Streamlit + Plotly")
