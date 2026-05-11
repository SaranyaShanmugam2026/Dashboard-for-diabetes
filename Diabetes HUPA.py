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
# UI THEME (CLEAN + MODERN)
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

.kpi {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 16px;
    border-radius: 16px;
    text-align: center;
}

.kpi-title {
    font-size: 13px;
    color: #B8D7FF;
}

.kpi-value {
    font-size: 26px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():

    demo = pd.read_csv("cleaned_demographics.csv")
    df = pd.read_excel(
        "cleaned_hupa_diabetes_recent.xlsb",
        engine="pyxlsb"
    )

    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for col in ["glucose", "heart_rate", "steps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_rolling_std"] = df["glucose"].rolling(12).std().fillna(0)

    return df


df = load_data()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("🧠 Clinical AI System")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Descriptive Analytics", "Predictive Analytics", "Prescriptive Analytics"]
)

# patient filter
if "patient_id" in df.columns:
    patients = ["All Patients"] + list(df["patient_id"].dropna().unique())
    patient = st.sidebar.selectbox("Select Patient", patients)

    if patient != "All Patients":
        df = df[df["patient_id"] == patient]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# ======================================================
# KPI ENGINE (FIXED + SAFE)
# ======================================================
if "glucose" in df.columns and len(df) > 0:
    d = df.dropna(subset=["glucose"])

    avg = d["glucose"].mean()
    mx = d["glucose"].max()
    mn = d["glucose"].min()
    tir = d["glucose"].between(70, 180).mean() * 100
else:
    avg = mx = mn = tir = 0

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Glucose</div><div class='kpi-value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='kpi-title'>Max Glucose</div><div class='kpi-value'>{mx:.1f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='kpi-title'>Min Glucose</div><div class='kpi-value'>{mn:.1f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='kpi-title'>Time in Range</div><div class='kpi-value'>{tir:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# OVERVIEW (CLINICAL INTELLIGENCE VIEW)
# ======================================================
if menu == "Overview":

    st.subheader("📊 Glucose Clinical Timeline")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["glucose"],
        mode="lines",
        name="Glucose",
        line=dict(color="#4FC3F7")
    ))

    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1)
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=180, line_dash="dash", line_color="orange")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    🧠 Interpretation:
    • Green zone = safe glucose range  
    • Red line = hypoglycemia risk  
    • Orange line = hyperglycemia risk  
    • Spikes indicate insulin or meal imbalance
    """)

# ======================================================
# DESCRIPTIVE ANALYTICS (IMPROVED)
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Clinical Pattern Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="glucose", nbins=30,
                            title="Glucose Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        pie_df = pd.DataFrame({
            "Type": ["Low", "In Range", "High"],
            "Count": [
                (df["glucose"] < 70).sum(),
                df["glucose"].between(70, 180).sum(),
                (df["glucose"] > 180).sum()
            ]
        })

        fig2 = px.pie(pie_df, names="Type", values="Count",
                      title="Time in Range Breakdown")

        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(df, x="time", y="glucose_rolling_std",
                   title="Glucose Variability (Stability Indicator)")
    st.plotly_chart(fig3, use_container_width=True)

    st.info("""
    📌 Clinical Insight:
    • High variability = unstable diabetes  
    • Pie chart = risk distribution  
    • Histogram = glucose spread
    """)

# ======================================================
# PREDICTIVE ANALYTICS
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction Engine")

    df["risk_score"] = (
        df["glucose_roc"].fillna(0).abs() +
        df["glucose_rolling_std"].fillna(0)
    )

    fig = px.line(df, x="time", y="risk_score",
                  title="Glucose Instability Risk Score")

    st.plotly_chart(fig, use_container_width=True)

    st.success("Higher score = higher risk of glucose instability")

# ======================================================
# PRESCRIPTIVE ANALYTICS
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decision Support")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(df, x="time", y="glucose",
                     color="risk_level")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    AI Recommendations:
    • Optimize insulin timing  
    • Reduce late-night carbs  
    • Maintain activity consistency  
    """)
