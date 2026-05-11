import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Diabetes Intelligence System",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# GLOBAL UI THEME (NEW MODERN LOOK)
# ======================================================
st.markdown("""
<style>

/* BACKGROUND */
.main {
    background: linear-gradient(135deg, #050B18, #0A1F3D, #0D2A52);
}

/* FONT */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

/* HEADINGS */
h1, h2, h3 {
    color: #EAF3FF;
    font-weight: 700;
}

/* KPI CARD */
.kpi {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    backdrop-filter: blur(10px);
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

/* INSIGHT BOX */
.insight {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    border-left: 5px solid #4FC3F7;
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

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

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
    ["Overview", "Predictive Analytics", "Prescriptive Analytics"]
)

if "patient_id" in df.columns:
    patients = ["All Patients"] + list(df["patient_id"].dropna().unique())
else:
    patients = ["All Patients"]

patient = st.sidebar.selectbox("Select Patient", patients)

if patient != "All Patients":
    df = df[df["patient_id"] == patient]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# ======================================================
# KPIs
# ======================================================
avg = df["glucose"].mean()
mx = df["glucose"].max()
mn = df["glucose"].min()
tir = df["glucose"].between(70,180).mean()*100

c1,c2,c3,c4 = st.columns(4)

c1.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Glucose</div><div class='kpi-value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='kpi-title'>Max Glucose</div><div class='kpi-value'>{mx:.1f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='kpi-title'>Min Glucose</div><div class='kpi-value'>{mn:.1f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='kpi-title'>Time in Range</div><div class='kpi-value'>{tir:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# OVERVIEW (IMPROVED EXPLAINABLE GRAPH)
# ======================================================
if menu == "Overview":

    st.subheader("📊 Clinical Glucose Intelligence View")

    # 🔥 SMART CLINICAL GRAPH (NOT JUST LINE CHART)
    fig = go.Figure()

    # glucose line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["glucose"],
        mode="lines",
        name="Glucose Level",
        line=dict(color="#4FC3F7", width=2)
    ))

    # safe range zone
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="green",
        opacity=0.1,
        line_width=0
    )

    # hypoglycemia threshold
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red"
    )

    # hyperglycemia threshold
    fig.add_hline(
        y=180,
        line_dash="dash",
        line_color="orange"
    )

    fig.update_layout(
        title="Glucose Pattern with Clinical Risk Zones",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # EXPLANATION
    st.info("""
    🧠 Clinical Interpretation:
    • Blue line = glucose trend over time  
    • Green zone = safe glucose range (70–180 mg/dL)  
    • Red line = hypoglycemia risk  
    • Orange line = hyperglycemia risk  

    👉 This helps clinicians instantly identify instability patterns.
    """)

# ======================================================
# PREDICTIVE
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction Engine")

    df["risk"] = df["glucose_roc"].abs() + df["glucose_rolling_std"]

    fig = px.line(df, x="time", y="risk", title="Glucose Instability Risk Score")
    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    st.info("Higher score = unstable glucose + higher clinical risk")

# ======================================================
# PRESCRIPTIVE
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decision Support")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(df, x="time", y="glucose", color="risk_level")
    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    st.success("AI suggests monitoring insulin response & meal timing closely")
