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
    layout="wide"
)

# ======================================================
# UI STYLING (YOUR ATTRACTIVE DESIGN BACK)
# ======================================================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #081229, #0B1F3A);
}

h1, h2, h3 {
    color: white;
}

/* KPI CARDS */
.kpi {
    background: #EAF3FF;
    padding: 16px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

.kpi-title {
    font-size: 14px;
    color: #1B3B6F;
    font-weight: 600;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #081229;
}

/* INSIGHT BOX */
.insight {
    background: #EAF3FF;
    padding: 18px;
    border-radius: 18px;
    border-left: 5px solid #1B4F8C;
    color: #081229;
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
st.sidebar.title("🧠 AI Clinical System")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Predictive Analytics", "Prescriptive Analytics"]
)

# patient filter
if "patient_id" in df.columns:
    patient_list = ["All Patients"] + list(df["patient_id"].dropna().unique())
else:
    patient_list = ["All Patients"]

patient = st.sidebar.selectbox("Select Patient", patient_list)

if patient != "All Patients":
    df = df[df["patient_id"] == patient]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Diabetes Intelligence Dashboard")

# ======================================================
# KPIs (BEAUTIFUL CARDS)
# ======================================================
avg = df["glucose"].mean()
mx = df["glucose"].max()
mn = df["glucose"].min()
tir = df["glucose"].between(70, 180).mean() * 100

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Glucose</div><div class='kpi-value'>{avg:.1f}</div></div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Max Glucose</div><div class='kpi-value'>{mx:.1f}</div></div>", unsafe_allow_html=True)

with c3:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Min Glucose</div><div class='kpi-value'>{mn:.1f}</div></div>", unsafe_allow_html=True)

with c4:
    st.markdown(f"<div class='kpi'><div class='kpi-title'>Time in Range</div><div class='kpi-value'>{tir:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# OVERVIEW
# ======================================================
if menu == "Overview":

    st.subheader("📊 Glucose Trends")

    fig = px.line(df, x="time", y="glucose", color="patient_id" if "patient_id" in df.columns else None)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(df, names=pd.cut(df["glucose"], bins=[0,70,180,400], labels=["Low","Normal","High"]))
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# PREDICTIVE
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction")

    df["risk"] = df["glucose_roc"].abs() + df["glucose_rolling_std"]

    fig = px.line(df, x="time", y="risk", title="Risk Score Trend")
    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PRESCRIPTIVE
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decisions")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(df, x="time", y="glucose", color="risk_level")
    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight'>
    <b>AI Insight:</b><br><br>
    ✔ High glucose → adjust insulin strategy<br>
    ✔ Variability → monitor diet & activity<br>
    ✔ Stable range → continue current plan
    </div>
    """, unsafe_allow_html=True)
