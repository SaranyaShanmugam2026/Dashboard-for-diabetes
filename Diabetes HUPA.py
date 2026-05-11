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

.kpi {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 16px;
    border-radius: 16px;
    text-align: center;
}

.kpi-title { font-size: 13px; color: #B8D7FF; }
.kpi-value { font-size: 26px; font-weight: bold; color: white; }

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

    demo = pd.read_csv("cleaned_demographics(1).csv")
    df = pd.read_excel("cleaned_hupa_diabetes_recent(1).xlsb", engine="pyxlsb")

    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for col in ["glucose", "heart_rate", "steps", "calories"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
# KPI FIXED (NO BLANK ISSUE)
# ======================================================
if len(df) > 0:

    avg = df["glucose"].mean()
    mx = df["glucose"].max()
    mn = df["glucose"].min()
    tir = df["glucose"].between(70, 180).mean() * 100

else:
    avg = mx = mn = tir = 0

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Glucose</div><div class='kpi-value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='kpi-title'>Max Glucose</div><div class='kpi-value'>{mx:.1f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='kpi-title'>Min Glucose</div><div class='kpi-value'>{mn:.1f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='kpi-title'>Time in Range</div><div class='kpi-value'>{tir:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# OVERVIEW (SMART VISUAL)
# ======================================================
if menu == "Dataset Overview":

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

# ======================================================
# DESCRIPTIVE (IMPROVED)
# ======================================================
elif menu == "Descriptive Analytics":

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="glucose", nbins=30, title="Glucose Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        pie_df = pd.DataFrame({
            "Type": ["Low", "In Range", "High"],
            "Value": [
                (df["glucose"] < 70).mean(),
                df["glucose"].between(70, 180).mean(),
                (df["glucose"] > 180).mean()
            ]
        })
        fig2 = px.pie(pie_df, names="Type", values="Value", title="TIR Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(df, x="time", y="glucose_rolling_std",
                   title="Glucose Variability (Instability)")
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# PREDICTIVE (FIXED)
# ======================================================
elif menu == "Predictive Analytics":

    df["risk_score"] = (
        df["glucose_roc"].fillna(0).abs() +
        df["glucose_rolling_std"].fillna(0)
    )

    fig = px.line(df, x="time", y="risk_score", title="Risk Score Trend")
    st.plotly_chart(fig, use_container_width=True)

    fig_var = px.scatter(
        df,
        x="glucose_rolling_std",
        y="glucose",
        color="steps",
        size="heart_rate" if "heart_rate" in df.columns else None,
        title="Glucose Variability Pattern"
    )

    st.plotly_chart(fig_var, use_container_width=True)

# ======================================================
# PRESCRIPTIVE
# ======================================================
elif menu == "Prescriptive Analytics":

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(df, x="time", y="glucose", color="risk_level")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# INSIGHT
# ======================================================
st.markdown("---")

st.subheader("📌 Clinical Summary")

st.info(f"""
✔ Avg: {avg:.1f} mg/dL  
✔ Max: {mx:.1f}  
✔ Min: {mn:.1f}  
✔ Time in Range: {tir:.1f}%  

👉 {'Stable control' if tir > 70 else 'Needs intervention'}
""")
