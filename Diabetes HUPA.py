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
# GLOBAL UI
# ======================================================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #050B18, #0A1F3D, #0D2A52);
}

html, body, [class*="css"]  {
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
# LOAD DATA (FIXED)
# ======================================================
@st.cache_data
def load_data():

    demo = pd.read_csv("cleaned_demographics.csv")
    df = pd.read_excel(
        "cleaned_hupa_diabetes_recent.xlsb",
        engine="pyxlsb"
    )

    # safe merge
    if "patient_id" in demo.columns and "patient_id" in df.columns:
        df = df.merge(demo, on="patient_id", how="left")

    # time fix
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    # feature engineering (safe)
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

# SAFE PATIENT FILTER (FIXED CRASH ISSUE)
if "patient_id" in df.columns:
    patients = ["All Patients"] + list(df["patient_id"].dropna().unique())
    patient = st.sidebar.selectbox("Select Patient", patients)

    if patient != "All Patients":
        df = df[df["patient_id"].astype(str) == str(patient)]

# ======================================================
# HEADER
# ======================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# ======================================================
# SAFE KPI CALCULATION (FIXED BLANK ISSUE)
# ======================================================
if len(df) > 0 and "glucose" in df.columns:

    avg = df["glucose"].mean()
    mx = df["glucose"].max()
    mn = df["glucose"].min()
    tir = df["glucose"].between(70, 180).mean() * 100

else:
    avg = mx = mn = tir = 0

# KPI UI
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='kpi'><div class='kpi-title'>Avg Glucose</div><div class='kpi-value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'><div class='kpi-title'>Max Glucose</div><div class='kpi-value'>{mx:.1f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'><div class='kpi-title'>Min Glucose</div><div class='kpi-value'>{mn:.1f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'><div class='kpi-title'>Time in Range</div><div class='kpi-value'>{tir:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# OVERVIEW (CLINICAL EXPLAINABLE GRAPH)
# ======================================================
if menu == "Overview":

    st.subheader("📊 Glucose Trend (Clinical View)")

    if "glucose" in df.columns:

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
        • Green zone = safe range (70–180 mg/dL)  
        • Red line = hypoglycemia risk  
        • Orange line = hyperglycemia risk  
        """)

# ======================================================
# DESCRIPTIVE ANALYTICS (FIXED + ADDED PIE + VARIABILITY)
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Descriptive Clinical Analytics")

    if "glucose" in df.columns:

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="glucose", nbins=30,
                               title="Glucose Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            tir_df = pd.DataFrame({
                "Type": ["Low", "In Range", "High"],
                "Value": [
                    (df["glucose"] < 70).mean(),
                    df["glucose"].between(70, 180).mean(),
                    (df["glucose"] > 180).mean()
                ]
            })

            fig2 = px.pie(tir_df, names="Type", values="Value",
                          title="Time in Range Breakdown")

            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(df, x="time", y="glucose_rolling_std",
                       title="Glucose Variability (Instability Indicator)")

        st.plotly_chart(fig3, use_container_width=True)

        st.info("""
        📌 Clinical Meaning:
        • High variability = unstable diabetes control  
        • Pie chart shows risk distribution  
        • Histogram shows glucose spread pattern  
        """)

# ======================================================
# PREDICTIVE ANALYTICS
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Risk Prediction Engine")

    if "glucose" in df.columns:

        df["risk_score"] = (
            df["glucose_roc"].fillna(0).abs() +
            df["glucose_rolling_std"].fillna(0)
        )

        fig = px.line(df, x="time", y="risk_score",
                      title="Glucose Risk Score")

        st.plotly_chart(fig, use_container_width=True)

        st.info("Higher score = higher risk of glucose instability")

# ======================================================
# PRESCRIPTIVE ANALYTICS
# ======================================================
elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Clinical Decision Support")

    df["risk_level"] = np.where(df["glucose"] > 180, "High Risk", "Stable")

    fig = px.scatter(df, x="time", y="glucose",
                      color="risk_level")

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    AI Recommendation:
    • Monitor insulin timing  
    • Avoid high-carb spikes  
    • Maintain physical activity  
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

st.subheader("📌 Clinical Summary")

st.info(f"""
✔ Avg: {avg:.1f} mg/dL  
✔ Max: {mx:.1f}  
✔ Min: {mn:.1f}  
✔ Time in Range: {tir:.1f}%  

👉 Status: {'Stable control' if tir > 70 else 'Needs attention'}
""")
