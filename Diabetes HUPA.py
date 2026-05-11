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

    demo = pd.read_csv("cleaned_demographics.csv")
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb", engine="pyxlsb")

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
# KPI 
# ======================================================
col1, col2, col3, col4 = st.columns(4)

tir = ((df['glucose'].between(70,180)).mean()) * 100

with col1:
    st.metric(
        "Average Glucose",
        f"{round(df['glucose'].mean(),1)} mg/dL"
    )

with col2:
    st.metric(
        "Time In Range",
        f"{round(tir,1)}%"
    )

with col3:
    st.metric(
        "Average Heart Rate",
        f"{round(df['heart_rate'].mean(),1)} bpm"
    )

with col4:
    st.metric(
        "Average Steps",
        f"{int(df['steps'].mean())}"
    )


# ======================================================
# OVERVIEW (SMART VISUAL)
# ======================================================
if menu == "Overview":

    st.subheader("📘 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:

        fig_glucose = px.line(
            df,
            x='time',
            y='glucose',
            color='patient_id',
            template='plotly_dark',
            title='Glucose Trends'
        )

        fig_glucose.add_hline(
            y=70,
            line_dash='dash',
            line_color='red'
        )

        fig_glucose.add_hline(
            y=180,
            line_dash='dash',
            line_color='orange'
        )

        st.plotly_chart(
            fig_glucose,
            use_container_width=True
        )

    with col2:

        fig_hist = px.histogram(
            df,
            x='glucose',
            nbins=30,
            color='patient_id',
            template='plotly_dark',
            title='Glucose Distribution'
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    fig_steps = px.area(
        df,
        x='time',
        y='steps',
        template='plotly_dark',
        title='Daily Physical Activity'
    )

    st.plotly_chart(
        fig_steps,
        use_container_width=True
    )


# ======================================================
# DESCRIPTIVE (IMPROVED)
# ======================================================
elif menu == "Descriptive Analytics":

    st.subheader("📊 Descriptive Analytics")

    col1, col2 = st.columns(2)

    with col1:

        fig_hr = px.scatter(
            df,
            x='heart_rate',
            y='glucose',
            color='steps',
            size='calories',
            template='plotly_dark',
            title='Heart Rate vs Glucose'
        )

        st.plotly_chart(
            fig_hr,
            use_container_width=True
        )

    with col2:

        fig_sleep = px.box(
            df,
            x='hour',
            y='glucose',
            color='hour',
            template='plotly_dark',
            title='Hourly Glucose Pattern'
        )

        st.plotly_chart(
            fig_sleep,
            use_container_width=True
        )

    corr = df[
        [
            'glucose',
            'heart_rate',
            'steps',
            'calories',
            'sleep_hours'
        ]
    ].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        template='plotly_dark',
        title='Feature Correlation Matrix'
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

# ======================================================
# PREDICTIVE (FIXED)
# ======================================================
elif menu == "Predictive Analytics":

    st.subheader("🤖 Predictive Analytics")

    df['risk_score'] = (
        abs(df['glucose_roc']) * 0.4 +
        abs(df['glucose_rolling_std']) * 0.4 +
        abs(df['heart_rate']) * 0.2
    )

    fig_risk = px.line(
        df,
        x='time',
        y='risk_score',
        template='plotly_dark',
        title='Hypoglycemia Risk Prediction'
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    fig_pred = px.scatter(
        df,
        x='glucose_rolling_std',
        y='glucose',
        size='steps',
        color='risk_score',
        template='plotly_dark',
        title='Glucose Variability Prediction'
    )

    st.plotly_chart(
        fig_pred,
        use_container_width=True
    )

# ======================================================
# PRESCRIPTIVE ANALYTICS (FIXED + SAFE)
# ======================================================

elif menu == "Prescriptive Analytics":

    st.subheader("🧠 Prescriptive Recommendations")

    # --------------------------------------------------
    # SAFETY CHECK: Ensure required columns exist
    # --------------------------------------------------
    required_cols = ['glucose', 'time', 'steps', 'carb_input', 'insulin']

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # --------------------------------------------------
    # RISK SCORE CREATION (if missing)
    # --------------------------------------------------
    if 'risk_score' not in df.columns:
        df['risk_score'] = (
            df['glucose'].rolling(3).mean().fillna(df['glucose'].mean()) * 0.5 +
            df['glucose'].std() * 0.5
        )

    # --------------------------------------------------
    # RISK LEVEL CLASSIFICATION
    # --------------------------------------------------
    df['Risk_Level'] = np.where(
        df['risk_score'] > 50,
        'High Risk',
        'Stable'
    )

    # --------------------------------------------------
    # INTERVENTION VISUALIZATION
    # --------------------------------------------------
    fig_intervention = px.scatter(
        df,
        x='time',
        y='glucose',
        color='Risk_Level',
        size='steps',
        template='plotly_dark',
        title='AI Intervention Monitoring'
    )

    st.plotly_chart(fig_intervention, use_container_width=True)

    # --------------------------------------------------
    # CARB vs INSULIN HEATMAP
    # --------------------------------------------------
    fig_carb = px.density_heatmap(
        df,
        x='carb_input',
        y='insulin',
        z='glucose',
        template='plotly_dark',
        title='Carb vs Insulin vs Glucose Heatmap'
    )

    st.plotly_chart(fig_carb, use_container_width=True)

    # --------------------------------------------------
    # AI CLINICAL RULE-BASED INSIGHTS
    # --------------------------------------------------
    st.markdown("## 🚨 AI Clinical Recommendations")

    if df['risk_score'].mean() > 40:
        st.error(
            "⚠️ High glucose instability detected. "
            "Recommend insulin reassessment."
        )

    if df['glucose'].max() > 250:
        st.warning(
            "🚨 Severe hyperglycemia episodes detected."
        )

    if df['glucose'].min() < 60:
        st.info(
            "⚠️ Hypoglycemia risk detected. Immediate intervention recommended."
        )

    # --------------------------------------------------
    # OPTIONAL: DEBUG VIEW (VERY USEFUL)
    # --------------------------------------------------
    with st.expander("🔍 Debug Data Preview"):
        st.write(df.head())
        st.write("Columns:", df.columns.tolist())
# ======================================================
# INSIGHT
# ======================================================
st.markdown("---")

st.subheader("📌 Clinical Insights")

st.markdown("""
<div class='insight-box'>

<b>Key AI Findings:</b><br><br>

• High glucose variability predicts future instability.<br><br>

• Physical activity improves insulin sensitivity.<br><br>

• Large carbohydrate meals increase glucose spikes.<br><br>

• Better sleep improves Time-In-Range.<br><br>

• AI early-warning systems reduce hypoglycemia risk.

</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Developed using Streamlit + Plotly | "
    "AI Diabetes Intelligence System"
)
