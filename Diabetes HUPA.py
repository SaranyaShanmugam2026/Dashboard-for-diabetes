import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Clinical AI Diabetes Dashboard",
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

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    df["glucose_roc"] = df["glucose"].diff()
    df["glucose_std"] = df["glucose"].rolling(12).std().fillna(0)

    return df

df = load_data()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("🧠 Clinical Controls")

hypo = st.sidebar.slider("Hypoglycemia Threshold", 50, 80, 70)
hyper = st.sidebar.slider("Hyperglycemia Threshold", 150, 250, 180)

patient = st.sidebar.selectbox(
    "Select Patient",
    ["All"] + list(df["patient_id"].unique())
)

if patient != "All":
    df = df[df["patient_id"] == patient]

# =========================================================
# TITLE
# =========================================================
st.title("🩺 AI Clinical Diabetes Intelligence System")

# =========================================================
# KPI CALCULATION
# =========================================================
avg_glucose = df["glucose"].mean()
max_glucose = df["glucose"].max()
min_glucose = df["glucose"].min()

tir = df["glucose"].between(hypo, hyper).mean() * 100

hyper_count = (df["glucose"] > hyper).sum()
hypo_count = (df["glucose"] < hypo).sum()

# =========================================================
# BEAUTIFUL KPI CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Avg Glucose", f"{avg_glucose:.1f} mg/dL")
col2.metric("🔺 Max Glucose", f"{max_glucose:.1f}")
col3.metric("🔻 Min Glucose", f"{min_glucose:.1f}")
col4.metric("🎯 Time in Range", f"{tir:.1f}%")

# =========================================================
# PIE CHART - TIME IN RANGE
# =========================================================
st.subheader("📌 Time in Range Distribution")

tir_data = pd.DataFrame({
    "Category": ["Low (<70)", "In Range (70–180)", "High (>180)"],
    "Value": [
        hypo_count,
        ((df["glucose"] >= hypo) & (df["glucose"] <= hyper)).sum(),
        hyper_count
    ]
})

fig_pie = px.pie(
    tir_data,
    names="Category",
    values="Value",
    title="Glucose Distribution Overview",
    hole=0.4
)

st.plotly_chart(fig_pie, use_container_width=True)

# =========================================================
# GLUCOSE TREND
# =========================================================
st.subheader("📈 24-Hour Glucose Trend")

fig_line = px.line(
    df,
    x="time",
    y="glucose",
    title="Glucose Trend Over Time",
    color_discrete_sequence=["#00B4DB"]
)

fig_line.add_hline(y=hypo, line_dash="dash", line_color="red")
fig_line.add_hline(y=hyper, line_dash="dash", line_color="orange")

st.plotly_chart(fig_line, use_container_width=True)

# =========================================================
# RISK DISTRIBUTION (DONUT CHART)
# =========================================================
st.subheader("🚨 Risk Distribution")

df["risk_category"] = np.where(
    df["glucose"] > hyper, "High Risk",
    np.where(df["glucose"] < hypo, "Low Risk", "Stable")
)

risk_fig = px.pie(
    df,
    names="risk_category",
    title="Risk Level Breakdown",
    hole=0.5,
    color_discrete_map={
        "High Risk": "red",
        "Low Risk": "blue",
        "Stable": "green"
    }
)

st.plotly_chart(risk_fig, use_container_width=True)

# =========================================================
# GLUCOSE VARIABILITY
# =========================================================
st.subheader("📊 Glucose Variability (Stability Indicator)")

fig_var = px.line(
    df,
    y="glucose_std",
    title="Rolling Glucose Variability"
)

st.plotly_chart(fig_var, use_container_width=True)

# =========================================================
# CLINICAL EXPLANATION BOXES
# =========================================================
st.markdown("---")
st.subheader("🧠 Clinical Interpretation")

if tir > 70:
    st.success("✔ Patient is maintaining good glucose control (TIR > 70%)")
else:
    st.warning("⚠ Patient is below optimal glucose stability threshold")

if hyper_count > 10:
    st.error("⚠ Frequent hyperglycemia detected → review insulin strategy")

if hypo_count > 5:
    st.error("⚠ Hypoglycemia risk detected → adjust basal insulin")

# =========================================================
# INSIGHT SUMMARY BOX
# =========================================================
st.markdown("### 📌 AI Summary")

st.info(
    f"""
    • Average glucose: {avg_glucose:.1f} mg/dL  
    • Time in Range: {tir:.1f}%  
    • Hyper events: {hyper_count}  
    • Hypo events: {hypo_count}  

    👉 Overall pattern suggests:
    {'Stable metabolic control' if tir > 70 else 'Unstable glucose regulation'}
    """
)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Clinical AI Diabetes Dashboard | Enhanced Visualization System")
