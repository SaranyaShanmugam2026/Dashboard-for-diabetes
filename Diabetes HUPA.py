# app.py
# ===========================
# GLUCOSE INTELLIGENCE PLATFORM (CLEAN VERSION)
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="Glucose Intelligence Platform",
    page_icon="🩺",
    layout="wide"
)

# ===========================
# CLEAN HEADER
# ===========================

st.markdown("""
<div style="
background:linear-gradient(90deg,#0f766e,#2563eb);
padding:25px;
border-radius:20px;
color:white;
text-align:center;
margin-bottom:20px;">
<h1>🩺 Glucose Intelligence Platform</h1>
<p>Clinical Decision Support System for Continuous Glucose Monitoring, Insulin, Meals & Activity Analysis</p>
</div>
""", unsafe_allow_html=True)

# ===========================
# LOAD DATA
# ===========================

@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_hupa_diabetes_recent.xlsb")
    demo = pd.read_csv("cleaned_demographics.csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# ===========================
# CLEANING PIPELINE
# ===========================

df = df.dropna(subset=["glucose", "time", "patient_id"])
df = df.sort_values(["patient_id", "time"])

df["glucose"] = df.groupby("patient_id")["glucose"].transform(
    lambda x: x.interpolate(limit=2)
)

df["glucose"] = df["glucose"].clip(40, 500)

df["hour"] = df["time"].dt.hour
df["is_night"] = df["hour"].between(0, 6).astype(int)

df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["rolling_mean"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12).mean()
    .reset_index(level=0, drop=True)
)

df["is_in_range"] = df["glucose"].between(70, 180).astype(int)

# ===========================
# NAVIGATION MENU (CLICKABLE LINKS)
# ===========================

st.markdown("### 📌 Navigation")

nav = st.columns(6)

page = None

with nav[0]:
    if st.button("Home"):
        page = "home"

with nav[1]:
    if st.button("Cleaning"):
        page = "cleaning"

with nav[2]:
    if st.button("Exploration"):
        page = "exploration"

with nav[3]:
    if st.button("Meals"):
        page = "meals"

with nav[4]:
    if st.button("Activity"):
        page = "activity"

with nav[5]:
    if st.button("Model"):
        page = "model"

# default page
if page is None:
    page = "home"

st.markdown("---")

# ===========================
# HOME
# ===========================

if page == "home":

    st.subheader("Clinical Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Patients", df["patient_id"].nunique())
    col2.metric("Records", len(df))
    col3.metric("Avg Glucose", round(df["glucose"].mean(), 1))

    st.info("""
    This system integrates glucose monitoring, insulin delivery, meal intake, and activity data
    to support clinical decision-making and patient monitoring.
    """)

# ===========================
# DATA CLEANING
# ===========================

elif page == "cleaning":

    st.subheader("Data Cleaning Overview")

    st.write("Missing Values")
    st.dataframe(df.isnull().sum())

    st.write("Duplicate Records")
    st.write(df.duplicated().sum())

    fig = px.histogram(df, x="glucose", nbins=40, title="Glucose Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, y="glucose", title="Outlier Detection")
    st.plotly_chart(fig2, use_container_width=True)

# ===========================
# EXPLORATION
# ===========================

elif page == "exploration":

    st.subheader("Patient-Level Analysis")

    patient = st.selectbox("Select Patient", df["patient_id"].unique())

    temp = df[df["patient_id"] == patient]

    fig = px.line(temp, x="time", y="glucose", title="Glucose Trend")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    col1.metric("Avg Glucose", round(temp["glucose"].mean(), 2))
    col2.metric("Time in Range", round(temp["is_in_range"].mean()*100, 2))

# ===========================
# MEALS
# ===========================

elif page == "meals":

    st.subheader("Meal Impact Analysis")

    meal = df[df["carb_input"] > 0].copy()

    meal["spike"] = meal.groupby("patient_id")["glucose"].shift(-12) - meal["glucose"]

    fig = px.scatter(meal, x="carb_input", y="spike", title="Carbs vs Glucose Spike")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# ACTIVITY
# ===========================

elif page == "activity":

    st.subheader("Activity vs Glucose Stability")

    fig = px.scatter(df, x="steps", y="glucose", title="Steps vs Glucose")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# MODEL
# ===========================

elif page == "model":

    st.subheader("Hypoglycemia Prediction Model")

    model_df = df.copy()
    model_df["target"] = (model_df["glucose"].shift(-6) < 70).astype(int)

    features = ["glucose", "glucose_roc", "hour"]

    model_df = model_df.dropna()

    X = model_df[features]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    st.metric("Accuracy", round(accuracy_score(y_test, pred), 3))
    st.metric("ROC-AUC", round(roc_auc_score(y_test, prob), 3))

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
