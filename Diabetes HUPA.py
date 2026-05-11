import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(
    page_title="AI Diabetes Intelligence Dashboard",
    page_icon="🩺",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():

    demo_df = pd.read_csv("cleaned_demographics.csv")

    diabetes_df = pd.read_excel(
        "cleaned_hupa_diabetes_recent.xlsb",
        engine="pyxlsb"
    )

    if "patient_id" in demo_df.columns and "patient_id" in diabetes_df.columns:
        df = diabetes_df.merge(demo_df, on="patient_id", how="left")
    else:
        df = diabetes_df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour

    if "glucose" in df.columns:
        df["glucose_roc"] = df["glucose"].diff()
        df["glucose_rolling_std"] = df["glucose"].rolling(12).std().fillna(0)

    return df

df = load_data()

# TITLE
st.title("🩺 AI Diabetes Dashboard")

# SIDEBAR
menu = st.sidebar.radio("Navigation", ["Overview", "Analytics", "Insights"])

# OVERVIEW
if menu == "Overview":
    st.subheader("Overview")

    if "glucose" in df.columns:
        fig = px.line(df, x="time", y="glucose")
        st.plotly_chart(fig)

# ANALYTICS
elif menu == "Analytics":
    st.subheader("Analytics")

    if "heart_rate" in df.columns and "glucose" in df.columns:
        fig = px.scatter(df, x="heart_rate", y="glucose")
        st.plotly_chart(fig)

# INSIGHTS
elif menu == "Insights":
    st.subheader("Insights")

    if "glucose" in df.columns:
        avg = df["glucose"].mean()

        if avg > 180:
            st.error("High glucose risk")
        else:
            st.success("Stable glucose levels")