import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_rainfall_data
from utils.features import add_rainfall_features, add_statistical_features
from utils.labeling import add_flood_label
from utils.predict import load_model, prepare_input
from utils.explain import get_feature_importance


st.set_page_config(
    page_title="Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Load data & model
@st.cache_data
def load_data():
    df = load_rainfall_data()
    df = add_rainfall_features(df)
    df = add_statistical_features(df)
    df = add_flood_label(df)
    return df

@st.cache_resource
def load_flood_model():
    return load_model()

data = load_data()
model = load_flood_model()

FEATURE_NAMES = [
    "annual_rainfall",
    "monsoon_rainfall",
    "pre_monsoon_rainfall",
    "post_monsoon_rainfall",
    "monsoon_anomaly",
    "monsoon_percentile"
]

# Sidebar navigation
st.sidebar.title("🌊 Flood Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["🤖 Flood Risk Prediction (ML)", "📊 Rainfall Analysis"]
)

state = st.sidebar.selectbox(
    "Select State",
    sorted(data["SUBDIVISION"].unique())
)

# Sidebar options ONLY for analysis
if page == "📊 Rainfall Analysis":
    month = st.sidebar.selectbox(
        "Select Month",
        data.columns[2:14]
    )
    period = st.sidebar.radio(
        "Select Period",
        ["Annual", "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]
    )

# Filter state data
state_data = data[data["SUBDIVISION"] == state].copy()

# Helper for analysis period rainfall
def compute_period_rainfall(df, period):
    if period == "Annual":
        return df.iloc[:, 2:14].sum(axis=1)
    elif period == "Jan-Feb":
        return df.iloc[:, 2:4].sum(axis=1)
    elif period == "Mar-May":
        return df.iloc[:, 4:7].sum(axis=1)
    elif period == "Jun-Sep":
        return df.iloc[:, 7:11].sum(axis=1)
    elif period == "Oct-Dec":
        return df.iloc[:, 11:14].sum(axis=1)


# 🤖 PAGE 1 — ML FLOOD RISK PREDICTION
if page == "🤖 Flood Risk Prediction (ML)":

    st.title("🤖 Flood Risk Prediction")

    st.info(
        "This page predicts **seasonal / annual flood risk** using a machine "
        "learning model trained on historical rainfall patterns."
    )

    # Use latest year automatically
    latest_year = state_data["YEAR"].max()

    col1, col2 = st.columns(2)
    col1.metric("State", state)

    # ML prediction (NO PERIOD DEPENDENCY)
    X_input = prepare_input(state_data)
    probability = model.predict_proba(X_input)[0][1]

    st.subheader("📈 Flood Risk Probability")
    st.metric("Flood Probability", f"{probability * 100:.1f}%")

    # Risk thresholds
    if probability > 0.55:
        st.error("Flood Risk Level: HIGH")
    elif probability > 0.35:
        st.warning("Flood Risk Level: MEDIUM")
    else:
        st.success("Flood Risk Level: LOW")

    st.caption(
        "Prediction is based on aggregated seasonal rainfall features. "
    )

    # Explainability
    st.subheader("🔍 Model Explainability")

    importance_df = get_feature_importance(model, FEATURE_NAMES)
    importance_df["Importance (%)"] = importance_df["Importance"] * 100

    fig = px.bar(
        importance_df,
        x="Importance (%)",
        y="Feature",
        orientation="h",
        title="Feature Importance in Flood Risk Prediction"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# 📊 PAGE 2 — RAINFALL ANALYSIS
elif page == "📊 Rainfall Analysis":

    st.title("📊 Rainfall Trends & Analysis")

    st.info(
        "This page provides **exploratory analysis** of historical rainfall data. "
    )

    # Monthly statistics
    avg_rainfall = state_data[month].mean()
    max_rainfall = state_data[month].max()
    min_rainfall = state_data[month].min()
    total_rainfall = state_data[month].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Rainfall", f"{avg_rainfall:.2f} mm")
    col2.metric("Maximum Rainfall", f"{max_rainfall:.2f} mm")
    col3.metric("Minimum Rainfall", f"{min_rainfall:.2f} mm")
    col4.metric("Total Rainfall", f"{total_rainfall:.2f} mm")

    # Monthly trend
    fig1 = px.line(
        state_data,
        x="YEAR",
        y=month,
        title=f"{month} Rainfall Trend in {state}"
    )
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # Monthly averages
    monthly_avg = state_data.iloc[:, 2:14].mean()

    fig2 = px.bar(
        x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        y=monthly_avg,
        title=f"Average Monthly Rainfall in {state}",
        color=monthly_avg,
        color_continuous_scale="Blues"
    )
    fig2.update_layout(template="plotly_dark", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Period rainfall trend
    state_data["period_rainfall"] = compute_period_rainfall(state_data, period)

    fig3 = px.bar(
        state_data,
        x="YEAR",
        y="period_rainfall",
        title=f"{period} Cumulative Rainfall Over Years",
        color="period_rainfall",
        color_continuous_scale="Viridis"
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)
