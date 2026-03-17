import sys
import os

# Adds the root directory to the system path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
from src.data_processor import load_and_merge_data
from src.predictive_model import load_model, save_model, train_cost_predictor


# Set Page Config for Professional Look
st.set_page_config(page_title="Community Health Equity Tracker", layout="wide")

st.title("🏥 Community Health Equity & Social Needs Tracker")
st.markdown("Analyzing Vertical Equity and Intersectional Health Disparities in California.")

# Paths (avoid dependence on current working directory)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cost_predictor.pkl"


@st.cache_data(show_spinner=False)
def get_data():
    return load_and_merge_data()


@st.cache_resource(show_spinner=False)
def get_or_train_model(data: pd.DataFrame):
    """
    Load the trained model if present; otherwise train and persist it.
    """
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    result = train_cost_predictor(data)
    save_model(result.model, MODEL_PATH)
    return result.model


# Load Data and Model
data, report = get_data()
model = get_or_train_model(data)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Analytics")
selected_city = st.sidebar.selectbox("Select City", options=sorted(data['CITY'].unique()))
selected_race = st.sidebar.multiselect("Select Race", options=data['RACE'].unique(), default=data['RACE'].unique())
selected_gender = st.sidebar.multiselect(
    "Select Gender", options=data["GENDER"].unique(), default=data["GENDER"].unique()
)

# --- MAIN DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Intersectional Cost Burden")
    # Displays the $1,350 vs $895 gap you identified in preliminary reporting
    filtered_report = report[
        (report["RACE"].isin(selected_race)) & (report["GENDER"].isin(selected_gender))
    ]
    chart = (
        alt.Chart(filtered_report)
        .mark_bar()
        .encode(
            x=alt.X("RACE:N", sort="-y", title="Race"),
            y=alt.Y("TOTAL_CLAIM_COST:Q", title="Average Total Claim Cost"),
            color=alt.Color("GENDER:N", title="Gender"),
            tooltip=["RACE", "GENDER", alt.Tooltip("TOTAL_CLAIM_COST:Q", format=",.2f"), "Encounter_Count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Average Claim Cost by Race/Demographic Segment")

with col2:
    st.subheader("🔮 Predictive Risk Tool")
    st.markdown("Input demographics to predict financial burden.")
    age = st.number_input("Age", min_value=0, max_value=110, value=45)
    income = st.number_input("Annual Income ($)", min_value=0, value=25000)
    race = st.selectbox("Race", options=sorted(data["RACE"].dropna().unique().tolist()))
    gender = st.selectbox("Gender", options=sorted(data["GENDER"].dropna().unique().tolist()))
    
    # Real prediction using the trained pipeline
    if st.button("Predict Healthcare Cost"):
        features = pd.DataFrame(
            [
                {
                    "AGE": age,
                    "INCOME": income,
                    "RACE": race,
                    "GENDER": gender,
                }
            ]
        )
        pred = float(model.predict(features)[0])
        st.success(f"Predicted Encounter Claim Cost: ${pred:,.2f}")
        st.caption(
            "This is trained from the dataset's `TOTAL_CLAIM_COST` per encounter; "
            "it is not a clinical risk score."
        )

# --- GEOGRAPHIC CLUSTERS ---
st.subheader(f"📍 Disease Clusters in {selected_city}")
city_data = data[data['CITY'] == selected_city]
# Identifies clusters like Hypertension in LA (312 cases)
st.write(city_data['DESCRIPTION'].value_counts().head(5))
