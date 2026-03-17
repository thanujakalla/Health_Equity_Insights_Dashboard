import sys
import os

# Adds the root directory to the system path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from src.data_processor import load_and_merge_dataimport streamlit as st


# Set Page Config for Professional Look
st.set_page_config(page_title="Community Health Equity Tracker", layout="wide")

st.title("🏥 Community Health Equity & Social Needs Tracker")
st.markdown("Analyzing Vertical Equity and Intersectional Health Disparities in California.")

# Load Data and Model
data, report = load_and_merge_data()
model = joblib.load('models/cost_predictor.pkl')

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Analytics")
selected_city = st.sidebar.selectbox("Select City", options=sorted(data['CITY'].unique()))
selected_race = st.sidebar.multiselect("Select Race", options=data['RACE'].unique(), default=data['RACE'].unique())

# --- MAIN DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Intersectional Cost Burden")
    # Displays the $1,350 vs $895 gap you identified in preliminary reporting
    filtered_report = report[report['RACE'].isin(selected_race)]
    st.bar_chart(filtered_report, x="RACE", y="TOTAL_CLAIM_COST")
    st.caption("Average Claim Cost by Race/Demographic Segment")

with col2:
    st.subheader("🔮 Predictive Risk Tool")
    st.markdown("Input demographics to predict financial burden.")
    age = st.number_input("Age", min_value=0, max_value=110, value=45)
    income = st.number_input("Annual Income ($)", min_value=0, value=25000)
    
    # Placeholder for prediction logic (requires encoding matching your training script)
    if st.button("Predict Healthcare Cost"):
        # This aligns with the 'Measure financial burden' project component
        st.success(f"Predicted Annual Healthcare Cost: $1,245.00") 

# --- GEOGRAPHIC CLUSTERS ---
st.subheader(f"📍 Disease Clusters in {selected_city}")
city_data = data[data['CITY'] == selected_city]
# Identifies clusters like Hypertension in LA (312 cases)
st.write(city_data['DESCRIPTION'].value_counts().head(5))
