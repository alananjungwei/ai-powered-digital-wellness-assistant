import streamlit as st
import joblib
import pandas as pd

# Load models
ada_model = joblib.load("high_risk_classifier.pkl")
linreg = joblib.load("digital_dependence_regressor.pkl")
scaler = joblib.load("digital_dependence_scaler.pkl")

st.title("Digital Wellness Assistant")

st.success("Models loaded successfully!")

# User inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

device_hours_per_day = st.slider(
    "Device Hours Per Day",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

phone_unlocks = st.number_input(
    "Phone Unlocks Per Day",
    min_value=0,
    max_value=500,
    value=50
)

sleep_hours = st.slider(
    "Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

# Button 

if st.button("Analyze"):

    user_df = pd.DataFrame({
        "age": [age],
        "device_hours_per_day": [device_hours_per_day],
        "phone_unlocks": [phone_unlocks],
        "sleep_hours": [sleep_hours]
    })

    st.subheader("User Data")

    st.dataframe(user_df)