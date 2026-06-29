import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load model
# -----------------------
model = joblib.load("models/fraud_pipeline.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Online Transaction Fraud Detection")
st.write("Predict whether a transaction is fraudulent.")

# -----------------------
# User Inputs
# -----------------------

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

payment_method = st.selectbox(
    "Payment Method",
    ["UPI", "Card", "Wallet", "Net Banking"]
)

authentication_method = st.selectbox(
    "Authentication",
    ["OTP", "PIN", "Biometric", "Password"]
)

device_change = st.selectbox(
    "Device Changed?",
    ["No", "Yes"]
)

location_change = st.selectbox(
    "Location Changed?",
    ["No", "Yes"]
)

is_weekend = st.selectbox(
    "Weekend Transaction?",
    ["No", "Yes"]
)

hour = st.slider("Hour of Transaction", 0, 23, 12)

ip_risk_score = st.slider(
    "IP Risk Score",
    0,
    100,
    20
)

device_risk_score = st.slider(
    "Device Risk Score",
    0,
    100,
    20
)

average_amount = st.number_input(
    "Customer Average Transaction Amount",
    value=amount
)

# -----------------------
# Feature Engineering
# -----------------------

amount_deviation_diff = amount - average_amount

amount_deviation_ratio = (
    amount / average_amount
    if average_amount != 0
    else 1
)

ip_device_risk_interaction = (
    ip_risk_score * device_risk_score
)

combined_change_flag = int(
    device_change == "Yes"
    and location_change == "Yes"
)

weekend_night_flag = int(
    is_weekend == "Yes"
    and (hour >= 22 or hour <= 5)
)

weak_auth_flag = int(
    authentication_method in ["OTP", "Password"]
)

high_risk_payment_flag = int(
    payment_method in ["Wallet", "UPI"]
)

combined_risk_index = (
    ip_risk_score
    + device_risk_score
)

# -----------------------
# DataFrame
# -----------------------

input_df = pd.DataFrame({

    "amount":[amount],

    "paymentmethod":[payment_method],

    "authenticationmethod":[authentication_method],

    "devicechange":[device_change],

    "locationchange":[location_change],

    "hour":[hour],

    "ipriskscore":[ip_risk_score],

    "deviceriskscore":[device_risk_score],

    "amount_deviation_diff":[amount_deviation_diff],

    "amount_deviation_ratio":[amount_deviation_ratio],

    "ip_device_risk_interaction":[ip_device_risk_interaction],

    "combined_change_flag":[combined_change_flag],

    "weekend_night_flag":[weekend_night_flag],

    "weak_auth_flag":[weak_auth_flag],

    "high_risk_payment_flag":[high_risk_payment_flag],

    "combined_risk_index":[combined_risk_index]

})

# -----------------------
# Prediction
# -----------------------

if st.button("Predict"):

    pred = model.predict(input_df)[0]

    prob = model.predict_proba(input_df)[0][1]

    st.metric(
        "Fraud Probability",
        f"{prob*100:.2f}%"
    )

    if pred == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")
