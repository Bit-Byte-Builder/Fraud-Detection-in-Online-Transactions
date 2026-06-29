import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection App", layout="wide")

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/fraudpipeline.pkl")
    metadata = joblib.load("models/modelmetadata.pkl")
    return pipeline, metadata

pipeline, metadata = load_artifacts()

st.title("Fraud Detection in Online Transactions")
st.caption("Real-time fraud risk scoring for payment transactions")

st.sidebar.header("Transaction Details")

def input_float(label, value=0.0, min_value=0.0, max_value=1e9, step=0.01):
    return st.sidebar.number_input(label, value=float(value), min_value=float(min_value), max_value=float(max_value), step=float(step))

def input_int(label, value=0, min_value=0, max_value=1_000_000, step=1):
    return st.sidebar.number_input(label, value=int(value), min_value=int(min_value), max_value=int(max_value), step=int(step))

transaction_id = input_int("Transaction ID", 1)
customer_id = input_int("Customer ID", 1000)
device_id = input_int("Device ID", 20000)
merchant_id = input_int("Merchant ID", 500)

amount = input_float("Amount", 1000.0, 0.0, 1000000.0, 0.01)
is_international = st.sidebar.selectbox("International Transaction", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
payment_method = st.sidebar.selectbox("Payment Method", ["CARD", "UPI", "WALLET", "NETBANKING"])
merchant_category = st.sidebar.selectbox("Merchant Category", ["Electronics", "Fashion", "Grocery", "Gaming", "Utilities", "Travel"])
ip_address_risk_score = input_float("IP Address Risk Score", 0.2, 0.0, 1.0, 0.01)
device_trust_score = input_float("Device Trust Score", 0.8, 0.0, 1.0, 0.01)
txn_count_last_24h = input_int("Txn Count Last 24h", 3, 0, 1000, 1)
avg_amount_last_24h = input_float("Avg Amount Last 24h", 900.0, 0.0, 1000000.0, 0.01)
merchant_diversity_last_7d = input_int("Merchant Diversity Last 7d", 2, 0, 1000, 1)
device_change_flag = st.sidebar.selectbox("Device Change Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
location_change_flag = st.sidebar.selectbox("Location Change Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
authentication_method = st.sidebar.selectbox("Authentication Method", ["OTP", "PIN", "3DS", "NONE"])
otp_success_rate_customer = input_float("OTP Success Rate Customer", 0.7, 0.0, 1.0, 0.01)
past_fraud_count_customer = input_int("Past Fraud Count Customer", 0, 0, 1000, 1)
past_disputes_customer = input_int("Past Disputes Customer", 0, 0, 1000, 1)
merchant_historical_fraud_rate = input_float("Merchant Historical Fraud Rate", 0.05, 0.0, 1.0, 0.001)
hour_of_day = input_int("Hour of Day", 12, 0, 23, 1)
day_of_week = input_int("Day of Week", 2, 0, 6, 1)
is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

input_df = pd.DataFrame([{
    "transactionid": transaction_id,
    "customerid": customer_id,
    "deviceid": device_id,
    "merchantid": merchant_id,
    "amount": amount,
    "paymentmethod": payment_method,
    "isinternational": is_international,
    "merchantcategory": merchant_category,
    "ipaddressriskscore": ip_address_risk_score,
    "devicetrustscore": device_trust_score,
    "txncountlast24h": txn_count_last_24h,
    "avgamountlast24h": avg_amount_last_24h,
    "merchantdiversitylast7d": merchant_diversity_last_7d,
    "devicechangeflag": device_change_flag,
    "locationchangeflag": location_change_flag,
    "authenticationmethod": authentication_method,
    "otpsuccessratecustomer": otp_success_rate_customer,
    "pastfraudcountcustomer": past_fraud_count_customer,
    "pastdisputescustomer": past_disputes_customer,
    "merchanthistoricalfraudrate": merchant_historical_fraud_rate,
    "hourofday": hour_of_day,
    "dayofweek": day_of_week,
    "isweekend": is_weekend
}])

st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Fraud Risk"):
    try:
        proba = pipeline.predict_proba(input_df)[:, 1][0]
        threshold = metadata.get("threshold", 0.61)
        prediction = int(proba >= threshold)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{proba:.4f}")
        col2.metric("Threshold", f"{threshold:.2f}")
        col3.metric("Prediction", "Fraud" if prediction == 1 else "Legitimate")

        if prediction == 1:
            st.error("High fraud risk detected.")
        else:
            st.success("Transaction appears legitimate.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("### Notes")
st.write("The app uses the saved fraud pipeline and model metadata from the notebook.")
