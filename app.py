import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Online Transaction Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Online Transaction Fraud Detection")
st.write("Predict whether an online transaction is fraudulent.")

# Load model
MODEL_PATH = "models/fraud_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

model = joblib.load(MODEL_PATH)

st.header("Transaction Details")

customerid = st.number_input("Customer ID", value=1000)
deviceid = st.number_input("Device ID", value=20000)
merchantid = st.number_input("Merchant ID", value=500)

merchantcategory = st.selectbox(
    "Merchant Category",
    [
        "Electronics",
        "Fashion",
        "Grocery",
        "Travel",
        "Food",
        "Healthcare",
        "Entertainment"
    ]
)

isinternational = st.selectbox(
    "International Transaction?",
    [0, 1]
)

isweekend = st.selectbox(
    "Weekend Transaction?",
    [0, 1]
)

hourofday = st.slider(
    "Hour of Day",
    0,
    23,
    12
)

dayofweek = st.slider(
    "Day of Week (0=Monday)",
    0,
    6,
    2
)

avgamountlast24h = st.number_input(
    "Average Amount (Last 24 Hours)",
    value=1000.0
)

txncountlast24h = st.number_input(
    "Transactions in Last 24 Hours",
    value=5
)

merchanthistoricalfraudrate = st.slider(
    "Merchant Historical Fraud Rate",
    0.0,
    1.0,
    0.05
)

pastfraudcountcustomer = st.number_input(
    "Past Fraud Count",
    value=0
)

pastdisputescustomer = st.number_input(
    "Past Disputes",
    value=0
)

devicetrustscore = st.slider(
    "Device Trust Score",
    0,
    100,
    80
)

ipaddressriskscore = st.slider(
    "IP Address Risk Score",
    0,
    100,
    20
)

otpsuccessratecustomer = st.slider(
    "OTP Success Rate",
    0.0,
    1.0,
    0.95
)

merchantdiversitylast7d = st.number_input(
    "Merchant Diversity (Last 7 Days)",
    value=3
)

locationchangeflag = st.selectbox(
    "Location Changed?",
    [0, 1]
)

devicechangeflag = st.selectbox(
    "Device Changed?",
    [0, 1]
)

if st.button("Predict Fraud"):

    input_df = pd.DataFrame([{
        "customerid": customerid,
        "deviceid": deviceid,
        "merchantid": merchantid,
        "merchantcategory": merchantcategory,
        "isinternational": isinternational,
        "isweekend": isweekend,
        "hourofday": hourofday,
        "dayofweek": dayofweek,
        "avgamountlast24h": avgamountlast24h,
        "txncountlast24h": txncountlast24h,
        "merchanthistoricalfraudrate": merchanthistoricalfraudrate,
        "pastfraudcountcustomer": pastfraudcountcustomer,
        "pastdisputescustomer": pastdisputescustomer,
        "devicetrustscore": devicetrustscore,
        "ipaddressriskscore": ipaddressriskscore,
        "otpsuccessratecustomer": otpsuccessratecustomer,
        "merchantdiversitylast7d": merchantdiversitylast7d,
        "locationchangeflag": locationchangeflag,
        "devicechangeflag": devicechangeflag
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

    st.metric(
        "Fraud Probability",
        f"{probability*100:.2f}%"
    )
