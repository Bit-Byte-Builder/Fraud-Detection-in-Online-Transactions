import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained pipeline
# ----------------------------
model = joblib.load("models/fraud_pipeline.pkl")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is fraudulent.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    [
        "UPI",
        "Card",
        "Net Banking",
        "Wallet",
        "ATM"
    ]
)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    [
        "Retail",
        "Grocery",
        "Travel",
        "Entertainment",
        "Electronics",
        "Food"
    ]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Debit Card",
        "Credit Card",
        "UPI",
        "Wallet",
        "Net Banking"
    ]
)

device_type = st.sidebar.selectbox(
    "Device Type",
    [
        "Android",
        "iOS",
        "Desktop"
    ]
)

authentication = st.sidebar.selectbox(
    "Authentication Method",
    [
        "PIN",
        "OTP",
        "Biometric"
    ]
)

risk_score = st.sidebar.slider(
    "Risk Score",
    0.0,
    100.0,
    25.0
)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

# --------------------------------------------------
# NOTE:
# Replace the columns below with the exact feature
# names used in your training notebook.
# --------------------------------------------------

input_df = pd.DataFrame({
    "amount": [amount],
    "transactiontype": [transaction_type],
    "merchantcategory": [merchant_category],
    "paymentmethod": [payment_method],
    "devicetype": [device_type],
    "authenticationmethod": [authentication],
    "riskscore": [risk_score],
    "hour": [hour],
    "dayofweek": [day_of_week]
})

if st.button("Predict Fraud"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.subheader("Business Decision")

    if probability >= 0.80:
        st.error("🚫 Hard Block")
    elif probability >= 0.60:
        st.warning("📞 Soft Review")
    elif probability >= 0.30:
        st.info("👀 Monitor")
    else:
        st.success("✅ Approve")
