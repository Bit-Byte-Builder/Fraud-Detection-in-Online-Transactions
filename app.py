import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "models", "best_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
metadata = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.pkl"))

st.write("Input columns:", list(input_df.columns))
st.write("Expected features:", feature_names)

st.title("Online Transaction Fraud Detection")
st.write("Predict whether a transaction is fraudulent or genuine.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

feature_columns = [
    "amount", "paymentmethod", "isinternational", "merchantcategory",
    "ipaddressriskscore", "devicetrustscore", "txncountlast24h",
    "avgamountlast24h", "merchantdiversitylast7d", "devicechangeflag",
    "locationchangeflag", "authenticationmethod", "otpsuccessratecustomer",
    "pastfraudcountcustomer", "pastdisputescustomer",
    "merchanthistoricalfraudrate", "hourofday", "dayofweek", "isweekend"
]

with tab1:
    amount = st.number_input("Amount", min_value=0.0, value=5000.0)
    paymentmethod = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
    isinternational = st.selectbox("Is International", [0, 1])
    merchantcategory = st.selectbox("Merchant Category", ["Electronics", "Gaming", "Travel", "Grocery", "Fashion", "Utilities"])
    ipaddressriskscore = st.slider("IP Address Risk Score", 0.0, 1.0, 0.5)
    devicetrustscore = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
    txncountlast24h = st.number_input("Txn Count Last 24h", min_value=0, value=2)
    avgamountlast24h = st.number_input("Avg Amount Last 24h", min_value=0.0, value=3000.0)
    merchantdiversitylast7d = st.number_input("Merchant Diversity Last 7d", min_value=0, value=2)
    devicechangeflag = st.selectbox("Device Change Flag", [0, 1])
    locationchangeflag = st.selectbox("Location Change Flag", [0, 1])
    authenticationmethod = st.selectbox("Authentication Method", ["OTP", "3DS", "PIN", "NONE"])
    otpsuccessratecustomer = st.slider("OTP Success Rate", 0.0, 1.0, 0.7)
    pastfraudcountcustomer = st.number_input("Past Fraud Count", min_value=0, value=0)
    pastdisputescustomer = st.number_input("Past Disputes", min_value=0, value=0)
    merchanthistoricalfraudrate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, 0.05)
    hourofday = st.slider("Hour of Day", 0, 23, 12)
    dayofweek = st.slider("Day of Week", 0, 6, 2)
    isweekend = st.selectbox("Is Weekend", [0, 1])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "amount": amount,
            "paymentmethod": paymentmethod,
            "isinternational": isinternational,
            "merchantcategory": merchantcategory,
            "ipaddressriskscore": ipaddressriskscore,
            "devicetrustscore": devicetrustscore,
            "txncountlast24h": txncountlast24h,
            "avgamountlast24h": avgamountlast24h,
            "merchantdiversitylast7d": merchantdiversitylast7d,
            "devicechangeflag": devicechangeflag,
            "locationchangeflag": locationchangeflag,
            "authenticationmethod": authenticationmethod,
            "otpsuccessratecustomer": otpsuccessratecustomer,
            "pastfraudcountcustomer": pastfraudcountcustomer,
            "pastdisputescustomer": pastdisputescustomer,
            "merchanthistoricalfraudrate": merchanthistoricalfraudrate,
            "hourofday": hourofday,
            "dayofweek": dayofweek,
            "isweekend": isweekend
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Result")
        st.write("Fraud" if pred == 1 else "Genuine")
        st.write(f"Fraud probability: {proba:.2%}")

with tab2:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        preds = model.predict(batch_df)
        probs = model.predict_proba(batch_df)[:, 1]
        batch_df["prediction"] = preds
        batch_df["fraud_probability"] = probs
        st.dataframe(batch_df.head())
        st.download_button(
            "Download Results",
            batch_df.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv"
        )
