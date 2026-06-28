import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "models", "best_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
metadata = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.pkl"))

st.title("Online Transaction Fraud Detection")
st.write("Predict whether a transaction is fraudulent or genuine.")

threshold = metadata.get("optimal_threshold", 0.5) if isinstance(metadata, dict) else 0.5

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

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

        st.write("Input columns:", list(input_df.columns))
        st.write("Expected features:", feature_names)

        X = input_df.copy()

        if list(X.columns) != list(feature_names):
            missing = [c for c in feature_names if c not in X.columns]
            extra = [c for c in X.columns if c not in feature_names]
            st.error(f"Feature mismatch. Missing: {missing}, Extra: {extra}")
            st.stop()

        X = X[feature_names]

        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

        try:
            proba = model.predict_proba(X_scaled)[0][1]
            pred = 1 if proba >= threshold else 0
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Result")
        st.success("Fraud" if pred == 1 else "Genuine")
        st.write(f"Fraud probability: {proba:.2%}")
        st.write(f"Decision threshold: {threshold:.4f}")

with tab2:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write("Uploaded columns:", list(batch_df.columns))

        missing = [c for c in feature_names if c not in batch_df.columns]
        extra = [c for c in batch_df.columns if c not in feature_names]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X_batch = batch_df[feature_names].copy()

        try:
            X_batch_scaled = scaler.transform(X_batch)
            probs = model.predict_proba(X_batch_scaled)[:, 1]
            preds = (probs >= threshold).astype(int)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            st.stop()

        result_df = batch_df.copy()
        result_df["fraud_probability"] = probs
        result_df["prediction"] = preds
        result_df["prediction_label"] = result_df["prediction"].map({0: "Genuine", 1: "Fraud"})

        st.dataframe(result_df.head())

        st.download_button(
            "Download Results",
            result_df.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv"
        )
