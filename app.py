import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
pipeline = joblib.load(os.path.join(BASE_DIR, "models", "fraudpipeline.pkl"))
metadata = joblib.load(os.path.join(BASE_DIR, "models", "modelmetadata.pkl"))
threshold = metadata.get("threshold", 0.55)

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection App")
st.write("Enter transaction details below or upload a CSV.")

input_data = {
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
    "isweekend": isweekend,
}

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(df)[0][1]
    pred = int(proba >= threshold)

    if pred == 1:
        st.error(f"Fraud likely. Risk score: {proba:.4f}")
    else:
        st.success(f"Transaction likely genuine. Risk score: {proba:.4f}")
