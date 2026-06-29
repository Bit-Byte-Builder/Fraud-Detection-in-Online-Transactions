import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraudpipeline.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "modelmetadata.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection in Online Transactions")
st.write("Enter transaction details or upload a CSV to score fraud risk.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found at models/fraudpipeline.pkl")
    st.stop()

pipeline = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
threshold = metadata.get("threshold", 0.61)

feature_cols = [
    "amount",
    "paymentmethod",
    "isinternational",
    "merchantcategory",
    "ipaddressriskscore",
    "devicetrustscore",
    "txncountlast24h",
    "avgamountlast24h",
    "merchantdiversitylast7d",
    "devicechangeflag",
    "locationchangeflag",
    "authenticationmethod",
    "otpsuccessratecustomer",
    "pastfraudcountcustomer",
    "pastdisputescustomer",
    "merchanthistoricalfraudrate",
    "hourofday",
    "dayofweek",
    "isweekend",
]

st.header("Single transaction prediction")

with st.form("fraud_form"):
    c1, c2 = st.columns(2)

    with c1:
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=1.0)
        isinternational = st.selectbox("Is International", [0, 1], index=0)
        ipaddressriskscore = st.number_input("IP Address Risk Score", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        txncountlast24h = st.number_input("Txn Count Last 24h", min_value=0, value=2, step=1)
        merchantdiversitylast7d = st.number_input("Merchant Diversity Last 7d", min_value=0, value=2, step=1)
        locationchangeflag = st.selectbox("Location Change Flag", [0, 1], index=0)
        otpsuccessratecustomer = st.number_input("OTP Success Rate Customer", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        pastdisputescustomer = st.number_input("Past Disputes Customer", min_value=0, value=0, step=1)
        hourofday = st.slider("Hour of Day", 0, 23, 12)

    with c2:
        paymentmethod = st.selectbox("Payment Method", ["CARD", "UPI", "WALLET", "NETBANKING"])
        merchantcategory = st.selectbox("Merchant Category", ["Travel", "Electronics", "Fashion", "Utilities", "Gaming"])
        devicetrustscore = st.number_input("Device Trust Score", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        avgamountlast24h = st.number_input("Avg Amount Last 24h", min_value=0.0, value=500.0, step=1.0)
        devicechangeflag = st.selectbox("Device Change Flag", [0, 1], index=0)
        authenticationmethod = st.selectbox("Authentication Method", ["OTP", "PIN", "NONE"])
        pastfraudcountcustomer = st.number_input("Past Fraud Count Customer", min_value=0, value=0, step=1)
        merchanthistoricalfraudrate = st.number_input("Merchant Historical Fraud Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        dayofweek = st.slider("Day of Week", 0, 6, 2)
        isweekend = st.selectbox("Is Weekend", [0, 1], index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
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
    }])

    prob = float(pipeline.predict_proba(row)[0][1])
    pred = int(prob >= threshold)

    st.subheader("Result")
    if pred == 1:
        st.error(f"Fraud likely. Risk score: {prob:.4f}")
    else:
        st.success(f"Likely genuine. Risk score: {prob:.4f}")

    if prob >= 0.80:
        action = "Hard Block"
    elif prob >= 0.60:
        action = "Soft Review"
    elif prob >= 0.40:
        action = "OTP Challenge"
    else:
        action = "Approve"

    st.info(f"Recommended action: {action}")

st.header("CSV upload")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        scores = pipeline.predict_proba(df[feature_cols])[:, 1]
        out = df.copy()
        out["fraud_probability"] = scores
        out["predicted_fraud"] = (out["fraud_probability"] >= threshold).astype(int)
        out["recommended_action"] = out["fraud_probability"].apply(
            lambda p: "Hard Block" if p >= 0.80 else
                      "Soft Review" if p >= 0.60 else
                      "OTP Challenge" if p >= 0.40 else
                      "Approve"
        )
        st.dataframe(out.head(50))
        st.download_button(
            "Download scored CSV",
            out.to_csv(index=False).encode("utf-8"),
            "fraud_scored_output.csv",
            "text/csv"
        )
