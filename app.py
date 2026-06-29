import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Trained Pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_pipeline.pkl")

model = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("💳 AI-Powered Financial Fraud Detection System")

st.markdown("""
This application predicts whether a financial transaction is **Fraudulent** or **Legitimate**
using a Machine Learning model trained on historical transaction data.
""")

st.divider()

# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("Transaction Information")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

paymentmethod = st.sidebar.selectbox(
    "Payment Method",
    ["CARD", "UPI", "BANK_TRANSFER", "WALLET"]
)

merchantcategory = st.sidebar.selectbox(
    "Merchant Category",
    [
        "RETAIL",
        "FOOD",
        "TRAVEL",
        "ELECTRONICS",
        "GROCERY",
        "HEALTHCARE",
        "ENTERTAINMENT"
    ]
)

authenticationmethod = st.sidebar.selectbox(
    "Authentication Method",
    [
        "OTP",
        "PIN",
        "BIOMETRIC",
        "NONE"
    ]
)

isinternational = st.sidebar.selectbox(
    "International Transaction",
    [0,1]
)

hourofday = st.sidebar.slider(
    "Hour of Transaction",
    0,
    23,
    12
)

dayofweek = st.sidebar.slider(
    "Day of Week",
    0,
    6,
    2
)

isweekend = 1 if dayofweek in [5,6] else 0

# ============================================================
# ADVANCED SETTINGS
# ============================================================

with st.sidebar.expander("Advanced Parameters"):

    customerid = st.number_input(
        "Customer ID",
        value=10001
    )

    deviceid = st.number_input(
        "Device ID",
        value=20001
    )

    merchantid = st.number_input(
        "Merchant ID",
        value=30001
    )

    ipaddressriskscore = st.slider(
        "IP Address Risk Score",
        0.0,
        1.0,
        0.20
    )

    devicetrustscore = st.slider(
        "Device Trust Score",
        0.0,
        1.0,
        0.90
    )

    txncountlast24h = st.number_input(
        "Transactions (Last 24h)",
        value=3
    )

    avgamountlast24h = st.number_input(
        "Average Amount (Last 24h)",
        value=3500.0
    )

    merchantdiversitylast7d = st.number_input(
        "Merchant Diversity (Last 7d)",
        value=2
    )

    devicechangeflag = st.selectbox(
        "Device Changed",
        [0,1]
    )

    locationchangeflag = st.selectbox(
        "Location Changed",
        [0,1]
    )

    otpsuccessratecustomer = st.slider(
        "OTP Success Rate",
        0.0,
        1.0,
        0.95
    )

    pastfraudcountcustomer = st.number_input(
        "Past Fraud Count",
        value=0
    )

    pastdisputescustomer = st.number_input(
        "Past Disputes",
        value=0
    )

    merchanthistoricalfraudrate = st.slider(
        "Merchant Historical Fraud Rate",
        0.0,
        1.0,
        0.05
    )

# ============================================================
# FEATURE ENGINEERING
# ============================================================

amount_deviation_ratio = amount / (avgamountlast24h + 1e-6)

amount_deviation_diff = amount - avgamountlast24h

ip_device_risk_interaction = (
    ipaddressriskscore * (1 - devicetrustscore)
)

combined_change_flag = (
    devicechangeflag + locationchangeflag
)

weekend_night_flag = int(
    isweekend == 1 and hourofday in [23, 0, 1, 2, 3, 4]
)

weak_auth_flag = int(
    authenticationmethod.upper() == "NONE"
)

high_risk_payment_flag = int(
    paymentmethod.upper() == "CARD"
    and isinternational == 1
)

combined_risk_index = np.mean([
    ipaddressriskscore,
    merchanthistoricalfraudrate,
    pastfraudcountcustomer,
    devicechangeflag,
    locationchangeflag
])

# ============================================================
# CREATE MODEL INPUT
# (Must match training feature order exactly)
# ============================================================

input_df = pd.DataFrame([{

    "customerid": customerid,
    "deviceid": deviceid,
    "merchantid": merchantid,

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

    "amount_deviation_ratio": amount_deviation_ratio,

    "amount_deviation_diff": amount_deviation_diff,

    "ip_device_risk_interaction": ip_device_risk_interaction,

    "combined_change_flag": combined_change_flag,

    "weekend_night_flag": weekend_night_flag,

    "weak_auth_flag": weak_auth_flag,

    "high_risk_payment_flag": high_risk_payment_flag,

    "combined_risk_index": combined_risk_index

}])

# Preview
with st.expander("Preview Model Input"):
    st.dataframe(input_df)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Fraud", use_container_width=True):

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get prediction probabilities
    proba = model.predict_proba(input_df)[0]

    # Probability of Fraud class
    if 1 in model.classes_:
        fraud_index = list(model.classes_).index(1)
        probability = proba[fraud_index]
    else:
        probability = proba[0]

    st.divider()

    st.header("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Genuine Transaction")

    with col2:
        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col3:
        st.metric(
            "Model Confidence",
            f"{max(probability, 1 - probability):.2%}"
        )
    # ------------------------------------------
    # Risk Level
    # ------------------------------------------

    if probability < 0.30:
        risk = "🟢 LOW"

    elif probability < 0.60:
        risk = "🟡 MEDIUM"

    elif probability < 0.85:
        risk = "🟠 HIGH"

    else:
        risk = "🔴 VERY HIGH"

    st.subheader("Risk Level")
    st.info(risk)

    # ------------------------------------------
    # Probability Bar
    # ------------------------------------------

    st.subheader("Fraud Probability")

    st.progress(float(probability))

    st.write(f"Probability = **{probability:.2%}**")

    # ------------------------------------------
    # Business Recommendation
    # ------------------------------------------

    st.subheader("Recommended Action")

    if probability < 0.30:

        st.success("""
✅ APPROVE TRANSACTION

Transaction appears legitimate.

No further verification required.
""")

    elif probability < 0.60:

        st.warning("""
⚠ REVIEW TRANSACTION

Ask customer for OTP verification.

Continue monitoring.
""")

    elif probability < 0.85:

        st.warning("""
🟠 HIGH RISK

Perform manual verification.

Temporary hold recommended.
""")

    else:

        st.error("""
🚫 BLOCK TRANSACTION

Very high fraud probability.

Escalate to fraud investigation team.
""")

    # ------------------------------------------
    # Display Engineered Features
    # ------------------------------------------

    with st.expander("Engineered Features"):

        engineered = pd.DataFrame({

            "Feature":[
                "Amount Deviation Ratio",
                "Amount Deviation Difference",
                "IP × Device Risk",
                "Combined Change Flag",
                "Weekend Night Flag",
                "Weak Authentication",
                "High Risk Payment",
                "Combined Risk Index"
            ],

            "Value":[
                amount_deviation_ratio,
                amount_deviation_diff,
                ip_device_risk_interaction,
                combined_change_flag,
                weekend_night_flag,
                weak_auth_flag,
                high_risk_payment_flag,
                combined_risk_index
            ]

        })

        st.dataframe(
            engineered,
            use_container_width=True
        )

# ============================================================
# DOWNLOAD INPUT DATA
# ============================================================

st.divider()

st.subheader("Download Input Data")

csv = input_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Input Features (CSV)",
    data=csv,
    file_name="transaction_input.csv",
    mime="text/csv"
)

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("About This Model"):

    st.markdown("""
### Fraud Detection Pipeline

**Machine Learning Algorithm**
- Random Forest Classifier

**Features Used**
- 30 Features
- 3 Categorical
- 27 Numerical

**Feature Engineering**
- Amount Deviation Ratio
- Amount Deviation Difference
- IP × Device Risk Interaction
- Combined Change Flag
- Weekend Night Flag
- Weak Authentication Flag
- High-Risk Payment Flag
- Combined Risk Index

**Preprocessing**
- One-Hot Encoding
- Standard Scaling
- ColumnTransformer Pipeline

**Prediction**
- Fraud (1)
- Genuine (0)
""")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Developed by Sachin Kumar | "
    "AI-Powered Financial Fraud Detection Capstone Project"
)
