import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection App", layout="wide")

MODEL_PATH = "fraud_model.pkl"

EXPECTED_COLUMNS = [
    'amount',
    'is_international',
    'ip_address_risk_score',
    'device_trust_score',
    'txn_count_last_24h',
    'avg_amount_last_24h',
    'merchant_diversity_last_7d',
    'device_change_flag',
    'location_change_flag',
    'otp_success_rate_customer',
    'past_fraud_count_customer',
    'past_disputes_customer',
    'merchant_historical_fraud_rate',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'amount_deviation',
    'amount_ratio',
    'high_amount_flag',
    'low_amount_flag',
    'combined_risk_score',
    'risky_hour_flag',
    'high_velocity_flag',
    'multi_merchant_flag',
    'risk_flag_sum',
    'fraud_history_score',
    'otp_device_interaction',
    'payment_method_NETBANKING',
    'payment_method_UPI',
    'payment_method_WALLET',
    'merchant_category_Fashion',
    'merchant_category_Gaming',
    'merchant_category_Grocery',
    'merchant_category_Travel',
    'merchant_category_Utilities',
    'authentication_method_NONE',
    'authentication_method_OTP',
    'authentication_method_PIN',
    'amount_bucket_Low',
    'amount_bucket_Medium',
    'amount_bucket_High',
    'amount_bucket_Very High'
]

PAYMENT_METHODS = ["CARD", "NETBANKING", "UPI", "WALLET"]
MERCHANT_CATEGORIES = ["Electronics", "Gaming", "Travel", "Grocery", "Fashion", "Utilities"]
AUTH_METHODS = ["3DS", "NONE", "OTP", "PIN"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        'paymentmethod': 'payment_method',
        'isinternational': 'is_international',
        'merchantcategory': 'merchant_category',
        'ipaddressriskscore': 'ip_address_risk_score',
        'devicetrustscore': 'device_trust_score',
        'txncountlast24h': 'txn_count_last_24h',
        'avgamountlast24h': 'avg_amount_last_24h',
        'merchantdiversitylast7d': 'merchant_diversity_last_7d',
        'devicechangeflag': 'device_change_flag',
        'locationchangeflag': 'location_change_flag',
        'authenticationmethod': 'authentication_method',
        'otpsuccessratecustomer': 'otp_success_rate_customer',
        'pastfraudcountcustomer': 'past_fraud_count_customer',
        'pastdisputescustomer': 'past_disputes_customer',
        'merchanthistoricalfraudrate': 'merchant_historical_fraud_rate',
        'hourofday': 'hour_of_day',
        'dayofweek': 'day_of_week',
        'isweekend': 'is_weekend'
    }

    df.rename(columns=rename_map, inplace=True)

    required_raw = [
        'amount',
        'payment_method',
        'is_international',
        'merchant_category',
        'ip_address_risk_score',
        'device_trust_score',
        'txn_count_last_24h',
        'avg_amount_last_24h',
        'merchant_diversity_last_7d',
        'device_change_flag',
        'location_change_flag',
        'authentication_method',
        'otp_success_rate_customer',
        'past_fraud_count_customer',
        'past_disputes_customer',
        'merchant_historical_fraud_rate',
        'hour_of_day',
        'day_of_week',
        'is_weekend'
    ]

    missing = [col for col in required_raw if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    df['amount_deviation'] = df['amount'] - df['avg_amount_last_24h']
    df['amount_ratio'] = df['amount'] / df['avg_amount_last_24h'].replace(0, 1)
    df['high_amount_flag'] = (df['amount'] > 15000).astype(int)
    df['low_amount_flag'] = (df['amount'] < 500).astype(int)
    df['combined_risk_score'] = df['ip_address_risk_score'] + (1 - df['device_trust_score'])
    df['risky_hour_flag'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df['high_velocity_flag'] = (df['txn_count_last_24h'] > 8).astype(int)
    df['multi_merchant_flag'] = (df['merchant_diversity_last_7d'] > 7).astype(int)
    df['risk_flag_sum'] = (
        df['device_change_flag'] +
        df['location_change_flag'] +
        df['is_international']
    )
    df['fraud_history_score'] = (
        df['past_fraud_count_customer'] +
        df['past_disputes_customer']
    )
    df['otp_device_interaction'] = (
        df['otp_success_rate_customer'] * df['device_trust_score']
    )

    df['amount_bucket'] = pd.cut(
        df['amount'],
        bins=[0, 1000, 5000, 10000, 15000, 20000],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    cat_columns = ['payment_method', 'merchant_category', 'authentication_method', 'amount_bucket']
    df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True, dtype=int)

    for col in EXPECTED_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[EXPECTED_COLUMNS]
    return df_encoded


def predict_single(model, input_dict):
    input_df = pd.DataFrame([input_dict])
    processed_df = preprocess_input(input_df)
    prediction = model.predict(processed_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(processed_df)[0][1]
    else:
        probability = None

    return prediction, probability, processed_df


st.title("PaySphere Fraud Detection")
st.write("Predict whether an online transaction is fraudulent or genuine.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model file: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
        paymentmethod = st.selectbox("Payment Method", PAYMENT_METHODS)
        isinternational = st.selectbox("Is International", [0, 1])
        merchantcategory = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
        ipaddressriskscore = st.slider("IP Address Risk Score", 0.0, 1.0, 0.50, 0.01)
        devicetrustscore = st.slider("Device Trust Score", 0.0, 1.0, 0.50, 0.01)
        txncountlast24h = st.number_input("Txn Count Last 24h", min_value=0, max_value=50, value=5)

    with col2:
        avgamountlast24h = st.number_input("Avg Amount Last 24h", min_value=0.0, value=4000.0, step=100.0)
        merchantdiversitylast7d = st.number_input("Merchant Diversity Last 7d", min_value=0, max_value=50, value=3)
        devicechangeflag = st.selectbox("Device Change Flag", [0, 1])
        locationchangeflag = st.selectbox("Location Change Flag", [0, 1])
        authenticationmethod = st.selectbox("Authentication Method", AUTH_METHODS)
        otpsuccessratecustomer = st.slider("OTP Success Rate Customer", 0.0, 1.0, 0.70, 0.01)
        pastfraudcountcustomer = st.number_input("Past Fraud Count Customer", min_value=0, max_value=20, value=0)

    with col3:
        pastdisputescustomer = st.number_input("Past Disputes Customer", min_value=0, max_value=20, value=1)
        merchanthistoricalfraudrate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, 0.05, 0.001)
        hourofday = st.slider("Hour of Day", 0, 23, 12)
        dayofweek = st.slider("Day of Week", 0, 6, 2)
        isweekend = st.selectbox("Is Weekend", [0, 1])

    user_input = {
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
    }

    if st.button("Predict Fraud"):
        try:
            pred, proba, processed = predict_single(model, user_input)

            st.markdown("### Prediction Result")
            if pred == 1:
                st.error("Fraudulent Transaction Detected")
            else:
                st.success("Genuine Transaction")

            if proba is not None:
                st.metric("Fraud Probability", f"{proba:.2%}")

            with st.expander("Show Processed Model Features"):
                st.dataframe(processed)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch Prediction via CSV Upload")

    st.write(
        "Upload a CSV using raw notebook column names such as: "
        "`amount`, `paymentmethod`, `isinternational`, `merchantcategory`, "
        "`ipaddressriskscore`, `devicetrustscore`, `txncountlast24h`, "
        "`avgamountlast24h`, `merchantdiversitylast7d`, `devicechangeflag`, "
        "`locationchangeflag`, `authenticationmethod`, `otpsuccessratecustomer`, "
        "`pastfraudcountcustomer`, `pastdisputescustomer`, `merchanthistoricalfraudrate`, "
        "`hourofday`, `dayofweek`, `isweekend`."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            processed_batch = preprocess_input(batch_df)

            preds = model.predict(processed_batch)

            result_df = batch_df.copy()
            result_df["prediction"] = preds
            result_df["prediction_label"] = np.where(preds == 1, "Fraud", "Genuine")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(processed_batch)[:, 1]
                result_df["fraud_probability"] = probs

            st.success("Batch prediction completed.")
            st.dataframe(result_df.head(20))

            csv_output = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
