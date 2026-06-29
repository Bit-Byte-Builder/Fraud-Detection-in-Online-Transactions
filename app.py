import os
import joblib
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
pipeline = joblib.load(os.path.join(BASE_DIR, "models", "fraud_pipeline.pkl"))
metadata = joblib.load(os.path.join(BASE_DIR, "models", "model_metadata.pkl"))
threshold = metadata.get("threshold", 0.61)

st.title("Fraud Detection App")

st.write("Enter transaction details below or upload a CSV.")

# Example: add your input widgets here using the same columns your model expects
