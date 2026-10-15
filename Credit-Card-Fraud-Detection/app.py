
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
)

# Load the trained model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "saved_models"
    / "tuned_xgboost.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Credit Card Fraud Detection")
st.write("Enter the transaction features and click Predict.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Check saved_models/tuned_xgboost.joblib.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Input form
with st.form("transaction_form"):
    time = st.number_input(
        "Time (seconds)", min_value=0.0, value=0.0
    )

    amount = st.number_input(
        "Transaction Amount", min_value=0.0, value=100.0
    )

    st.subheader("Additional transaction features")
    st.caption("Enter the actual V1–V28 values from your dataset.")

    features = {}

    # Two simple columns to keep the form manageable
    left, right = st.columns(2)

    for i in range(1, 29):
        with left if i % 2 == 1 else right:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}",
            )

    submitted = st.form_submit_button("Predict Transaction")

if submitted:
    input_data = {
        "Time": time,
        **features,
        "Amount": amount,
    }

    input_df = pd.DataFrame([input_data])

    try:
        # Apply the same approximate Amount clipping used in training
        input_df["Amount"] = input_df["Amount"].clip(
            lower=-102.55,
            upper=186.09,
        )

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]
        fraud_index = list(model.classes_).index(1)
        fraud_score = float(probabilities[fraud_index])

        st.subheader("Prediction")

        if prediction == 1:
            st.error("⚠️ Potential Fraud Detected")
        else:
            st.success("✅ Classified as Normal")

        st.metric(
            "Estimated Fraud Score",
            f"{fraud_score * 100:.2f}%",
        )

        st.caption(
            "This is a model prediction, not confirmation of fraud. "
            "The score may not represent the real-world probability."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
