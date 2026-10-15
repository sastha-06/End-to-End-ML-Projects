
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "saved_models"
    / "tuned_xgboost.joblib"
)

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

AMOUNT_LOWER = -102.55
AMOUNT_UPPER = 186.09


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Credit Card Fraud Detection")
st.write(
    "Enter transaction features to predict potential fraud "
    "and understand how the model reached its prediction."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Check "
        "saved_models/tuned_xgboost.joblib."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# --------------------------------------------------
# Transaction input form
# --------------------------------------------------

with st.form("transaction_form"):

    time_value = st.number_input(
        "Time (seconds)",
        min_value=0.0,
        value=0.0,
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
    )

    st.subheader("Additional Transaction Features")
    st.caption(
        "Enter the actual V1–V28 values from your dataset "
        "for meaningful testing."
    )

    features = {}
    left, right = st.columns(2)

    for i in range(1, 29):
        column = left if i % 2 == 1 else right

        with column:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}",
            )

    submitted = st.form_submit_button("Predict Transaction")


# --------------------------------------------------
# Prediction and input-specific explanation
# --------------------------------------------------

if submitted:

    input_data = {
        "Time": time_value,
        **features,
        "Amount": amount,
    }

    input_df = pd.DataFrame([input_data], columns=FEATURES)

    try:
        # Apply the approximate Amount clipping limits
        input_df["Amount"] = input_df["Amount"].clip(
            lower=AMOUNT_LOWER,
            upper=AMOUNT_UPPER,
        )

        # Predict transaction class
        prediction = int(model.predict(input_df)[0])

        # Get estimated fraud probability
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Fraud class 1 was not found in the model.")

        fraud_index = classes.index(1)
        fraud_score = float(probabilities[fraud_index])

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ Potential Fraud Detected")
            else:
                st.success("✅ Classified as Normal")

        with col2:
            st.metric(
                "Estimated Fraud Score",
                f"{fraud_score * 100:.2f}%",
            )

        st.caption(
            "This is a model prediction, not confirmation of fraud. "
            "The score may not represent the real-world probability."
        )

        # ------------------------------------------
        # Calculate feature contributions
        # ------------------------------------------

        st.divider()
        st.subheader("Why Did the Model Make This Prediction?")

        booster = model.get_booster()

        dmatrix = xgb.DMatrix(
            input_df,
            feature_names=FEATURES,
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
        )[0]

        # Last contribution is the bias, not an input feature
        feature_contributions = contributions[:-1]

        explanation = pd.DataFrame({
            "Feature": FEATURES,
            "Your Input": input_df.iloc[0].to_numpy(),
            "Contribution": feature_contributions,
        })

        # Features pushing the score toward fraud
        toward_fraud = (
            explanation[explanation["Contribution"] > 0]
            .sort_values("Contribution", ascending=False)
            .head(5)
        )

        # Features pushing the score away from fraud
        away_from_fraud = (
            explanation[explanation["Contribution"] < 0]
            .sort_values("Contribution", ascending=True)
            .head(5)
        )

        st.write(
            "These results are calculated from the current transaction's "
            "input values using your trained XGBoost model."
        )

        st.markdown("### Features Pushing Toward Fraud")

        if not toward_fraud.empty:
            display_fraud = toward_fraud.copy()
            display_fraud["Your Input"] = (
                display_fraud["Your Input"].round(6)
            )
            display_fraud["Contribution"] = (
                display_fraud["Contribution"].round(4)
            )

            st.dataframe(
                display_fraud.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write(
                "No features had a positive contribution "
                "toward the Fraud class."
            )

        st.markdown("### Features Pushing Away from Fraud")

        if not away_from_fraud.empty:
            display_normal = away_from_fraud.copy()
            display_normal["Your Input"] = (
                display_normal["Your Input"].round(6)
            )
            display_normal["Contribution"] = (
                display_normal["Contribution"].round(4)
            )

            st.dataframe(
                display_normal.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write(
                "No features had a negative contribution "
                "away from the Fraud class."
            )

        # ------------------------------------------
        # Simple explanation
        # ------------------------------------------

        st.subheader("Simple Explanation")

        if prediction == 1:
            st.warning(
                f"The model classified this transaction as Potential "
                f"Fraud, with an estimated fraud score of "
                f"{fraud_score * 100:.2f}%. The first table shows the "
                "features pushing its score toward Fraud; the second "
                "shows features pushing it away."
            )
        else:
            st.success(
                f"The model classified this transaction as Normal, "
                f"with an estimated fraud score of "
                f"{fraud_score * 100:.2f}%. The combined feature "
                "contributions resulted in a Normal prediction."
            )

        st.caption(
            "Positive contributions push the raw model output toward "
            "Fraud; negative contributions push it away. These values "
            "are not percentage changes in probability and do not prove "
            "that any feature caused real-world fraud."
        )

        with st.expander("View all input values"):
            st.dataframe(
                input_df.T.rename(columns={0: "Input Value"}),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Prediction or explanation failed: {e}")
