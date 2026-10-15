
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "tuned_xgboost.joblib"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

PAGES = [
    "Home",
    "Analyze Transaction",
    "Batch Analysis",
    "Model Performance",
    "About Project",
]

# Replace these approximate values with the exact training values
# from your notebook if they differ.
AMOUNT_LOWER = -102.55
AMOUNT_UPPER = 186.09

# Recorded evaluation results for the tuned XGBoost model.
MODEL_METRICS = {
    "Test Average Precision": 0.8139,
    "Precision": 0.8621,
    "Recall": 0.7895,
    "F1-score": 0.8242,
}


# ============================================================
# 2. DESIGN AND STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #f7f7fb;
        color: #211a30;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8e1f0;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #302344;
    }

    .hero {
        padding: 32px;
        border-radius: 20px;
        background: linear-gradient(
            125deg, #281044 0%, #52239a 58%, #8050d5 100%
        );
        color: #ffffff;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px rgba(70, 35, 120, 0.12);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(1.9rem, 4vw, 2.65rem);
        line-height: 1.2;
        font-weight: 800;
        margin: 8px 0 12px 0;
    }

    .hero p {
        color: #eee4ff !important;
        line-height: 1.75;
        margin: 0;
    }

    .eyebrow {
        color: #e0caff;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-intro {
        color: #5f536f;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e8e1f0;
        border-radius: 16px;
        padding: 22px;
        min-height: 150px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        color: #452078 !important;
        font-size: 1.05rem;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #5f536f !important;
        font-size: 0.91rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .notice {
        background: #f0eaff;
        border-left: 4px solid #7646bd;
        color: #382052;
        border-radius: 9px;
        padding: 15px 18px;
        line-height: 1.7;
        margin: 16px 0 22px 0;
    }

    .small-muted {
        color: #5f536f;
        font-size: 0.84rem;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e1f0;
        border-radius: 15px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #38205e !important;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e8e1f0;
        border-radius: 16px;
        padding: 22px;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #302344;
        letter-spacing: -0.4px;
    }

    .stApp p,
    .stApp li,
    .stApp label,
    .stApp [data-testid="stWidgetLabel"] {
        color: #302344;
    }

    .stApp input,
    .stApp textarea {
        color: #211a30 !important;
        background-color: #ffffff !important;
    }

    .stApp a {
        color: #54249b;
    }

    /* Consistent button appearance */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        background: #eee8f7 !important;
        color: #38205e !important;
        border: 1px solid #d8c9ed !important;
        border-radius: 10px !important;
        min-height: 44px;
        font-weight: 600 !important;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    [data-testid="stFormSubmitButton"] button * {
        color: #38205e !important;
        fill: #38205e !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #dfd1f2 !important;
        border-color: #7040bd !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: #6031a7 !important;
        color: #ffffff !important;
        border-color: #6031a7 !important;
    }

    .stButton > button[kind="primary"] *,
    .stDownloadButton > button[kind="primary"] *,
    [data-testid="stFormSubmitButton"] button[kind="primary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: #48217f !important;
        border-color: #48217f !important;
    }

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] * {
        color: #302344;
    }

    [data-testid="stExpander"] {
        border-color: #e8e1f0;
        border-radius: 10px;
    }

    hr {
        border-color: #e8e1f0;
    }

    @media (max-width: 640px) {
        .hero {
            padding: 23px;
            border-radius: 15px;
        }

        .info-card {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "The XGBoost model file was not found. Confirm that "
        "saved_models/tuned_xgboost.joblib exists in your repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Check that your requirements.txt uses library versions "
        "compatible with the versions used to save the model."
    )
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def show_hero(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_input(data):
    """Validate the input and apply the Amount cap used in training."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    prepared = data[FEATURES].copy()

    for col in FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="raise")

    values = prepared.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing, infinite, or invalid numerical values."
        )

    if (prepared["Time"] < 0).any():
        raise ValueError("Time must be zero or greater.")

    if (prepared["Amount"] < 0).any():
        raise ValueError("Transaction Amount must be zero or greater.")

    prepared["Amount"] = prepared["Amount"].clip(
        lower=AMOUNT_LOWER,
        upper=AMOUNT_UPPER,
    )

    return prepared


def predict_transactions(data):
    """Generate the model prediction and fraud probability."""

    predictions = model.predict(data).astype(int)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This saved model does not provide probability predictions."
        )

    probabilities = model.predict_proba(data)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The saved model does not contain fraud class 1.")

    fraud_index = classes.index(1)
    scores = probabilities[:, fraud_index]

    return predictions, scores


def analyze_transactions(data):
    """Validate transactions and create a prediction report."""

    prepared = prepare_input(data)
    predictions, scores = predict_transactions(prepared)

    results = prepared.copy()
    results.insert(
        0,
        "Transaction_ID",
        range(1, len(results) + 1),
    )
    results["Prediction"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Normal",
    )
    results["Estimated_Fraud_Score (%)"] = np.round(scores * 100, 2)

    return results


def show_result(prediction, score):
    st.divider()
    st.subheader("Analysis Result")

    if prediction == 1:
        st.error(
            "⚠️ Potential fraud detected. This transaction may need "
            "further investigation."
        )
    else:
        st.success("✓ The model classified this transaction as normal.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Classification",
            "Potential Fraud" if prediction == 1 else "Normal",
        )

    with col2:
        st.metric("Estimated Fraud Score", f"{score * 100:.2f}%")

    st.caption(
        "This is a machine-learning prediction, not confirmation of fraud. "
        "The probability score may not be calibrated to real-world fraud rates."
    )


def navigate_to(destination):
    """Set the actual sidebar radio value before rerunning the app."""

    st.session_state["page_navigation"] = destination
    st.rerun()


# ============================================================
# 5. SIDEBAR NAVIGATION — FIXED
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 12px 0 6px 0;">
        <div style="font-size: 2rem;">🛡️</div>
        <div style="font-size: 1.55rem; font-weight: 800; color: #54249b;">
            FraudGuard
        </div>
        <div style="font-size: 0.85rem; color: #5f536f;">
            Intelligent Transaction Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# This key is shared by the sidebar and the Home page navigation buttons.
# Changing it and rerunning selects the requested page.
if st.session_state.get("page_navigation") not in PAGES:
    st.session_state["page_navigation"] = "Home"

page = st.sidebar.radio(
    "NAVIGATION",
    options=PAGES,
    key="page_navigation",
)

st.sidebar.divider()
st.sidebar.markdown("**Active Model**")
st.sidebar.success("Tuned XGBoost")

st.sidebar.markdown(
    """
    <div class="small-muted">
        Built with Python, XGBoost and Streamlit.<br><br>
        Educational machine-learning project.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. HOME
# ============================================================

if page == "Home":

    show_hero(
        "MACHINE LEARNING • TRANSACTION SECURITY",
        "Smarter Fraud Detection.",
        "Explore transaction risk using a trained XGBoost classifier. "
        "Analyze a single transaction or process a CSV file.",
    )

    st.subheader("Your Fraud Analysis Workspace")

    st.markdown(
        '<div class="section-intro">'
        "A simple interface for exploring potential credit card fraud "
        "with supervised machine learning."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Active Model", "XGBoost")

    with c2:
        st.metric("Input Features", "30")

    with c3:
        st.metric("Test Average Precision", "0.8139")

    st.markdown("### What would you like to do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🔎 Single Transaction Analysis</h3>
                <p>Enter transaction details and the 28 anonymized
                numerical features to receive a model prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Analyze a Transaction →",
            key="home_single",
            type="primary",
            use_container_width=True,
        ):
            navigate_to("Analyze Transaction")

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Batch CSV Analysis</h3>
                <p>Upload a CSV, review the predicted classifications,
                and download a report of the results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Analyze a CSV →",
            key="home_batch",
            type="primary",
            use_container_width=True,
        ):
            navigate_to("Batch Analysis")

    st.markdown("### Model at a glance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🌲 Gradient Boosting</h3>
                <p>XGBoost combines decision trees to learn patterns
                associated with potentially fraudulent transactions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>⚖️ Imbalanced Dataset</h3>
                <p>The cleaned dataset contains 473 fraud cases among
                283,726 transactions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
                <h3>📈 Model Evaluation</h3>
                <p>Average Precision, precision, recall and F1-score
                help evaluate performance beyond accuracy alone.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            <strong>Important:</strong> This is an educational demonstration.
            It is not connected to a bank and should not be used to make
            real financial decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. SINGLE TRANSACTION ANALYSIS
# ============================================================

elif page == "Analyze Transaction":

    show_hero(
        "TRANSACTION ANALYSIS",
        "Analyze a Transaction.",
        "Enter the numerical features to see how the trained "
        "XGBoost model classifies a transaction.",
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Before you begin:</strong> V1–V28 are anonymized
            numerical features from the original dataset. They are not
            ordinary customer details. For meaningful predictions,
            use actual feature values from a compatible dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Transaction Details")

    with st.form("single_transaction_form"):

        c1, c2 = st.columns(2)

        with c1:
            time_value = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Elapsed seconds in the original dataset.",
            )

        with c2:
            amount_value = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Use the same amount units as the training dataset.",
            )

        st.markdown("#### Anonymized Features")

        use_zero_features = st.checkbox(
            "Use zero for all V1–V28 features",
            value=True,
            help=(
                "Zero placeholders are for demonstration only. "
                "They do not represent a realistic transaction."
            ),
        )

        feature_values = {}

        if use_zero_features:
            feature_values = {
                f"V{i}": 0.0 for i in range(1, 29)
            }
        else:
            for start in range(1, 29, 4):
                cols = st.columns(4)

                for offset, col in enumerate(cols):
                    index = start + offset

                    if index <= 28:
                        with col:
                            feature_values[f"V{index}"] = st.number_input(
                                f"V{index}",
                                value=0.0,
                                format="%.6f",
                                key=f"single_v{index}",
                            )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        transaction = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value,
        }

        try:
            input_df = pd.DataFrame([transaction])
            results = analyze_transactions(input_df)

            prediction = (
                1
                if results["Prediction"].iloc[0] == "Potential Fraud"
                else 0
            )
            score = (
                results["Estimated_Fraud_Score (%)"].iloc[0] / 100
            )

            show_result(prediction, score)

            with st.expander("View processed model input"):
                st.dataframe(
                    results[FEATURES],
                    use_container_width=True,
                    hide_index=True,
                )

            if use_zero_features:
                st.warning(
                    "All 28 anonymized features were set to zero. "
                    "This prediction is for artificial input and should "
                    "not be interpreted as a realistic fraud assessment."
                )

        except Exception as exc:
            st.error(f"Unable to analyze this transaction: {exc}")


# ============================================================
# 8. BATCH CSV ANALYSIS
# ============================================================

elif page == "Batch Analysis":

    show_hero(
        "BATCH PROCESSING",
        "Analyze Multiple Transactions.",
        "Upload a compatible CSV, review the predicted classifications, "
        "and export the results.",
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Required columns:</strong> Time, V1, V2, ...,
            V28 and Amount. Column names must match the training features.
            An optional Class column is not used for prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"],
        help="Upload a CSV with the 30 required numerical feature columns.",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.warning("The uploaded file contains no transactions.")

            else:
                missing = [
                    col for col in FEATURES
                    if col not in uploaded_df.columns
                ]

                if missing:
                    st.error(
                        "The CSV is missing required columns: "
                        + ", ".join(missing)
                    )

                elif len(uploaded_df) > 10000:
                    st.warning(
                        "This file contains more than 10,000 rows. "
                        "Please split it into smaller files."
                    )

                else:
                    st.caption(
                        f"{len(uploaded_df):,} transactions loaded."
                    )

                    st.dataframe(
                        uploaded_df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "Run Fraud Analysis",
                        type="primary",
                        use_container_width=True,
                        key="run_batch_analysis",
                    ):
                        try:
                            model_input = uploaded_df[FEATURES].copy()
                            st.session_state["batch_results"] = (
                                analyze_transactions(model_input)
                            )
                        except Exception as exc:
                            st.session_state.pop("batch_results", None)
                            st.error(
                                f"Unable to analyze the uploaded data: {exc}"
                            )

                    # Keep results available when the filter or another
                    # widget changes and causes Streamlit to rerun.
                    results = st.session_state.get("batch_results")

                    if results is not None:
                        fraud_count = int(
                            (
                                results["Prediction"] == "Potential Fraud"
                            ).sum()
                        )
                        normal_count = len(results) - fraud_count

                        st.markdown("### Analysis Summary")

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.metric(
                                "Transactions Analyzed",
                                f"{len(results):,}",
                            )

                        with c2:
                            st.metric(
                                "Potential Fraud Flags",
                                f"{fraud_count:,}",
                            )

                        with c3:
                            st.metric(
                                "Classified as Normal",
                                f"{normal_count:,}",
                            )

                        if fraud_count:
                            st.warning(
                                f"The model flagged {fraud_count:,} "
                                "transaction(s) for further review."
                            )
                        else:
                            st.success(
                                "The model did not flag any transactions "
                                "in this batch."
                            )

                        st.markdown("### Prediction Results")

                        filter_choice = st.selectbox(
                            "Filter results",
                            [
                                "All Transactions",
                                "Potential Fraud Only",
                                "Normal Only",
                            ],
                        )

                        if filter_choice == "Potential Fraud Only":
                            display_df = results[
                                results["Prediction"] == "Potential Fraud"
                            ]
                        elif filter_choice == "Normal Only":
                            display_df = results[
                                results["Prediction"] == "Normal"
                            ]
                        else:
                            display_df = results

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "Download Complete Results (CSV)",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="fraud_detection_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        st.caption(
                            "A fraud prediction is not confirmation of fraud. "
                            "Review flagged transactions independently."
                        )

        except Exception as exc:
            st.error(f"Unable to read the CSV file: {exc}")

    else:
        st.markdown("### Need a sample template?")

        template = pd.DataFrame(
            [{feature: 0.0 for feature in FEATURES}]
        )

        st.download_button(
            "Download CSV Template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="transaction_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "The template contains placeholder zeros. Replace them with "
            "valid feature values before interpreting predictions."
        )


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    show_hero(
        "MODEL EVALUATION",
        "Performance Overview.",
        "Review the recorded test results of the tuned XGBoost model "
        "selected for this application.",
    )

    st.markdown(
        """
        <div class="notice">
            These results come from your earlier notebook evaluation.
            The app displays the recorded metrics; it does not retrain
            or reevaluate the model on this page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Test Average Precision",
            f"{MODEL_METRICS['Test Average Precision']:.4f}",
        )

    with c2:
        st.metric(
            "Precision",
            f"{MODEL_METRICS['Precision']:.4f}",
        )

    with c3:
        st.metric(
            "Recall",
            f"{MODEL_METRICS['Recall']:.4f}",
        )

    with c4:
        st.metric(
            "F1-score",
            f"{MODEL_METRICS['F1-score']:.4f}",
        )

    st.markdown("### Evaluation Metrics")

    metric_data = pd.DataFrame(
        {
            "Metric": list(MODEL_METRICS.keys()),
            "Score": list(MODEL_METRICS.values()),
        }
    )

    st.bar_chart(
        metric_data.set_index("Metric"),
        y="Score",
        color="#7040bd",
    )

    st.markdown("### What do these results mean?")

    st.markdown(
        """
        - **Average Precision (0.8139):** Summarizes precision-recall
          performance across different score thresholds.
        - **Precision (0.8621):** Approximately 86.21% of the transactions
          flagged as fraud in the recorded evaluation were fraud cases.
        - **Recall (0.7895):** The model detected approximately 78.95% of
          actual fraud cases in the recorded evaluation.
        - **F1-score (0.8242):** Combines precision and recall into one score.
        """
    )

    st.markdown("### Training Configuration")

    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": [
                    "Model",
                    "Number of estimators",
                    "Maximum tree depth",
                    "Learning rate",
                    "Subsample",
                    "Column sample by tree",
                ],
                "Value": [
                    "Tuned XGBoost",
                    "150",
                    "5",
                    "0.1",
                    "1.0",
                    "1.0",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Repeated experimentation on the same test set can make reported "
        "performance optimistic. Independent validation is recommended."
    )


# ============================================================
# 10. ABOUT THE PROJECT
# ============================================================

elif page == "About Project":

    show_hero(
        "PROJECT INFORMATION",
        "About FraudGuard.",
        "A supervised machine-learning project exploring how transaction "
        "features can help identify potentially fraudulent activity.",
    )

    st.subheader("Dataset")

    st.markdown(
        "[Credit Card Fraud Detection dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )

    st.write(
        "After removing 1,081 exact duplicate rows, the cleaned dataset "
        "contains 283,726 transactions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Transactions", "283,726")

    with c2:
        st.metric("Normal Transactions", "283,253")

    with c3:
        st.metric("Fraud Transactions", "473")

    st.markdown("### Features Used")

    st.markdown(
        """
        - **Time:** Seconds elapsed since the first transaction.
        - **V1–V28:** Anonymized numerical features derived using PCA.
        - **Amount:** The transaction amount.
        - **Class:** The original target label (0 = normal, 1 = fraud).
          Class is not an input to the deployed model.
        """
    )

    st.markdown("### Methodology")

    st.markdown(
        """
        1. Remove exact duplicate records.
        2. Split data into stratified training and testing sets.
        3. Cap Amount outliers using limits computed on training data.
        4. Train and compare Logistic Regression, Random Forest and XGBoost.
        5. Select tuned XGBoost for deployment based on its recorded
           test Average Precision.
        """
    )

    st.markdown("### Limitations")

    st.warning(
        "This is a student project, not a production fraud prevention system. "
        "The features are anonymized, the model uses historical data, and "
        "its probability scores may not be calibrated. Real-world deployment "
        "requires independent validation, reliable feature generation, "
        "monitoring, security and privacy controls."
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:10px 0 22px 0;">
        <div style="font-weight:700; color:#54249b;">
            🛡️ FraudGuard
        </div>
        <div class="small-muted">
            Credit Card Fraud Detection • Built with Streamlit and XGBoost
        </div>
        <div class="small-muted">
            For educational and demonstration purposes only.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
