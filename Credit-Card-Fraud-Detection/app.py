
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "tuned_xgboost.joblib"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Approximate training-set Amount capping limits.
# Replace these with the exact values from your training notebook if needed.
AMOUNT_LOWER = -102.55
AMOUNT_UPPER = 186.09


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f7f7fb;
        color: #211a30;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e9e5f2;
    }

    [data-testid="stSidebar"] h1 {
        color: #54249b;
        font-weight: 800;
    }

    .hero {
        padding: 34px;
        border-radius: 22px;
        background: linear-gradient(125deg, #281044 0%, #52239a 55%, #8050d5 100%);
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(70, 35, 120, 0.13);
    }

    .hero h1 {
        font-size: 2.35rem;
        line-height: 1.2;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .hero p {
        color: #eee4ff;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #e0caff;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .section-intro {
        color: #706780;
        line-height: 1.65;
        margin-bottom: 20px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #ebe6f3;
        border-radius: 16px;
        padding: 22px;
        min-height: 135px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        color: #452078;
        font-size: 1rem;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #716982;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .notice {
        background: #f0eaff;
        border-left: 4px solid #7646bd;
        color: #382052;
        border-radius: 8px;
        padding: 15px 18px;
        line-height: 1.6;
        margin: 14px 0 20px 0;
    }

    .small-muted {
        color: #746b83;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #ebe6f3;
        border-radius: 15px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #716780;
    }

    div[data-testid="stMetricValue"] {
        color: #38205e;
        font-weight: 750;
    }

    /* All buttons: readable text and consistent styling */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        background-color: #eee8f7 !important;
        color: #38205e !important;
        border: 1px solid #d8c9ed !important;
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }

    /* Ensure nested button text and icons inherit the visible color */
    .stButton > button *,
    .stDownloadButton > button *,
    [data-testid="stFormSubmitButton"] button * {
        color: inherit !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #dfd1f2 !important;
        color: #281044 !important;
        border-color: #7040bd !important;
    }

    /* Primary action buttons use purple with white text */
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background-color: #6031a7 !important;
        color: #ffffff !important;
        border: 1px solid #6031a7 !important;
    }

    .stButton > button[kind="primary"] *,
    .stDownloadButton > button[kind="primary"] *,
    [data-testid="stFormSubmitButton"] button[kind="primary"] * {
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background-color: #48217f !important;
        color: #ffffff !important;
        border-color: #48217f !important;
    }

    /* Disabled buttons remain readable */
    .stButton > button:disabled,
    .stDownloadButton > button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled {
        background-color: #e8e5ed !important;
        color: #655b72 !important;
        border-color: #d7d1df !important;
        opacity: 1 !important;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #ebe6f3;
        border-radius: 16px;
        padding: 20px;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    hr {
        border-color: #e9e4f0;
    }

    /* --------------------------------------------------------
       ACCESSIBILITY: keep text readable across the whole app
       -------------------------------------------------------- */

    :root {
        --text-color: #211a30;
        --secondary-text-color: #554b65;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        color: #211a30;
    }

    /* Main page headings and normal Streamlit text */
    .stApp h1:not(.hero h1),
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp legend,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stText"],
    .stApp [data-testid="stWidgetLabel"] {
        color: #302344;
    }

    .stApp [data-testid="stCaptionContainer"],
    .stApp .stCaption,
    .stApp small {
        color: #5f536f;
    }

    /* Keep the hero banner's light text readable on its dark background */
    .stApp .hero,
    .stApp .hero h1,
    .stApp .hero p,
    .stApp .hero .eyebrow {
        color: #ffffff;
    }

    .stApp .hero p {
        color: #eee4ff;
    }

    .stApp .hero .eyebrow {
        color: #e0caff;
    }

    /* Sidebar: navigation, headings, descriptions and labels */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #302344;
    }

    [data-testid="stSidebar"] .small-muted {
        color: #5f536f;
    }

    /* Inputs, dropdowns, radio buttons and checkboxes */
    .stApp input,
    .stApp textarea,
    .stApp select,
    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="input"] *,
    .stApp [data-baseweb="textarea"] *,
    .stApp [data-testid="stNumberInput"] input {
        color: #211a30;
    }

    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: #6b6078;
        opacity: 1;
    }

    /* Metric labels and values */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"] {
        color: #38205e;
    }

    /* Tables, dataframes, tabs, expanders and help text */
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"],
    .stApp [data-testid="stExpander"],
    .stApp [data-testid="stTab"],
    .stApp [data-testid="stTooltipContent"] {
        color: #302344;
    }

    .stApp [role="radiogroup"] label,
    .stApp [role="checkbox"] label {
        color: #302344;
    }

    /* Make links clearly visible against light backgrounds */
    .stApp a {
        color: #54249b;
    }

    .stApp a:hover {
        color: #351365;
    }

    /* Preserve readable button text */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        color: #302344;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        color: #ffffff;
    }

    /* Keep alert text readable on its colored background */
    [data-testid="stAlert"] {
        color: #211a30;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: inherit;
    }

    /* Streamlit sidebar success panel has its own green styling */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        color: #166534;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stAlert"] span {
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD THE SAVED XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "The trained XGBoost model was not found. Check that "
        "`saved_models/tuned_xgboost.joblib` exists in your repository."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(
        "The saved model could not be loaded. This may be caused by "
        "incompatible library versions or an invalid model file."
    )
    st.code(str(exc))
    st.info(
        "Check the scikit-learn and XGBoost versions used when the model "
        "was saved, and use compatible versions in requirements.txt."
    )
    st.stop()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def prepare_input(data):
    """Validate model features and apply the training Amount cap."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    missing = [column for column in FEATURES if column not in data.columns]

    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )

    prepared = data[FEATURES].copy()

    for column in FEATURES:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="raise",
        )

    values = prepared.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing, infinite, or invalid numerical values."
        )

    if (prepared["Time"] < 0).any():
        raise ValueError("Time must be zero or greater.")

    if (prepared["Amount"] < 0).any():
        raise ValueError("Transaction Amount must be zero or greater.")

    # Match the approximate Amount capping used during training.
    prepared["Amount"] = prepared["Amount"].clip(
        lower=AMOUNT_LOWER,
        upper=AMOUNT_UPPER,
    )

    return prepared


def predict_transactions(data):
    """Return binary predictions and estimated fraud scores."""

    predictions = model.predict(data).astype(int)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("The saved model does not contain fraud class 1.")

        fraud_index = classes.index(1)
        scores = probabilities[:, fraud_index]
    else:
        scores = np.full(len(data), np.nan)

    return predictions, scores


def analyze_transactions(data):
    """Create a result table for valid model input."""

    prepared = prepare_input(data)
    predictions, scores = predict_transactions(prepared)

    results = prepared.copy()
    results.insert(0, "Transaction_ID", range(1, len(results) + 1))
    results["Prediction"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Normal",
    )
    results["Estimated_Fraud_Score (%)"] = np.round(scores * 100, 2)

    return results


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


def show_result(prediction, score):
    st.divider()
    st.subheader("Analysis Result")

    if prediction == 1:
        st.error(
            "⚠️ Potential fraud detected by the model. "
            "This transaction may require further review."
        )
    else:
        st.success(
            "✓ The model classified this transaction as normal."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Classification",
            "Potential Fraud" if prediction == 1 else "Normal",
        )

    with col2:
        if np.isfinite(score):
            st.metric("Estimated Fraud Score", f"{score * 100:.2f}%")
        else:
            st.metric("Estimated Fraud Score", "Unavailable")

    st.caption(
        "This is a machine-learning prediction, not confirmation of fraud. "
        "The score is the model's estimated probability and has not "
        "necessarily been calibrated to real-world fraud rates."
    )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 12px 0 6px 0;">
        <div style="font-size: 2rem;">🛡️</div>
        <div style="font-size: 1.55rem; font-weight: 800; color: #54249b;">
            FraudGuard
        </div>
        <div style="font-size: 0.85rem; color: #776c87;">
            Intelligent Transaction Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Analyze Transaction",
        "Batch Analysis",
        "Model Performance",
        "About Project",
    ],
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
# 6. HOME DASHBOARD
# ============================================================

if page == "Home":

    show_hero(
        "MACHINE LEARNING • TRANSACTION SECURITY",
        "Smarter Fraud Detection.",
        "Explore transaction risk using a trained XGBoost classifier. "
        "Analyze individual transactions or process a CSV file in seconds.",
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
        st.metric("Model Input Features", "30")

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
            use_container_width=True,
        ):
            st.session_state["navigation_target"] = "Analyze Transaction"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Batch CSV Analysis</h3>
                <p>Upload a CSV of transactions, generate predictions,
                review the results, and download a report.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Analyze a CSV →",
            key="home_batch",
            use_container_width=True,
        ):
            st.session_state["navigation_target"] = "Batch Analysis"
            st.rerun()

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
                <h3>⚖️ Imbalanced Data</h3>
                <p>The cleaned dataset contains 473 fraud cases among
                283,726 transactions, making recall and precision important.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
                <h3>📈 Evaluation</h3>
                <p>Recorded results include Average Precision, precision,
                recall and F1-score rather than accuracy alone.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            <strong>Important:</strong> This application is an educational
            demonstration. It is not connected to a bank and should not be
            used to make real financial decisions.
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
        "Provide the transaction's numerical features to see how "
        "the trained XGBoost model classifies it.",
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Before you begin:</strong> V1–V28 are anonymized
            numerical features from the original dataset. They are not
            card numbers, names, or ordinary customer details.
            For meaningful predictions, use real feature values from
            a compatible dataset.
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

        st.markdown("#### Anonymized features")

        use_zero_features = st.checkbox(
            "Use zero for all V1–V28 features",
            value=True,
            help=(
                "This is a demonstration input, not a realistic transaction. "
                "For useful predictions, enter the actual feature values."
            ),
        )

        feature_values = {}

        if not use_zero_features:
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
        else:
            feature_values = {f"V{i}": 0.0 for i in range(1, 29)}

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
                1 if results["Prediction"].iloc[0] == "Potential Fraud" else 0
            )
            score = results["Estimated_Fraud_Score (%)"].iloc[0] / 100

            show_result(prediction, score)

            with st.expander("View processed model input"):
                st.dataframe(
                    results[FEATURES],
                    use_container_width=True,
                    hide_index=True,
                )

            if use_zero_features:
                st.warning(
                    "You used zero placeholders for all anonymized features. "
                    "The prediction is for this artificial input and should "
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
        "Upload a compatible CSV, review predicted classifications, "
        "and export the results for further analysis.",
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Required CSV columns:</strong> Time, V1, V2, ...,
            V28, Amount. Column names must match the original training
            features. An optional Class column is ignored during prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"],
        help="Use a CSV with the 30 required numerical feature columns.",
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if uploaded_df.empty:
                st.warning("The uploaded file contains no transactions.")

            else:
                missing = [
                    column for column in FEATURES
                    if column not in uploaded_df.columns
                ]

                if missing:
                    st.error(
                        "The CSV is missing these required columns: "
                        + ", ".join(missing)
                    )

                elif len(uploaded_df) > 10000:
                    st.warning(
                        "This file contains more than 10,000 rows. "
                        "Please split it into smaller files."
                    )

                else:
                    st.markdown("### Uploaded data preview")
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
                    ):

                        try:
                            # Class, if present, is never passed to the model.
                            model_input = uploaded_df[FEATURES].copy()
                            results = analyze_transactions(model_input)

                            fraud_count = int(
                                (results["Prediction"] == "Potential Fraud").sum()
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

                            csv_bytes = results.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "Download Complete Results (CSV)",
                                data=csv_bytes,
                                file_name="fraud_detection_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                            st.caption(
                                "A predicted fraud flag is not a confirmed "
                                "fraud case. Review flagged transactions "
                                "independently."
                            )

                        except Exception as exc:
                            st.error(
                                f"Unable to analyze the uploaded data: {exc}"
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
            "The template contains placeholder zeros only. Replace them "
            "with valid feature values before interpreting predictions."
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
            These metrics were recorded during your earlier notebook
            evaluation. They are not recalculated by this web application.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Test Average Precision", "0.8139")

    with c2:
        st.metric("Precision", "0.8621")

    with c3:
        st.metric("Recall", "0.7895")

    with c4:
        st.metric("F1-score", "0.8242")

    st.markdown("### Evaluation Metrics")

    metric_data = pd.DataFrame(
        {
            "Metric": [
                "Test Average Precision",
                "Precision",
                "Recall",
                "F1-score",
            ],
            "Score": [0.8139, 0.8621, 0.7895, 0.8242],
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
        - **Precision (0.8621):** Among transactions flagged by the model
          in the recorded test evaluation, approximately 86.21% were fraud.
        - **Recall (0.7895):** The model detected approximately 78.95% of
          the actual fraud cases in that evaluation.
        - **F1-score (0.8242):** Combines precision and recall into a
          single measure.
        """
    )

    st.markdown("### Training configuration")

    params = {
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

    st.dataframe(
        pd.DataFrame(params),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Results depend on the dataset and evaluation split. Repeated "
        "experimentation on the same test set can make reported performance "
        "optimistic. Independent validation is recommended."
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

    st.markdown("### Features used")

    st.markdown(
        """
        - **Time:** Seconds elapsed since the first transaction in the dataset.
        - **V1–V28:** Anonymized numerical features derived using PCA.
        - **Amount:** The transaction amount.
        - **Class:** The original target label (0 = normal, 1 = fraud).
          It is not an input to the deployed prediction model.
        """
    )

    st.markdown("### Methodology")

    st.markdown(
        """
        1. Remove exact duplicate records.
        2. Split data into stratified training and testing sets.
        3. Cap Amount outliers using limits computed on training data.
        4. Train and compare Logistic Regression, Random Forest and XGBoost.
        5. Select tuned XGBoost for this deployment based on its recorded
           test Average Precision.
        """
    )

    st.markdown("### Limitations")

    st.warning(
        "This is a student project and not a production fraud prevention "
        "system. The input features are anonymized, the model was trained "
        "on historical data, and probability scores may not be calibrated. "
        "Real-world deployment would require reliable feature generation, "
        "monitoring, independent validation, security and privacy controls."
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
