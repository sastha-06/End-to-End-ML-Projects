

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
