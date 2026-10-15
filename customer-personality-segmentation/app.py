
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# 1. Load saved model files
# --------------------------------------------------

MODEL_DIR = Path(__file__).parent / "savedmodel"

try:
    kmeans = joblib.load(MODEL_DIR / "kmeans_customer_model.pkl")
    scaler = joblib.load(MODEL_DIR / "customer_scaler.pkl")
    config = joblib.load(MODEL_DIR / "preprocessing_config.pkl")
except FileNotFoundError as e:
    st.error(f"Required model file not found: {e.filename}")
    st.stop()

FEATURE_NAMES = config["feature_names"]

if len(FEATURE_NAMES) != 28 or scaler.n_features_in_ != 28:
    st.error(
        "Feature mismatch: this app expects 28 features. "
        "Check your saved model and preprocessing configuration."
    )
    st.stop()

# --------------------------------------------------
# 2. Preprocess a customer's details
# --------------------------------------------------

def preprocess_customer(data):
    customer = pd.DataFrame([data])

    # Fill missing income and apply training-time limits
    customer["Income"] = pd.to_numeric(
        customer["Income"], errors="coerce"
    ).fillna(config["income_median"])

    customer["Income"] = customer["Income"].clip(
        lower=config["income_lower"],
        upper=config["income_upper"]
    )

    # Use the same feature engineering as training
    customer["Age"] = (
        config["reference_year"] - customer["Year_Birth"]
    )

    customer["Children"] = (
        customer["Kidhome"] + customer["Teenhome"]
    )

    customer.drop(
        columns=["Year_Birth", "Kidhome", "Teenhome"],
        inplace=True
    )

    # Apply one-hot encoding
    customer = pd.get_dummies(
        customer,
        columns=["Education", "Marital_Status"],
        dtype=int
    )

    # Restore the exact training feature order
    customer = customer.reindex(
        columns=FEATURE_NAMES,
        fill_value=0
    )

    return customer


# --------------------------------------------------
# 3. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Personality Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("Customer Personality Segmentation")
st.write(
    "Discover customer segments using your trained "
    "K-Means clustering model."
)

st.info(
    "Enter a customer's personal details, spending, "
    "and purchase activity to predict their segment."
)

# --------------------------------------------------
# 4. Customer input form
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        year_birth = st.number_input(
            "Year of Birth",
            min_value=1914,
            max_value=2008,
            value=1980,
            step=1
        )

        education = st.selectbox(
            "Education",
            ["Graduation", "PhD", "Master", "Basic", "2n Cycle"]
        )

    with col2:
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married", "Together", "Single", "Divorced",
                "Widow", "Alone", "Absurd", "YOLO"
            ]
        )

        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=50000,
            step=1000
        )

    with col3:
        kidhome = st.number_input(
            "Children at Home",
            min_value=0,
            max_value=10,
            value=0
        )

        teenhome = st.number_input(
            "Teenagers at Home",
            min_value=0,
            max_value=10,
            value=0
        )

    st.subheader("Spending by Product")

    col1, col2, col3 = st.columns(3)

    with col1:
        wines = st.number_input("Wine Spending", min_value=0, value=300)
        fruits = st.number_input("Fruit Spending", min_value=0, value=20)

    with col2:
        meat = st.number_input("Meat Spending", min_value=0, value=100)
        fish = st.number_input("Fish Spending", min_value=0, value=30)

    with col3:
        sweets = st.number_input("Sweet Spending", min_value=0, value=20)
        gold = st.number_input("Gold Spending", min_value=0, value=50)

    st.subheader("Purchase Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Days Since Last Purchase", min_value=0, value=30
        )
        deals = st.number_input(
            "Deals Purchases", min_value=0, value=2
        )

    with col2:
        web_purchases = st.number_input(
            "Web Purchases", min_value=0, value=4
        )
        catalog_purchases = st.number_input(
            "Catalog Purchases", min_value=0, value=2
        )

    with col3:
        store_purchases = st.number_input(
            "Store Purchases", min_value=0, value=5
        )
        web_visits = st.number_input(
            "Website Visits per Month", min_value=0, value=5
        )

    submitted = st.form_submit_button(
        "Predict Customer Segment",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Predict the segment
# --------------------------------------------------

if submitted:

    customer_data = {
        "Year_Birth": year_birth,
        "Education": education,
        "Marital_Status": marital_status,
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Recency": recency,
        "MntWines": wines,
        "MntFruits": fruits,
        "MntMeatProducts": meat,
        "MntFishProducts": fish,
        "MntSweetProducts": sweets,
        "MntGoldProds": gold,
        "NumDealsPurchases": deals,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalog_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits
    }

    age = config["reference_year"] - year_birth

    if not 18 <= age <= 100:
        st.error("Please enter a birth year that gives an age from 18 to 100.")
    else:
        try:
            customer_features = preprocess_customer(customer_data)

            if customer_features.shape[1] != scaler.n_features_in_:
                st.error("The prepared customer features do not match the scaler.")
                st.stop()

            scaled_features = scaler.transform(customer_features)
            cluster = int(kmeans.predict(scaled_features)[0])

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Predicted Customer Cluster: {cluster}")

            if cluster == 0:
                st.write(
                    "**Cluster 0 — Lower-spending segment (in your training data).**"
                )
                st.write(
                    "Your training profile showed lower average income and "
                    "spending, more children at home, and more monthly website visits."
                )
            else:
                st.write(
                    "**Cluster 1 — Higher-spending segment (in your training data).**"
                )
                st.write(
                    "Your training profile showed higher average income and "
                    "spending, more purchases, and fewer monthly website visits."
                )

            st.caption(
                "This prediction uses your saved K-Means model. "
                "Cluster labels are model-specific, not rankings."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
