
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SegmentIQ | Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# ==================================================
# 2. LOAD SAVED MODEL FILES
# ==================================================

MODEL_DIR = Path(__file__).parent / "savedmodel"

try:
    kmeans = joblib.load(
        MODEL_DIR / "kmeans_customer_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "customer_scaler.pkl"
    )

    config = joblib.load(
        MODEL_DIR / "preprocessing_config.pkl"
    )

except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Check the savedmodel folder and file names."
    )
    st.stop()

FEATURE_NAMES = config["feature_names"]

if (
    len(FEATURE_NAMES) != 28
    or scaler.n_features_in_ != 28
    or kmeans.n_features_in_ != 28
):
    st.error(
        "Feature mismatch. The model, scaler, and configuration "
        "must all use the same 28 features."
    )
    st.stop()


# ==================================================
# 3. CLUSTER PROFILES FROM TRAINING DATA
# ==================================================

# These are the previously calculated training-cluster
# averages. Update them if you retrain your model.

CLUSTER_PROFILES = {
    0: {
        "Income": 39327.55,
        "Age": 44.09,
        "Children": 1.25,
        "MntWines": 114.20,
        "MntMeatProducts": 38.40,
        "NumWebPurchases": 3.01,
        "NumStorePurchases": 4.01,
        "NumWebVisitsMonth": 6.42,
    },
    1: {
        "Income": 71501.48,
        "Age": 46.68,
        "Children": 0.48,
        "MntWines": 601.66,
        "MntMeatProducts": 368.47,
        "NumWebPurchases": 5.77,
        "NumStorePurchases": 8.60,
        "NumWebVisitsMonth": 3.60,
    },
}


# ==================================================
# 4. PREPROCESS CUSTOMER INPUT
# ==================================================

def preprocess_customer(data):
    customer = pd.DataFrame([data]).copy()

    # Fill missing income using the training median
    customer["Income"] = pd.to_numeric(
        customer["Income"],
        errors="coerce"
    ).fillna(config["income_median"])

    # Apply the training-time income limits
    customer["Income"] = customer["Income"].clip(
        lower=config["income_lower"],
        upper=config["income_upper"]
    )

    # Create age using the same reference year
    customer["Age"] = (
        config["reference_year"] - customer["Year_Birth"]
    )

    # Combine children and teenagers
    customer["Children"] = (
        customer["Kidhome"] + customer["Teenhome"]
    )

    customer.drop(
        columns=[
            "Year_Birth",
            "Kidhome",
            "Teenhome"
        ],
        inplace=True
    )

    # One-hot encode categorical variables
    customer = pd.get_dummies(
        customer,
        columns=["Education", "Marital_Status"],
        dtype=int
    )

    # Ensure exact training columns and their order
    customer = customer.reindex(
        columns=FEATURE_NAMES,
        fill_value=0
    )

    return customer


# ==================================================
# 5. EXPLAIN THE PREDICTION
# ==================================================

def explain_prediction(data, cluster):
    profile = CLUSTER_PROFILES[cluster]

    comparisons = [
        (
            "Annual income",
            data["Income"],
            profile["Income"],
            "₹{:,.0f}"
        ),
        (
            "Age",
            config["reference_year"] - data["Year_Birth"],
            profile["Age"],
            "{:.1f} years"
        ),
        (
            "Children at home",
            data["Kidhome"] + data["Teenhome"],
            profile["Children"],
            "{:.2f}"
        ),
        (
            "Wine spending",
            data["MntWines"],
            profile["MntWines"],
            "{:.2f}"
        ),
        (
            "Meat spending",
            data["MntMeatProducts"],
            profile["MntMeatProducts"],
            "{:.2f}"
        ),
        (
            "Web purchases",
            data["NumWebPurchases"],
            profile["NumWebPurchases"],
            "{:.2f}"
        ),
        (
            "Store purchases",
            data["NumStorePurchases"],
            profile["NumStorePurchases"],
            "{:.2f}"
        ),
        (
            "Website visits per month",
            data["NumWebVisitsMonth"],
            profile["NumWebVisitsMonth"],
            "{:.2f}"
        ),
    ]

    rows = []
    explanations = []

    for feature, entered, average, fmt in comparisons:
        difference = entered - average

        if difference > 0:
            direction = "Above cluster average"
        elif difference < 0:
            direction = "Below cluster average"
        else:
            direction = "Equal to cluster average"

        rows.append({
            "Feature": feature,
            "Your input": fmt.format(entered),
            "Cluster average": fmt.format(average),
            "Comparison": direction
        })

        # Explain the largest meaningful differences.
        # This is descriptive, not a causal explanation.
        if average != 0:
            relative_difference = abs(difference) / abs(average)
        else:
            relative_difference = 0

        if relative_difference >= 0.20:
            if difference > 0:
                explanations.append(
                    f"**{feature}:** your value is above "
                    f"this cluster's average "
                    f"({fmt.format(entered)} versus "
                    f"{fmt.format(average)})."
                )
            elif difference < 0:
                explanations.append(
                    f"**{feature}:** your value is below "
                    f"this cluster's average "
                    f"({fmt.format(entered)} versus "
                    f"{fmt.format(average)})."
                )

    return pd.DataFrame(rows), explanations


# ==================================================
# 6. APP HEADER
# ==================================================

st.title("👥 SegmentIQ")
st.subheader("Customer Personality Segmentation")
st.write(
    "Discover customer segments using your trained "
    "K-Means clustering model."
)

st.info(
    "Enter customer details to predict a segment. "
    "The explanation compares selected customer values "
    "with the corresponding training-cluster averages."
)


# ==================================================
# 7. CUSTOMER INPUT FORM
# ==================================================

with st.form("customer_form"):

    st.header("Customer Details")

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
            [
                "Graduation",
                "PhD",
                "Master",
                "Basic",
                "2n Cycle"
            ]
        )

    with col2:
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Together",
                "Single",
                "Divorced",
                "Widow",
                "Alone",
                "Absurd",
                "YOLO"
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
        wines = st.number_input(
            "Wine Spending",
            min_value=0,
            value=300
        )
        fruits = st.number_input(
            "Fruit Spending",
            min_value=0,
            value=20
        )

    with col2:
        meat = st.number_input(
            "Meat Spending",
            min_value=0,
            value=100
        )
        fish = st.number_input(
            "Fish Spending",
            min_value=0,
            value=30
        )

    with col3:
        sweets = st.number_input(
            "Sweet Spending",
            min_value=0,
            value=20
        )
        gold = st.number_input(
            "Gold Spending",
            min_value=0,
            value=50
        )

    st.subheader("Purchase Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            value=30
        )
        deals = st.number_input(
            "Deals Purchases",
            min_value=0,
            value=2
        )

    with col2:
        web_purchases = st.number_input(
            "Web Purchases",
            min_value=0,
            value=4
        )
        catalog_purchases = st.number_input(
            "Catalog Purchases",
            min_value=0,
            value=2
        )

    with col3:
        store_purchases = st.number_input(
            "Store Purchases",
            min_value=0,
            value=5
        )
        web_visits = st.number_input(
            "Website Visits per Month",
            min_value=0,
            value=5
        )

    submitted = st.form_submit_button(
        "Predict Customer Segment",
        type="primary",
        use_container_width=True
    )


# ==================================================
# 8. PREDICTION AND INPUT-BASED EXPLANATION
# ==================================================

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
        st.error(
            "Please enter a birth year that gives an age "
            "between 18 and 100."
        )

    else:
        try:
            # Prepare the 28 input features
            customer_features = preprocess_customer(
                customer_data
            )

            if customer_features.shape[1] != 28:
                st.error(
                    "The prepared input does not contain "
                    "exactly 28 features."
                )
                st.stop()

            # Apply the already-fitted training scaler
            scaled_features = scaler.transform(
                customer_features
            )

            # Predict using the saved K-Means model
            cluster = int(
                kmeans.predict(scaled_features)[0]
            )

            st.divider()
            st.header("Prediction Result")

            st.success(
                f"Predicted Customer Cluster: {cluster}"
            )

            if cluster == 0:
                st.markdown(
                    "### Cluster 0 — Lower-spending profile"
                )
                st.write(
                    "In your training data, this cluster had "
                    "lower average income and product spending, "
                    "more children at home, fewer purchases, "
                    "and more monthly website visits."
                )
            else:
                st.markdown(
                    "### Cluster 1 — Higher-spending profile"
                )
                st.write(
                    "In your training data, this cluster had "
                    "higher average income and product spending, "
                    "more purchases, fewer children at home, "
                    "and fewer monthly website visits."
                )

            # Compare the actual inputs against cluster averages
            st.subheader(
                "Why was this customer assigned to this cluster?"
            )

            st.write(
                "The table compares your entered values with "
                "the average profile of the predicted cluster."
            )

            comparison_df, explanations = explain_prediction(
                customer_data,
                cluster
            )

            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True
            )

            st.subheader("Input-based explanation")

            if explanations:
                for explanation in explanations:
                    st.markdown(f"- {explanation}")
            else:
                st.write(
                    "The selected values are relatively close "
                    "to this cluster's averages. The assignment "
                    "may reflect the combined pattern across "
                    "all 28 features."
                )

            st.caption(
                "K-Means uses all 28 scaled features to assign "
                "the cluster. These comparisons describe selected "
                "differences from cluster averages; they do not "
                "prove that any single input caused the assignment."
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )
