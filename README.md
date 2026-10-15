
# End-to-End Machine Learning Projects

A collection of end-to-end machine learning projects covering data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and application development.

## Projects

### 1. Customer Personality Segmentation

An unsupervised machine learning project that groups customers based on their demographics, income, spending patterns, and purchasing behaviour.

- **Techniques:** Data preprocessing, feature engineering, feature scaling, and clustering.
- **Models:** K-Means Clustering and DBSCAN.
- **Dataset:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Application:** Streamlit.

[View Project](./customer-personality-segmentation/)

### 2. Credit Card Fraud Detection

A supervised machine learning project to identify potentially fraudulent credit card transactions using classification algorithms.

- **Techniques:** Data cleaning, preprocessing, model training, evaluation, and model comparison.
- **Models:** Logistic Regression, Random Forest, and XGBoost.
- **Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Application:** Streamlit.

[View Project](./credit-card-fraud-detection/)

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Jupyter Notebook / Google Colab
- Git and GitHub

## Repository Structure

```text
End-to-End-ML-Projects/
│
├── README.md
│
├── customer-personality-segmentation/
│   ├── README.md
│   ├── notebook/
│   │   └── customer_personality_segmentation.ipynb
│   ├── app.py
│   ├── requirements.txt
│   └── savedmodel/
│       ├── kmeans_customer_model.pkl
│       ├── customer_scaler.pkl
│       └── preprocessing_config.pkl
│
└── credit-card-fraud-detection/
    ├── README.md
    ├── notebook/
    │   └── credit_card_fraud_detection.ipynb
    ├── app.py
    ├── requirements.txt
    └── saved_models/
        ├── logistic_regression.joblib
        ├── random_forest.joblib
        └── xgboost_model.joblib
```

*The structure and filenames above are illustrative. Keep them consistent with the actual files in your repository.*

## Objectives

- Apply machine learning concepts to practical datasets.
- Perform data cleaning, preprocessing, and exploratory data analysis.
- Train and compare suitable machine learning models.
- Evaluate models using appropriate performance metrics.
- Interpret model results and extract useful insights.
- Build interactive applications for machine learning predictions.
- Organise notebooks, trained models, and documentation for reproducibility.

## Project Status

Both projects are being developed and documented independently. Their implementation, application development, and deployment status may vary.

## Future Improvements

- Explore additional datasets and machine learning techniques.
- Improve model performance and evaluation.
- Enhance interactive visualisations and application usability.
- Deploy applications for public access.
- Add more end-to-end machine learning projects.

## Repository Purpose

This repository serves as a practical learning portfolio demonstrating the process of developing machine learning solutions, from data preparation and model experimentation to prediction and application development.
