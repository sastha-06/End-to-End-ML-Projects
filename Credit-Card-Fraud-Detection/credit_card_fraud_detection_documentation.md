# Credit Card Fraud Detection Using Machine Learning

## 1. Project Overview

Credit card fraud is a major challenge in digital financial transactions. Fraudulent transactions can cause financial losses to customers and financial institutions. Machine learning can help identify potentially fraudulent transactions by learning patterns from historical transaction data.

This project develops and compares supervised machine learning classification models to distinguish fraudulent credit card transactions from legitimate transactions. Since fraudulent transactions represent only a small proportion of the dataset, the project focuses on evaluation metrics suitable for highly imbalanced classification problems.

The project also includes saving trained models and developing a Streamlit application to demonstrate predictions on user-provided transaction features.

### 1.1 Objectives

- Understand and preprocess credit card transaction data.
- Analyze the imbalance between normal and fraudulent transactions.
- Train multiple supervised machine learning models.
- Evaluate model performance using appropriate metrics.
- Compare model results and select a preferred model.
- Examine feature importance to understand model behavior.
- Save trained models for future use.
- Develop a Streamlit interface for transaction prediction.
- Display prediction results and individual feature contributions.
- Prepare the project for GitHub sharing and deployment.

## 2. Dataset Description

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains European cardholder transactions made over two days in September 2013.

- Original records: 284,807
- Input features: 30
- Target column: `Class`
- Total columns: 31

### 2.1 Important Columns

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1`–`V28` | Anonymized numerical features obtained through PCA transformation |
| `Class` | Target label: 0 = normal, 1 = fraud |

The `Class` column is the target variable. The remaining 30 columns are used as input features.

The anonymized `V1`–`V28` features do not directly reveal the original transaction attributes.

## 3. Data Cleaning and Preprocessing

### 3.1 Duplicate Removal

The original dataset contained 284,807 records. A total of 1,081 exact duplicate rows were removed.

| Description | Records |
|---|---:|
| Original dataset | 284,807 |
| Duplicate rows removed | 1,081 |
| Cleaned dataset | 283,726 |

Removing duplicate rows reduced repeated records before model training.

### 3.2 Class Imbalance Analysis

After duplicate removal, the class distribution was:

| Transaction type | Count | Approximate percentage |
|---|---:|---:|
| Normal (0) | 283,253 | 99.833% |
| Fraud (1) | 473 | 0.167% |
| Total | 283,726 | 100% |

The dataset is highly imbalanced because fraudulent transactions account for only a small proportion of all records.

A model that predicts every transaction as normal could achieve high accuracy while failing to identify fraud. Therefore, accuracy alone is not sufficient to evaluate the models.

Average Precision, precision, recall, and F1 score were prioritized.

### 3.3 Train-Test Split

The cleaned dataset was divided into training and testing sets using a stratified 80:20 split.

**Parameters:**

- Test size: 20%
- Random state: 42
- Stratification: Enabled

| Set | Records |
|---|---:|
| Training set | 226,980 |
| Testing set | 56,746 |
| Total | 283,726 |

The training set contained 226,602 normal transactions and 378 fraudulent transactions.

The testing set contained 56,651 normal transactions and 95 fraudulent transactions.

Stratification preserved approximately the same class proportions in both sets.

### 3.4 Outlier Handling

The Interquartile Range (IQR) method was used to calculate outlier limits for the `Amount` feature using only the training data.

The approximate limits were:

- Lower limit: -102.55
- Upper limit: 186.09

Values outside these limits were capped rather than removed. The same limits calculated from the training set were applied to the testing set to prevent information leakage from the test set during preprocessing.

| Dataset | Values capped |
|---|---:|
| Training set | 25,316 |
| Testing set | 6,267 |

The processed feature sets were stored in `X_train_capped` and `X_test_capped`.

Since valid transaction amounts are ordinarily nonnegative, the negative lower limit did not affect valid nonnegative amounts.

The capping limits are specific to this dataset and preprocessing experiment.

## 4. Supervised Machine Learning Models

Three supervised machine learning algorithms were explored: Logistic Regression, Random Forest, and XGBoost.

The models learned from labeled transactions, where `Class = 0` represented normal transactions and `Class = 1` represented fraudulent transactions.

### 4.1 Logistic Regression

Logistic Regression was used as a baseline classification model.

It estimates the probability of a transaction belonging to a particular class and applies a classification threshold to generate a prediction.

Experiments included:

- Uncapped and capped `Amount` data.
- StandardScaler for feature scaling.
- SMOTE for addressing class imbalance.

SMOTE generates synthetic minority-class examples during training. In these experiments, SMOTE did not produce a major improvement in test performance.

Logistic Regression was useful as a baseline for comparison with the tree-based models.

### 4.2 Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The baseline model used:

- Number of trees: 100
- Class weighting: `balanced_subsample`
- Random state: 42

Additional experiments involved regularization and hyperparameter tuning.

The tuned Random Forest used:

- Number of trees: 50
- Maximum depth: 20
- Minimum samples split: 10
- Minimum samples leaf: 2
- Maximum features: `sqrt`

Random Forest can model nonlinear relationships and interactions between transaction features.

### 4.3 XGBoost

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially, with each new tree helping improve the model's predictions.

A baseline XGBoost model was trained before performing randomized hyperparameter search using Average Precision as the scoring metric.

The tuned XGBoost model used:

| Parameter | Value |
|---|---:|
| Number of estimators | 150 |
| Maximum depth | 5 |
| Learning rate | 0.1 |
| Subsample | 1.0 |
| Column sampling by tree | 1.0 |

A more strongly regularized version was also tested, but its performance was lower than that of the selected tuned model.

The tuned XGBoost model was selected as the preferred candidate based on the recorded Average Precision and F1 score.

## 5. Model Evaluation

### 5.1 Evaluation Metrics

**Average Precision (AP):** Summarizes precision-recall performance across classification thresholds. It is useful for evaluating models on highly imbalanced datasets.

**Precision:** Measures the proportion of transactions predicted as fraudulent that were actually fraudulent.

**Recall:** Measures the proportion of actual fraudulent transactions correctly detected.

**F1 Score:** The harmonic mean of precision and recall. It balances both metrics.

These metrics provide more useful information than accuracy alone when the number of fraudulent transactions is very small.

### 5.2 Final Model Comparison

The following table compares the two final candidate models.

| Model | Test AP | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Tuned Random Forest | 0.8113 | 0.9324 | 0.7263 | 0.8166 |
| Tuned XGBoost | 0.8139 | 0.8621 | 0.7895 | 0.8242 |

### 5.3 Interpretation of Results

**Tuned Random Forest**

- Achieved a precision of 0.9324.
- Had a lower recall than tuned XGBoost.
- Was more selective in its fraud predictions, producing fewer false-positive alerts relative to the number of alerts it generated.

**Tuned XGBoost**

- Achieved the highest Average Precision in this comparison.
- Detected a larger proportion of fraudulent transactions.
- Achieved the highest F1 score among the two final candidates.

Based on these results, tuned XGBoost was selected as the preferred final candidate because it provided a better balance between precision and recall.

The final choice depends on business requirements. Random Forest may be preferable when minimizing false alarms is especially important.

### 5.4 Approximate Fraud Detection Performance

The testing set contained 95 fraudulent transactions.

With a recorded recall of 0.7895, tuned XGBoost detected approximately 75 of these fraudulent transactions and missed approximately 20 at the evaluated classification threshold.

This is an approximate interpretation of the reported recall, not a substitute for a verified confusion matrix.

### 5.5 Evaluation Limitation

The testing set was reused during model comparison and experimentation. Therefore, the reported test results should be treated as provisional estimates rather than as an unbiased final assessment using a completely untouched holdout set.

A future evaluation should use a separate holdout dataset that is not used for model selection or threshold tuning.

## 6. Feature Importance

Feature importance was examined for the tuned XGBoost model to identify which input features contributed most to its learned decision patterns.

### 6.1 Important Features

| Feature | Importance |
|---|---:|
| V14 | 0.498880 |
| V4 | 0.056857 |
| V12 | 0.034336 |
| V8 | 0.033845 |
| V10 | 0.028994 |
| V3 | 0.024644 |
| V19 | 0.022862 |
| V13 | 0.021139 |
| Amount | 0.019463 |
| V21 | 0.018285 |

V14 had substantially higher importance than the other listed features in this model.

### 6.2 Interpretation

Feature importance indicates how the trained model used features when making predictions. It does not establish a causal relationship between a feature and fraudulent activity.

The `V1`–`V28` columns are anonymized PCA-derived features. Their original real-world meanings are not available in the dataset.

Therefore, V14 should not be interpreted as a directly identifiable transaction characteristic or as proof of fraud.

## 7. Model Saving and Verification

The trained models were saved using Joblib for reuse without retraining.

The following files were created:

- `saved_models/logistic_regression.joblib`
- `saved_models/tuned_random_forest.joblib`
- `saved_models/tuned_xgboost.joblib`

The saved model files were loaded and tested on a small sample of five test records. All three models loaded successfully and produced predictions.

The sample predictions were all class 0. This verifies that the saved models could produce predictions on those five records, but it does not independently establish overall model performance.

The Logistic Regression model contains its fitted preprocessing and classification pipeline. The tree-based models require the same feature preparation used during training, including the established `Amount` capping limits.

## 8. Streamlit Application

A Streamlit application was developed to demonstrate the selected XGBoost model through a user interface.

The application allows users to enter transaction features and receive a model prediction without manually running the training notebook.

### 8.1 Application Features

The application includes:

- A transaction input form.
- A field for transaction time.
- A field for transaction amount.
- Input fields for `V1` through `V28`.
- A Predict Transaction button.
- A predicted class indicating potential fraud or normal classification.
- An estimated fraud score based on the model output.
- Feature-contribution explanations for individual predictions.

### 8.2 Model Loading

The application loads `saved_models/tuned_xgboost.joblib` using Joblib.

The model is loaded through a cached resource function to avoid repeatedly loading the model during normal Streamlit reruns.

The application also checks whether the model file exists and displays an error if the file cannot be found or loaded.

### 8.3 Transaction Input

Users provide values for the 30 input features:

- `Time`
- `V1` through `V28`
- `Amount`

The application constructs a single-row DataFrame containing these features in the expected order.

The `Amount` value is capped using the limits established during training before the model generates a prediction.

For meaningful evaluation, the input features should come from a consistent feature-generation process or a known dataset record. Arbitrary values may not represent a realistic transaction.

### 8.4 Prediction Output

After submission, the application uses the trained model to generate a classification.

The result is displayed as either:

- **Potential Fraud Detected** when the model predicts class 1.
- **Classified as Normal** when the model predicts class 0.

The application also displays an estimated fraud score calculated from the model's predicted class probabilities.

This score should not automatically be interpreted as a calibrated real-world probability of fraud. The output is a model estimate and should not be treated as a definitive financial decision.

## 9. Individual Prediction Explanations

The application uses XGBoost feature contributions to explain individual transaction predictions.

These contributions help show how each input feature influences the model's output for a particular transaction.

### 9.1 Positive Contribution

A positive contribution pushes the model's raw output toward the fraud class.

For example, a feature with a positive contribution may increase the model's tendency to classify the transaction as fraudulent.

### 9.2 Negative Contribution

A negative contribution pushes the model's raw output toward the normal class.

It indicates that the feature moves the model's output away from the fraud class for that particular prediction.

### 9.3 Contribution Tables

The application separates the strongest positive and negative feature contributions into two groups:

- Features Pushing Toward Fraud
- Features Pushing Away from Fraud

This makes it easier to inspect the input features that most influenced a particular prediction.

### 9.4 Technical Interpretation

The XGBoost contribution calculation includes contributions from individual features and a baseline bias term. The application displays feature contributions separately from the bias term.

The contributions explain the model's raw output, typically on the log-odds scale for binary logistic classification. They are not direct percentage-point changes in the displayed fraud score.

Positive and negative contributions do not prove that a feature caused fraud or prevented fraud. They describe how the model used the supplied feature values when making that prediction.

## 10. Application Workflow

The overall application workflow is as follows:

1. The user opens the Streamlit application.
2. The application loads the saved tuned XGBoost model.
3. The user enters `Time`, `V1`–`V28`, and `Amount`.
4. The application constructs a DataFrame with the expected input features.
5. The `Amount` feature is capped using the established training-derived limits.
6. The trained model predicts the transaction class.
7. The application obtains the predicted class probabilities and calculates the displayed fraud score.
8. XGBoost calculates feature contributions for the transaction.
9. The application displays the predicted class, fraud score, and strongest feature contributions.

This workflow demonstrates how a trained supervised machine learning model can be integrated into a simple interactive application.

## 11. Project Files and Repository Organization

The project repository should contain the files required to understand, reproduce, and demonstrate the work.

An example structure is:

```text
Credit-Card-Fraud-Detection/
├── credit_card_fraud_detection.ipynb
├── app.py
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
├── model_comparison.csv
├── .gitignore
└── saved_models/
    ├── logistic_regression.joblib
    ├── tuned_random_forest.joblib
    └── tuned_xgboost.joblib
```

This is the intended project structure. The actual repository should be checked to confirm which files are present and uploaded.

### 11.1 File Descriptions

| File | Purpose |
|---|---|
| `credit_card_fraud_detection.ipynb` | Data analysis, preprocessing, model training, and experiments |
| `app.py` | Streamlit application for transaction prediction |
| `requirements.txt` | Python dependencies required by the project |
| `README.md` | Short project overview, setup instructions, and links |
| `DOCUMENTATION.md` | Detailed project methodology, results, and limitations |
| `model_comparison.csv` | Recorded model comparison results |
| `.gitignore` | Excludes files that should not be committed |
| `saved_models/` | Serialized trained model files |

### 11.2 Dataset and Security

The original `creditcard.csv` dataset should not be uploaded to GitHub unless there is a specific reason and the repository's data-sharing requirements permit it.

Instead, the README should link to the original Kaggle dataset and explain how to obtain it.

Kaggle API tokens, passwords, credentials, and other secrets must never be committed to the repository.

The `.gitignore` file should exclude the dataset, temporary files, local environments, cache files, and credentials where applicable.

## 12. Running the Application Locally

The following commands describe the usual local setup for the Streamlit application. The dependency versions should match those verified with the saved model.

### Step 1: Install Python Dependencies

From the project directory, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Check the Model File

Ensure that the selected model exists at:

```text
saved_models/tuned_xgboost.joblib
```

### Step 3: Launch the Application

Run:

```bash
streamlit run app.py
```

Streamlit should open the application in a browser or provide a local URL.

### Step 4: Test a Prediction

Enter a consistent set of transaction features and submit the form.

Verify that the application displays the prediction, estimated fraud score, and feature-contribution explanations.

The commands above describe the intended procedure. Successful execution should be confirmed in the target environment.

## 13. Deployment

The application is intended for demonstration through Streamlit Community Cloud.

The deployment process generally involves:

1. Uploading the required project files to GitHub.
2. Ensuring that `app.py`, `requirements.txt`, and the selected model file are available in the repository.
3. Selecting the correct repository and branch in Streamlit Community Cloud.
4. Specifying `app.py` as the application entry point.
5. Allowing the platform to install the required dependencies.
6. Opening the deployed application and testing its main functionality.

The deployment configuration must use versions compatible with the saved XGBoost model and its training environment.

**Deployment status:** The actual live deployment URL and successful end-to-end cloud testing should be recorded only after they have been verified.

## 14. Limitations

The project has several limitations.

### 14.1 Class Imbalance

Fraudulent transactions account for only a small percentage of the dataset. This makes model evaluation and fraud detection challenging.

### 14.2 Anonymized Features

The original meanings of `V1`–`V28` are not available, which limits direct interpretation of the transaction characteristics.

### 14.3 Reuse of the Test Set

The test set was used repeatedly during experimentation and model comparison. Consequently, the reported test metrics may be optimistic and should not be treated as a completely unbiased estimate of performance on unseen data.

### 14.4 Model Score Interpretation

The displayed fraud score comes from the model's predicted class probabilities. It is not necessarily a calibrated estimate of the real-world probability of fraud.

### 14.5 Feature Contributions

Feature contributions describe the model's behavior, not the causal reason for a transaction being fraudulent.

### 14.6 Input Quality

The application accepts manually entered numerical features. Arbitrary inputs may not correspond to valid transactions or to the feature distribution seen during training.

### 14.7 Production Readiness

The project is a learning prototype. It has not been established as a production-grade fraud detection system. Real deployment would require additional validation, monitoring, security controls, and business-specific decision rules.

## 15. Future Enhancements

Potential improvements include:

1. Evaluate the selected model using a completely untouched holdout set.
2. Choose the classification threshold based on the cost of missed fraud and false-positive alerts.
3. Add confusion matrices and precision-recall curves to the application.
4. Improve input validation and error handling.
5. Evaluate the model on newer transaction data to investigate changes in fraud patterns.
6. Investigate probability calibration if the displayed score will be interpreted as a real-world risk probability.
7. Add automated tests for model loading, feature ordering, preprocessing, and prediction output.
8. Improve deployment monitoring and dependency management.
9. Explore ways to obtain meaningful explanations from anonymized features without claiming that they reveal the original transaction attributes.

These are possible future improvements; they should be marked as completed only after implementation and verification.

## 16. Conclusion

This project demonstrates an end-to-end supervised machine learning workflow for credit card fraud detection. The work includes duplicate removal, class imbalance analysis, stratified train-test splitting, outlier capping, model training, hyperparameter tuning, model evaluation, feature importance analysis, and model saving.

Logistic Regression, Random Forest, and XGBoost were explored. Among the two final candidate models, tuned XGBoost achieved a recorded test Average Precision of 0.8139 and an F1 score of 0.8242. Tuned Random Forest achieved higher precision at 0.9324, demonstrating the trade-off between detecting a larger proportion of fraudulent transactions and reducing false-positive alerts.

The selected tuned XGBoost model was also integrated into a Streamlit application designed to accept transaction features, display a prediction and estimated fraud score, and show individual feature contributions.

The results provide a useful learning demonstration of supervised classification for imbalanced data. However, the repeated use of the test set means that a fresh holdout evaluation is required before making strong claims about generalization performance. The project should be considered a prototype rather than a production-ready financial fraud detection system.