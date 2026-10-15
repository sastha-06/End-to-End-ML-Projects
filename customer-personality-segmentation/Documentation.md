## 1. Project Overview

Customer Personality Segmentation is an unsupervised machine learning project that groups customers based on their demographic information, income, purchasing behaviour, and product spending patterns.

The main objective is to identify different types of customers so that businesses can better understand their customers and develop suitable marketing strategies.

This project uses the **Customer Personality Analysis** dataset from Kaggle. We explored customer data, performed data preprocessing, applied clustering algorithms, evaluated the results, and prepared a Streamlit application to predict the customer segment for a new customer.

## 2. Problem Statement

Businesses often have customers with different income levels, spending habits, family situations, and purchasing preferences. Treating all customers in the same way may lead to ineffective marketing campaigns.

The objective of this project is to group similar customers using unsupervised machine learning and understand the characteristics of each group.

### Objectives

- Understand customer demographics and purchasing behaviour.
- Clean and preprocess the dataset for machine learning.
- Apply clustering algorithms to identify customer segments.
- Compare clustering results using evaluation metrics.
- Interpret the characteristics of the identified customer groups.
- Build an interactive application to predict the segment of a new customer.

## 3. Dataset Description

**Dataset:** Customer Personality Analysis

**Source:** [Kaggle – Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

The dataset contains customer information such as:

- **Demographic details:** Year of birth, education, marital status and income.
- **Family information:** Number of children and teenagers at home.
- **Spending behaviour:** Spending on wines, fruits, meat products, fish products, sweets and gold products.
- **Purchasing behaviour:** Web purchases, catalogue purchases, store purchases and deal purchases.
- **Website activity:** Number of website visits per month.
- **Campaign information:** Responses to previous marketing campaigns.

The original dataset contained **2,240 records and 29 columns**. We identified 24 missing values in the Income column and checked the dataset for duplicate records.

## 4. Project Workflow

The project followed these major stages:

1. Dataset collection and loading.
2. Exploratory data analysis.
3. Data cleaning and preprocessing.
4. Feature engineering and encoding.
5. Outlier handling and feature scaling.
6. Clustering using K-Means and DBSCAN.
7. Evaluation and comparison of clustering results.
8. Customer segment interpretation.
9. Model and preprocessing artifact saving.
10. Streamlit application development.

## 5. Exploratory Data Analysis (EDA)

Exploratory data analysis was performed to understand the structure and quality of the dataset before model training.

The following checks were carried out:

- Examined the number of rows and columns.
- Checked data types and missing values.
- Identified duplicate records.
- Examined distributions of numerical features.
- Studied customer income, age, spending and purchasing behaviour.
- Identified potentially unrealistic ages and extreme income values.

This stage helped us understand the dataset and decide which preprocessing operations were necessary.

## 6. Data Cleaning and Preprocessing

Data preprocessing was one of the most important stages because clustering algorithms depend on the quality and scale of the input features.

### 6.1 Removing Unnecessary Columns

We removed columns that were not required for the chosen customer segmentation approach:

- `ID` – customer identifier.
- `Dt_Customer` – customer enrolment date.
- `AcceptedCmp1` to `AcceptedCmp5` – previous campaign acceptance indicators.
- `Response` – campaign response indicator.
- `Complain` – complaint indicator.
- `Z_CostContact` and `Z_Revenue` – constant or non-informative fields.

These columns were excluded to focus the model on customer demographics and purchasing behaviour rather than identifiers or campaign-response information.

### 6.2 Handling Missing Values

The Income column contained 24 missing values.

We replaced missing Income values with the **median income**. Median imputation was chosen because it is less affected by extreme values than mean imputation.

### 6.3 Creating the Age Feature

The `Year_Birth` column was converted into an Age feature using the reference year used in the project:

```python
Age = 2014 - Year_Birth
```

We then removed `Year_Birth` because Age was more useful for interpreting customer profiles.

We filtered out records with ages outside the range of 18 to 100 to remove implausible or unsuitable age values for this analysis.

### 6.4 Creating the Children Feature

The `Kidhome` and `Teenhome` columns were combined into a single feature:

```python
Children = Kidhome + Teenhome
```

This simplified the representation of the number of children and teenagers living in a customer's household. The original `Kidhome` and `Teenhome` columns were then removed.

### 6.5 Encoding Categorical Variables

Machine learning algorithms require numerical inputs. Therefore, the categorical columns `Education` and `Marital_Status` were converted into numerical indicator columns using one-hot encoding.

This allowed the model to process categories without assigning them an artificial numerical order.

### 6.6 Handling Income Outliers

We used the Interquartile Range (IQR) method to calculate lower and upper bounds for Income.

Values outside these bounds were capped rather than deleting the corresponding customers. This process is called **outlier capping**.

The purpose was to reduce the influence of extreme income values while retaining customer records.

### 6.7 Final Feature Preparation

After cleaning, feature engineering, categorical encoding and filtering, the dataset contained **2,237 customer records and 28 model features**.

The target cluster was not available beforehand because this is an unsupervised learning problem. The clusters were generated by the algorithms during training.

## 7. Feature Scaling

We applied `StandardScaler` to standardise the model inputs.

Scaling was necessary because features such as Income and product spending have different numerical ranges. Without scaling, features with larger values could dominate distance-based clustering algorithms.

The scaled dataset had the following shape:

- Customer records: 2,237
- Model features: 28

The fitted scaler was saved so that new customer inputs could undergo the same transformation when using the application.

## 8. Clustering Algorithms

We explored two unsupervised clustering algorithms: K-Means and DBSCAN.

### 8.1 K-Means Clustering

K-Means groups observations into a predefined number of clusters by assigning each observation to the nearest cluster centroid and iteratively updating the centroids.

We tested different values of K and evaluated their silhouette scores.

| Number of clusters (K) | Silhouette score |
|---:|---:|
| 2 | 0.2001 |
| 3 | 0.1269 |
| 4 | 0.1463 |
| 5 | 0.1406 |
| 6 | 0.1071 |

The best silhouette score among the tested K-Means configurations was obtained with **K = 2**.

Therefore, we selected a K-Means model with two clusters, using `random_state=42` and `n_init=10` for reproducibility and multiple centroid initialisations.

### 8.2 DBSCAN Clustering

DBSCAN is a density-based clustering algorithm. Unlike K-Means, it does not require the number of clusters to be specified in advance. It can also identify observations that do not belong to a sufficiently dense group as noise.

We experimented with:

- `eps = 5.5`
- `min_samples = 10`

The resulting labels were:

| Label | Number of customers |
|---|---:|
| Noise (-1) | 24 |
| Cluster 0 | 2,085 |
| Cluster 1 | 52 |
| Cluster 2 | 76 |

The silhouette score for the DBSCAN clusters, excluding noise observations, was approximately **0.2345**.

This score is not directly comparable to the K-Means score because DBSCAN produced a different number of clusters and noise observations were excluded from the calculation.

DBSCAN was explored as an alternative, while K-Means was selected for the current Streamlit prediction workflow.

## 9. Model Evaluation

We used the silhouette score to evaluate how well the clusters were separated.

A higher silhouette score generally indicates that observations are closer to their own cluster than to other clusters.

The best tested K-Means configuration achieved a silhouette score of **0.2001**. This indicates that the customer groups have limited separation and that some customers may share characteristics across the two segments.

The selected model is useful for exploratory customer grouping, but its segments should not be treated as perfectly separated or definitive customer types.

## 10. Customer Segment Analysis

The selected K-Means model produced two customer segments.

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customer count | 1,366 | 871 |
| Percentage of customers | 61.1% | 38.9% |
| Average income | 39,327.55 | 71,501.48 |
| Average age | 44.09 | 46.68 |
| Average children in household | 1.25 | 0.48 |
| Average wine spending | 114.20 | 601.66 |
| Average meat-product spending | 38.40 | 368.47 |
| Average web purchases | 3.01 | 5.77 |
| Average store purchases | 4.01 | 8.60 |
| Average website visits per month | 6.42 | 3.60 |

*Values are based on the customer profiles calculated during the project. Income and spending values are in the dataset's original units.*

### Cluster 0: Lower-Spending Customer Segment

Cluster 0 contains approximately 61.1% of the customers.

Its average income and spending values are lower than those of Cluster 1. Customers in this group also have more children on average, make fewer web and store purchases, and visit the website more frequently.

**Possible business interpretation:** This segment may respond to affordable product recommendations, discounts and promotions designed around household needs.

### Cluster 1: Higher-Spending Customer Segment

Cluster 1 contains approximately 38.9% of the customers.

This group has higher average income, wine spending, meat-product spending and purchase counts. Customers also visit the website less frequently on average.

**Possible business interpretation:** This segment may be suitable for premium product recommendations, personalised offers and loyalty programmes.

These interpretations are based on observed averages. They describe patterns in the dataset and do not prove the motivations of individual customers.

## 11. Saving the Trained Model

After selecting the K-Means model, we saved the trained model and preprocessing components for reuse.

- `kmeans_customer_model.pkl` – trained K-Means model.
- `customer_scaler.pkl` – fitted `StandardScaler`.
- `preprocessing_config.pkl` – preprocessing configuration, including the reference year, income handling values and feature names.
- `customer_feature_names.pkl` – optional saved feature-name list.
- `dbscan_customer_model.pkl` – optional DBSCAN model for experimentation.

Saving these files avoids retraining the model whenever the application starts.

The preprocessing configuration and feature order must match the exact data used to train the model. Otherwise, predictions may be incorrect or fail because of feature mismatches.

## 12. Streamlit Application

We prepared an interactive Streamlit application to make the model usable without requiring users to execute the training notebook.

The application allows a user to enter customer information, including:

- Year of birth.
- Education and marital status.
- Income.
- Number of children and teenagers at home.
- Product spending.
- Purchasing behaviour.
- Website visits and deal purchases.

### Application Workflow

1. The user enters customer details.
2. The application handles missing or extreme income values according to the training configuration.
3. Age and total children are calculated.
4. Categorical values are converted using the same one-hot encoding scheme used during training.
5. The input features are aligned to the saved feature names and order.
6. The saved scaler transforms the features.
7. The trained K-Means model predicts the cluster.
8. The application displays the assigned segment and an explanation based on differences between the customer's details and the segment profile.

The explanation compares selected characteristics, such as income, age, household size, spending and purchase frequency, with the corresponding cluster averages.

This is a descriptive comparison, not a formal causal explanation of why the algorithm assigned the customer to a particular cluster.

## 13. Technology Stack

- **Python:** Main programming language.
- **Pandas:** Data loading, cleaning and manipulation.
- **NumPy:** Numerical operations.
- **Scikit-learn:** StandardScaler, K-Means, DBSCAN and clustering evaluation.
- **Joblib:** Saving and loading trained models and preprocessing artifacts.
- **Matplotlib / Seaborn:** Exploratory visualisation, where used.
- **Streamlit:** Interactive web application.
- **Jupyter Notebook / Google Colab:** Model development and experimentation.
- **GitHub:** Version control and project documentation.

## 14. Project Structure

```text
customer-personality-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
│
└── savedmodel/
    ├── kmeans_customer_model.pkl
    ├── customer_scaler.pkl
    ├── preprocessing_config.pkl
    ├── customer_feature_names.pkl
    └── dbscan_customer_model.pkl
```

The DBSCAN model and separate feature-name file are optional if they are not required by the final application.

## 15. Challenges Faced

During the project, we worked through several practical machine learning challenges:

- Handling missing values without unnecessarily removing customer records.
- Identifying unrealistic ages and reducing the effect of extreme income values.
- Converting categorical variables into numerical features.
- Scaling features with different numerical ranges.
- Selecting the number of clusters using silhouette scores.
- Understanding the differences between K-Means and DBSCAN.
- Interpreting cluster profiles in a meaningful business context.
- Saving the model and preprocessing steps for future use.
- Ensuring that new application inputs contain the same features, in the same order, as the training data.
- Preparing an explanation of the predicted customer segment for the Streamlit interface.

These steps helped us understand that building a machine learning application involves more than training a model. Data preparation, evaluation, interpretation and consistent preprocessing are equally important.

## 16. Limitations

- The silhouette score for the selected K-Means model is modest, indicating limited cluster separation.
- The two customer groups are based on the available dataset and may not represent all customer populations.
- Customer behaviour may change over time, so the model may require retraining with updated data.
- The cluster labels do not represent predefined categories such as loyal or high-value customers; those interpretations must be validated separately.
- The model identifies patterns and associations but does not establish causal relationships.
- The DBSCAN model cannot directly predict clusters for unseen observations using the standard scikit-learn `predict()` interface.

## 17. Future Enhancements

Possible improvements include:

- Exploring additional clustering algorithms and hyperparameters.
- Comparing clustering quality using more than one evaluation metric.
- Improving the separation and interpretability of customer groups.
- Adding interactive visualisations of cluster characteristics.
- Including customer-level reports and downloadable results.
- Evaluating the segments on additional customer data.
- Deploying the Streamlit application so that it can be accessed online.
- Monitoring model performance and retraining when customer behaviour changes.

## 18. Conclusion

The Customer Personality Segmentation project demonstrates an end-to-end unsupervised machine learning workflow. We explored customer data, handled missing values and outliers, engineered useful features, scaled the data, experimented with clustering algorithms, evaluated the results and interpreted the resulting customer segments.

K-Means with two clusters was selected because it achieved the best silhouette score among the tested K-Means configurations. The resulting groups showed differences in income, household size, spending and purchasing behaviour.

The project also prepared the model and preprocessing components for use in a Streamlit application, where a new customer's information can be processed and assigned to a segment.

Overall, this project provided practical experience in data preprocessing, unsupervised learning, model evaluation, customer behaviour analysis and machine learning application development.

## 19. References

1. [Customer Personality Analysis Dataset – Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [Pandas Documentation](https://pandas.pydata.org/docs/)
