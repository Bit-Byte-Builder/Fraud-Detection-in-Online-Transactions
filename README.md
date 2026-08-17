# 💳 AI-Powered Financial Fraud Detection System
An end-to-end machine learning system for detecting potentially fraudulent online transactions and translating model predictions into actionable risk decisions.


## 🚀 Live Demo

**Try the deployed application:**
👉 https://bit-byte-builder-fraud-detection-in-online-transacti-app-2ccd8d.streamlit.app/

---

### ✨ Key Features

- 🔍 Real-time Fraud Prediction
- 🤖 Machine Learning Powered (Random Forest)
- 📊 Fraud Probability Score
- ⚠️ Risk Level Assessment
- ✅ Business Action Recommendation
- 📈 Interactive Streamlit Dashboard
- 🧠 Advanced Feature Engineering

- ## 🖼️ Application Preview

### 🏠 Home Screen

![Home Screen](screenshots/01_Home_Page.png)

### 📊 Prediction Dashboard

![Prediction Dashboard](screenshots/03_Prediction_Result_Low_Risk.png)

### ✅ Recommended Action

![Recommended Action](screenshots/04_Approved_Transaction_Recommendation_Engineered_Features.png)


# Fraud Detection in Online Transactions

## 📌 Project Overview

This project builds an end-to-end machine learning pipeline to detect fraudulent online transactions.

The workflow covers:

**Data Validation → Exploratory Data Analysis (EDA) → Feature Engineering → Preprocessing → Model Training → Model Evaluation → Threshold Tuning → Business Recommendations → Deployment**

The project focuses not only on predicting fraudulent transactions, but also on translating model predictions into practical fraud-risk decisions.

---

## 💼 Business Problem

PaySphere Digital Payments Pvt. Ltd. processes millions of transactions across UPI, cards, net banking, and digital wallets.

As transaction volumes increase, the fraud detection team needs to identify suspicious transactions while minimising unnecessary disruption to legitimate customers.

The key challenges include:

- Increasing fraudulent activity
- Class imbalance
- False negatives
- False positives
- Customer friction caused by unnecessary transaction blocks

Therefore, the objective is to develop a machine learning solution that can identify potentially fraudulent transactions while maintaining a practical balance between fraud detection and false alerts.

---


## 🎯 Objectives

- Detect fraudulent transactions accurately.
- Reduce false positives and false negatives.
- Engineer meaningful behavioral and transaction-based features.
- Build and evaluate machine learning classification models.
- Handle class imbalance appropriately.
- Tune the fraud decision threshold for business use.
- Generate fraud probability and risk-level assessments.
- Provide actionable fraud-risk insights and recommendations.
- Deploy the trained model through an interactive Streamlit application.

---

## 📊 Dataset

The dataset contains 50,000 online transactions with features such as amount, payment method, international flag, merchant category, device trust score, IP risk score, OTP success rate, fraud history, disputes, and time-based patterns. The final target variable is 'isfraud'.


## 🔍 Data Validation

The notebook checks for missing values, duplicate transaction IDs, negative transaction amounts, invalid labels, timestamp issues, and out-of-range values. The dataset used in the project passed these validation checks.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand fraud prevalence and identify relationships between fraudulent transactions and important transaction, customer, device, payment, and behavioural characteristics.

The findings from EDA were used to guide feature engineering and model development.

## 🧠 Feature Engineering

Several fraud-relevant features were created from the raw dataset, including:
- Amount deviation ratio.
- Amount deviation difference.
- IP-device risk interaction.
- Combined change flag.
- Weekend-night flag.
- Weak authentication flag.
- High-risk payment flag.
- Customer transaction sequence.
- Customer-device pair count.
- Customer-merchant pair count.

These engineered features help capture behavioral anomalies and risk patterns more effectively than raw fields alone.

## 🤖 Model Training

The notebook compares multiple classification models, including:

# Logistic Regression

Used as a benchmark model to establish a baseline for comparison.

# Random Forest

Used as the primary model for deeper evaluation and deployment.

The modelling workflow includes:

1. Data preprocessing
2. Feature engineering
3. Class-imbalance handling
4. Model training
5. Probability prediction
6. Model evaluation
7. Decision-threshold tuning

## 📏 Model Evaluation

Because fraud detection is an imbalanced classification problem, accuracy alone is not sufficient for evaluating model performance.

The project evaluates the model using:

> ROC-AUC
> PR-AUC
> Precision
> Recall
> F1 Score
> Accuracy

## 📊 Model Performance

### Initial Model Evaluation (Default Threshold = 0.5)
| Metric              |    Score |
| ------------------- | -------: |
| ROC-AUC             | 0.688956 |
| PR-AUC              | 0.198123 |
| Precision (Class 1) |     0.22 |
| Recall (Class 1)    |     0.39 |
| F1 Score (Class 1)  |     0.28 |
| Accuracy            |     0.80 |


## 🎚️ After Threshold Tuning

The fraud decision threshold was tuned using F1 Score as the optimisation criterion.
| Metric              | Score |
| ------------------- | ----: |
| Precision (Class 1) |  0.23 |
| Recall (Class 1)    |  0.37 |
| F1 Score (Class 1)  |  0.28 |
| Accuracy            |  0.81 |


The tuned threshold slightly improves precision and accuracy while reducing recall.

This demonstrates the precision-recall trade-off involved in fraud detection.

A lower threshold may identify more potentially fraudulent transactions but can also increase false positives. A higher threshold may reduce false alerts but can allow more fraudulent transactions to pass undetected.

The appropriate threshold should therefore depend on the relative business costs of false positives and false negatives.

Note: The reported metrics are based on the evaluation setup documented in the project notebook and should be interpreted in the context of the dataset and validation methodology. They should not be considered production-level fraud-detection performance.


## 🖥️ Streamlit Application

The trained model is integrated into an interactive Streamlit application.

Application Capabilities
Real-time transaction fraud prediction
Fraud probability score
Risk-level assessment
Transaction risk classification
Business action recommendation
Interactive prediction dashboard

The application allows users to provide transaction information and receive a model-generated fraud-risk assessment.

## 💡 Business Insights
The analysis identified several patterns associated with increased fraud risk:
- Device changes can be important fraud indicators.
- Location changes can indicate potentially suspicious behaviour.
- International transactions show a higher fraud rate than domestic transactions.
- Fraud risk increases when risky behavior combines with weak authentication.
- Behavioural and interaction-based features can provide additional predictive information beyond basic transaction attributes.
- Threshold tuning is useful when balancing fraud capture against false positives.

  Important: These observations are specific to the analysed dataset and should be validated against production data before being used as operational fraud rules.

## 💾 Saved Artifacts

The final deployed model and supporting metadata are stored in the models/ directory.

models/
├── fraudpipeline.pkl
└── modelmetadata.pkl

fraudpipeline.pkl
Contains the trained fraud-detection pipeline/model used by the Streamlit application.

modelmetadata.pkl
Contains supporting model metadata required by the application.

## 🏗️ Project Structure

Fraud-Detection-in-Online-Transactions/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── fraudpipeline.pkl
│   └── modelmetadata.pkl
│
├── notebooks/
│   └── Fraud_Detection_Capstone_Project-1.ipynb
│
└── screenshots/
    ├── 01_Home_Page.png
    ├── 03_Prediction_Result_Low_Risk.png
    └── 04_Approved_Transaction_Recommendation_Engineered_Features.png

 ## 🛠️ Tech Stack
 
# Programming Language
Python

# Data Analysis
Pandas
NumPy

# Data Visualization
Matplotlib
Seaborn

# Machine Learning
Scikit-learn
XGBoost
Imbalanced-learn

# Model Management
Joblib

# Application & Deployment
Streamlit

## 📦 Requirements

The project requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- joblib
- streamlit
  
  All dependencies are also listed in requirements.txt.

## 🔮 Future Improvements

Potential improvements to the project include:

Hyperparameter optimisation for Random Forest and XGBoost.
- Cross-validation for more robust performance estimation.
- Temporal validation to better simulate real-world fraud detection.
- Probability calibration.
- Cost-sensitive threshold optimisation.
- SHAP-based model explainability.
- Batch transaction scoring.
- Real-time transaction monitoring.
- Data-drift monitoring.
- Model-performance monitoring.
-Integration with a production database or transaction API.
- Automated retraining pipelines.

## ⚠️ Limitations

This project is designed as a machine learning and portfolio project.

Model performance depends on:

- Dataset quality
- Feature availability
- Class distribution
- Validation methodology
- Model selection
- Decision-threshold selection

A production fraud-detection system would require additional considerations such as:

- Real-time transaction streams
- Data drift
- Concept drift
- Model monitoring
- Explainability
- Fraud investigation workflows
- Cost-sensitive decision-making
- Security controls
- Regulatory and compliance requirements

## 📚 Learning Outcomes

This project provided practical experience with:

- Data validation
- Exploratory Data Analysis
- Feature engineering
- Classification modelling
- Imbalanced datasets
- Model evaluation
- Precision-recall trade-offs
- Threshold optimisation
- Model persistence
- Streamlit deployment
- Translating machine learning predictions into business recommendations

## Conclusion
This project demonstrates a complete fraud detection pipeline for online transactions, from validation and feature engineering to model evaluation and deployment. The final model and metadata are saved in the 'models/' folder for Streamlit use and future scoring.

## 👨‍💻 Author
**Sachin Kumar**
