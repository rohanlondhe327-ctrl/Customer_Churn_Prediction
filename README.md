Customer Churn Prediction :-
Predicting customer churn using XGBoost and Random Forest on the Telco Customer Churn dataset. 
The goal is to identify customers likely to leave so the business can take proactive retention actions.

Project Overview :-
Customer churn is a major business problem in the telecom industry. This project builds and compares two machine learning models 
XGBoost and Random Forest to predict whether a customer will churn, with a focus on minimizing false alarms (high precision on the churn class).

Dataset :-
Source: Telco Customer Churn – Kaggle
Size: ~7,000 customers, 21 features
Target: Churn (Yes / No)


Workflow :-

1.Data Preprocessing — Handled missing values, dropped irrelevant columns, encoded target variable
2.Feature Engineering — Created domain-driven features:
    IsMonthToMonthContract — high-risk contract type
    IsElectronicCheck — payment method linked to churn
    NoInternetService — service usage indicator
    TenureGroup — bucketed customer tenure
    AvgMonthlySpend — intensity ratio of charges
    NumServices — total services used per customer
3.Encoding — One-hot encoding for all categorical variables
4.Class Imbalance — Applied SMOTE (k=5) to balance training data to 50-50
5.Feature Scaling — StandardScaler applied after SMOTE
6.Hyperparameter Tuning — GridSearchCV with 5-fold cross-validation on both models
7.Evaluation — Classification report on held-out test set


Tech Stack:-

Python, Pandas, NumPy
Scikit-learn (GridSearchCV, StandardScaler, StratifiedKFold)
XGBoost
imbalanced-learn (SMOTE)
KaggleHub




