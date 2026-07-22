# Customer Churn Prediction API

This project predicts customer churn using a FastAPI service backed by trained machine learning artifacts.
The current application exposes a lightweight REST endpoint for sending a customer record and receiving a churn prediction probability.

## Project Overview

Customer churn is a major business problem in the telecom industry. This project trains and stores churn prediction models, then serves them through a FastAPI application so predictions can be requested in real time.

The current implementation uses:
- XGBoost as the deployed prediction model
- StandardScaler for feature scaling
- Manual feature engineering to match training-time columns exactly
- Pydantic request validation for API inputs

## Dataset

Source: Telco Customer Churn dataset from Kaggle
Size: approximately 7,000 customers
Target: churn label (Yes / No)

## Current FastAPI Workflow

The API is built in [app/main.py](app/main.py) and follows this flow:

1. Receive a JSON payload through the POST request body
2. Validate the input using the schema in [app/schemas.py](app/schemas.py)
3. Engineer the same feature set used during training in [app/preprocessing.py](app/preprocessing.py)
4. Align the request row to the exact model feature columns
5. Scale the feature vector with the saved scaler
6. Produce churn probability and binary churn prediction from the XGBoost model

## API Endpoints

- `GET /`  
  Returns a welcome message

- `GET /health`  
  Returns API health information

- `POST /predict`  
  Accepts a customer input payload and returns:
  - `churn_prediction`
  - `churn_probability`
  - `model_used`

## Example Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 65.0,
  "TotalCharges": 780.0
}
```

## Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

## Tech Stack

- Python
- FastAPI
- Pydantic
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn
- joblib
- KaggleHub

## Notes

The repository also keeps a notebook-based exploration workflow under [notebook/notebook.ipynb](notebook/notebook.ipynb), but the deployed prediction service is the FastAPI application in [app/main.py](app/main.py).

