from pathlib import Path

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
import joblib
from app.schemas import CustomerInput
from app.preprocessing import engineer_features, align_columns

app = FastAPI(
    title="Customer Churn Prediction API",
    description="This API predicts customer churn based on input features.",
    version="1.0.0",
)

# --- Load model artifacts once, at startup (not per-request) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

xgb_model = joblib.load(MODEL_DIR / "model_xgb_v1.pkl")
scaler = joblib.load(MODEL_DIR / "scaler_v1.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns_v1.pkl")



@app.get("/")
def homepage():
    return {"message": "Welcome to the FastAPI application!"}


@app.get("/health")
def health():
    return {"Status": "Healthy", "Message": "The API is running Successfully!"}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        #Feature Engineering and Column Alignment
        df = engineer_features(customer)

        #Align the columns of the input DataFrame to match the training feature columns
        df = align_columns(df, feature_columns)
    
        #Scale Features
        X_scaled = scaler.transform(df)

        #Predict Churn Probability and Class
        proba = xgb_model.predict_proba(X_scaled)[0][1]

        #Probability threshold for classification (0.5)
        prediction = int(proba >= 0.5)

        return {
            "churn_prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(float(proba), 4),
            "model_used": "xgboost_v1",
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}")
