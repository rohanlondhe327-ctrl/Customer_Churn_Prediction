import pandas as pd

SERVICE_COLS = [
    "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Every column that was one-hot encoded during training everything
# that was still text after the engineered features were added.
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6m", "6-12m", "12-24m", "24-48m", "48m+"]


def engineer_features(customer) -> pd.DataFrame:
    """
    Reproduce the training notebook's feature engineering for ONE request.
    """
    customer_dict =(
        customer.model_dump()
        if hasattr(customer, "model_dump")
        else dict(customer)
    )
    
    df = pd.DataFrame([customer_dict])

    # --- engineered binary flags (same logic as notebook cell 6) ---
    df["IsMonthToMonthContract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["NoInternetService"] = (df["InternetService"] == "No").astype(int)

    # --- tenure bucket, built manually instead of via get_dummies ---
    tenure_val = df.loc[0, "tenure"]
    tenure_group = pd.cut([tenure_val], bins=TENURE_BINS, labels=TENURE_LABELS)[0]
    for label in TENURE_LABELS:
        df[f"TenureGroup_{label}"] = 1 if tenure_group == label else 0

    # --- ratio feature (notebook cell 8) ---
    df["AvgMOnthlySpend"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # --- number of active services (notebook cell 9) ---
    df["NumServices"] = (df[SERVICE_COLS] != "No").sum(axis=1)

    # --- manual one-hot encoding for every remaining categorical column ---
    for col in CATEGORICAL_COLS:
        value = df.loc[0, col]
        df[f"{col}_{value}"] = 1
        df.drop(columns=[col], inplace=True)

    return df


def align_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    return df.reindex(columns=feature_columns, fill_value=0)
