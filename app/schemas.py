
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: str = Field(..., description="Gender of the customer (Male/Female)")
    SeniorCitizen: int = Field(..., description="Whether the customer is a senior citizen (0/1)")
    Partner: str = Field(..., description="Has a partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Has multiple lines (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Has online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Has online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Has device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Has tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Has streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Has streaming movies (Yes/No/No internet service)")
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Uses paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., description="The total amount charged to the customer")
