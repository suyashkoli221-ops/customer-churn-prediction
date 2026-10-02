
import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """
    Predict customer churn using the saved model.
    """

    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]

    probability = model.predict_proba(customer_df)[0][1]

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk": risk
    }


# Test prediction
if __name__ == "__main__":

    sample_customer = {
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
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0
    }

    result = predict_churn(sample_customer)

    print("Prediction result:")
    print(result)

    if result["prediction"] == 1:
        print("Customer may churn.")
    else:
        print("Customer is predicted to stay.")