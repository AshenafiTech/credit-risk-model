from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse
import mlflow.pyfunc
import os

app = FastAPI()

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'CreditRiskBestModel')
MODEL_STAGE = os.getenv('MLFLOW_MODEL_STAGE', 'Production')

# Load model from MLflow Model Registry
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    # The input should be a list of features matching the model's expected order
    data = [request.data]
    prob = model.predict(data)[0]
    return PredictResponse(risk_probability=float(prob))
