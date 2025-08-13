from fastapi import FastAPI, HTTPException
from src.api.models import PredictRequest, PredictResponse
from src.models.predict import CreditRiskPredictor
from src.utils.logging import logger
from config.settings import settings
import mlflow.pyfunc

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Load model
try:
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{settings.model_name}/{settings.model_stage}"
    )
    predictor = CreditRiskPredictor(model)
    logger.info(f"Loaded model: {settings.model_name}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        risk_probability = predictor.predict(request.features)
        return PredictResponse(risk_probability=risk_probability)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")