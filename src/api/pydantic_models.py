from pydantic import BaseModel
from typing import List, Any

class PredictRequest(BaseModel):
    data: List[Any]  # Should match the model's feature order

class PredictResponse(BaseModel):
    risk_probability: float
