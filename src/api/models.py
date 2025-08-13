from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="RFM and engineered features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [30.0, 5.0, 1500.0, 300.0, 150.0, 0.0, 0.0, 1.0, 2.0]
            }
        }

class PredictResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability score")
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool