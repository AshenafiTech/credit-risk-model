import numpy as np
from src.utils.logging import logger

class CreditRiskPredictor:
    def __init__(self, model):
        self.model = model
        
    def predict(self, features: list) -> float:
        """Predict risk probability for given features"""
        try:
            # Ensure features is 2D array
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(features_array)[0][1]
            else:
                prob = self.model.predict(features_array)[0]
            
            logger.info(f"Prediction made: {prob:.4f}")
            return float(prob)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise