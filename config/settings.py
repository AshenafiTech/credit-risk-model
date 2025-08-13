from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    # Model settings
    model_name: str = "CreditRiskBestModel"
    model_stage: str = "Production"
    
    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "credit-risk-model"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Data settings
    data_path: str = "data"
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Feature engineering
    customer_id_col: str = "CustomerId"
    amount_col: str = "Amount"
    datetime_col: str = "TransactionStartTime"
    
    class Config:
        env_file = ".env"

settings = Settings()