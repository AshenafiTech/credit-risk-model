#!/usr/bin/env python3
"""Training script for credit risk model"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import load_raw_data
from src.data.target_creation import prepare_training_data
from src.models.train import train_and_evaluate
from src.utils.logging import setup_logging
from config.settings import settings

def main():
    logger = setup_logging(log_file="logs/training.log")
    
    try:
        # Load data
        logger.info("Loading raw data...")
        df = load_raw_data()
        
        # Prepare training data
        logger.info("Preparing training data...")
        X, y = prepare_training_data(df)
        
        # Train model
        logger.info("Training model...")
        best_model = train_and_evaluate(X, y, experiment_name=settings.mlflow_experiment_name)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()