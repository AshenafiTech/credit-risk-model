# Credit Scoring Model - Bati Bank Buy-Now-Pay-Later Service

## Project Overview

Bati Bank, a leading financial service provider with over 10 years of experience, is partnering with a successful eCommerce company to enable a buy-now-pay-later service. This project develops a Credit Scoring Model that transforms customer behavioral data into predictive risk signals using Recency, Frequency, and Monetary (RFM) patterns.

## Business Context

### Basel II Accord and Regulatory Compliance

The Basel II Accord emphasizes transparent and consistent credit risk measurement. Our models must be interpretable, auditable, and well-documented to support regulatory compliance, customer trust, and operational risk mitigation.

### RFM-Based Risk Proxy

Since direct default indicators are often unavailable, we engineer a proxy for credit risk using customer behavioral patterns. This approach transforms eCommerce transaction data into meaningful risk signals while acknowledging the inherent risks of proxy-based modeling.

### Model Interpretability vs Performance

We balance interpretable models (Logistic Regression with WoE encoding) against complex models (Gradient Boosting) to meet both regulatory requirements and predictive accuracy needs.

## Technical Architecture

### Core Components
- **Data Processing**: RFM feature engineering and proxy target creation
- **Model Training**: Multi-algorithm comparison with hyperparameter tuning
- **API Service**: FastAPI-based prediction endpoint
- **MLOps**: MLflow for experiment tracking and model registry
- **CI/CD**: Automated testing and deployment pipeline

### Technology Stack
- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **API**: FastAPI + Uvicorn
- **Testing**: pytest
- **Linting**: flake8
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Project Structure

```
credit-risk-model/
├── src/
│   ├── api/                    # FastAPI service
│   ├── data_processing.py      # RFM feature engineering
│   ├── proxy_target.py         # Risk proxy creation
│   ├── train_model.py          # Model training & tuning
│   └── predict.py              # Prediction utilities
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory data analysis
├── tests/                      # Unit tests
├── .github/workflows/          # CI/CD pipelines
└── docker-compose.yml          # Container orchestration
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training Models
```python
from src.train_model import train_and_evaluate

# Load your processed data
X, y = load_processed_data()
best_model = train_and_evaluate(X, y, experiment_name='credit-risk-v1')
```

### Running API Service
```bash
# Using Docker
docker-compose up

# Or directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Making Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [feature_values]}'
```

## Model Development Workflow

1. **Data Exploration**: Analyze customer transaction patterns
2. **Feature Engineering**: Create RFM features and risk proxies
3. **Model Training**: Compare algorithms with hyperparameter tuning
4. **Model Selection**: Choose best performing model via cross-validation
5. **Model Registry**: Register production model in MLflow
6. **API Deployment**: Serve model via FastAPI endpoint
7. **Monitoring**: Track model performance and drift

## Key Features

- **Advanced scikit-learn Usage**: Custom pipelines and transformers
- **Feature Engineering**: RFM analysis and behavioral pattern extraction
- **Model Comparison**: Automated evaluation of multiple algorithms
- **Hyperparameter Tuning**: Grid and randomized search optimization
- **MLOps Integration**: MLflow experiment tracking and model registry
- **Production API**: FastAPI service with Pydantic validation
- **CI/CD Pipeline**: Automated testing and deployment
- **Unit Testing**: Comprehensive test coverage
- **Code Quality**: Automated linting and formatting

## Model Performance Metrics

- **ROC-AUC**: Primary metric for model selection
- **Precision/Recall**: Balance false positives vs false negatives
- **F1-Score**: Harmonic mean of precision and recall
- **Business Metrics**: Expected loss, approval rates, profit optimization

## Deployment

The model is deployed as a containerized FastAPI service with:
- Health checks and monitoring
- Model versioning via MLflow
- Horizontal scaling capability
- Production-ready logging

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Lint code
flake8 src/ --max-line-length=120
```

## Contributing

1. Follow PEP 8 style guidelines
2. Write unit tests for new features
3. Update documentation for API changes
4. Ensure CI/CD pipeline passes

## Risk Considerations

- **Proxy Risk**: Model relies on behavioral proxies, not direct default data
- **Data Drift**: Monitor for changes in customer behavior patterns
- **Regulatory Compliance**: Ensure model interpretability for audits
- **Bias Detection**: Regular fairness assessments across customer segments

## License

Proprietary - Bati Bank Internal Use Only