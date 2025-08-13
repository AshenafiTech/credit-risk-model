import pandas as pd
import numpy as np
import pytest
from src.data_processing import RFMTransformer, FeatureEngineer, build_preprocessing_pipeline
from src.proxy_target import create_risk_proxy, prepare_training_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'transaction_amount': [100, 200, 150, 50, 300, 75, 125, 200, 100],
        'transaction_datetime': [
            '2023-01-01', '2023-01-10', '2023-01-15',
            '2023-01-05', '2023-01-20', 
            '2023-01-08', '2023-01-12', '2023-01-18', '2023-01-25'
        ]
    })

def test_rfm_transformer(sample_data):
    transformer = RFMTransformer()
    result = transformer.fit_transform(sample_data)
    
    assert 'recency' in result.columns
    assert 'frequency' in result.columns
    assert 'monetary' in result.columns
    assert len(result) == 3  # 3 unique customers
    assert result['frequency'].sum() == 9  # Total transactions

def test_feature_engineer(sample_data):
    rfm_transformer = RFMTransformer()
    rfm_data = rfm_transformer.fit_transform(sample_data)
    
    engineer = FeatureEngineer()
    result = engineer.fit_transform(rfm_data)
    
    assert 'risk_score' in result.columns
    assert 'high_recency' in result.columns
    assert result['risk_score'].min() >= 0
    assert result['risk_score'].max() <= 3

def test_preprocessing_pipeline(sample_data):
    pipeline = build_preprocessing_pipeline()
    X = pipeline.fit_transform(sample_data)
    
    assert X.shape[0] == 3  # 3 customers
    assert X.shape[1] == 9  # 9 features
    assert not np.isnan(X).any()  # No missing values

def test_risk_proxy_clustering(sample_data):
    rfm_transformer = RFMTransformer()
    rfm_data = rfm_transformer.fit_transform(sample_data)
    
    result = create_risk_proxy(rfm_data, method='clustering')
    assert 'is_high_risk' in result.columns
    assert set(result['is_high_risk']).issubset({0, 1})

def test_risk_proxy_threshold(sample_data):
    rfm_transformer = RFMTransformer()
    rfm_data = rfm_transformer.fit_transform(sample_data)
    
    result = create_risk_proxy(rfm_data, method='threshold')
    assert 'is_high_risk' in result.columns
    assert set(result['is_high_risk']).issubset({0, 1})

def test_prepare_training_data(sample_data):
    X, y = prepare_training_data(sample_data)
    
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 3  # 3 customers
    assert len(y.unique()) <= 2  # Binary target