import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class RFMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='customer_id', amount_col='transaction_amount', 
                 datetime_col='transaction_datetime', snapshot_date=None):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        if self.snapshot_date is None:
            self.snapshot_date = X[self.datetime_col].max() + pd.Timedelta(days=1)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        rfm = X.groupby(self.customer_id_col).agg({
            self.datetime_col: lambda x: (self.snapshot_date - x.max()).days,
            self.amount_col: ['count', 'sum', 'mean', 'std']
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary', 'avg_amount', 'std_amount']
        rfm['std_amount'] = rfm['std_amount'].fillna(0)
        
        # Add behavioral features
        rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        return rfm.reset_index()

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Risk indicators
        X['high_recency'] = (X['recency'] > X['recency'].quantile(0.75)).astype(int)
        X['low_frequency'] = (X['frequency'] < X['frequency'].quantile(0.25)).astype(int)
        X['low_monetary'] = (X['monetary'] < X['monetary'].quantile(0.25)).astype(int)
        
        # Composite scores
        X['rfm_score'] = X['recency_score'].astype(str) + X['frequency_score'].astype(str) + X['monetary_score'].astype(str)
        X['risk_score'] = X['high_recency'] + X['low_frequency'] + X['low_monetary']
        
        return X

def build_preprocessing_pipeline(customer_id_col='customer_id', amount_col='transaction_amount', 
                               datetime_col='transaction_datetime'):
    
    numerical_features = ['recency', 'frequency', 'monetary', 'avg_amount', 'std_amount', 
                         'high_recency', 'low_frequency', 'low_monetary', 'risk_score']
    
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features)
    ], remainder='drop')
    
    pipeline = Pipeline([
        ('rfm', RFMTransformer(customer_id_col, amount_col, datetime_col)),
        ('features', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

def get_feature_names():
    return ['recency', 'frequency', 'monetary', 'avg_amount', 'std_amount', 
            'high_recency', 'low_frequency', 'low_monetary', 'risk_score']