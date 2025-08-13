import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RFMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', 
                 datetime_col='TransactionStartTime', snapshot_date=None):
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
        
        return rfm.reset_index()