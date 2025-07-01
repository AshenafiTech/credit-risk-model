import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Example custom transformer for aggregation
class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='customer_id', amount_col='transaction_amount', datetime_col='transaction_datetime'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.customer_id_col).agg(
            total_transaction_amount=(self.amount_col, 'sum'),
            avg_transaction_amount=(self.amount_col, 'mean'),
            transaction_count=(self.amount_col, 'count'),
            std_transaction_amount=(self.amount_col, 'std')
        ).reset_index()
        return agg_df

# Example custom transformer for extracting datetime features
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='transaction_datetime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['transaction_hour'] = pd.to_datetime(X[self.datetime_col]).dt.hour
        X['transaction_day'] = pd.to_datetime(X[self.datetime_col]).dt.day
        X['transaction_month'] = pd.to_datetime(X[self.datetime_col]).dt.month
        X['transaction_year'] = pd.to_datetime(X[self.datetime_col]).dt.year
        return X

# Main function to build the pipeline
def build_feature_engineering_pipeline(categorical_cols, numerical_cols, customer_id_col='customer_id', amount_col='transaction_amount', datetime_col='transaction_datetime'):
    # Imputation
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Encoding
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Scaling
    scaler = StandardScaler()

    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', num_imputer),
            ('scaler', scaler)
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', cat_imputer),
            ('onehot', onehot)
        ]), categorical_cols)
    ])

    # Full pipeline
    pipeline = Pipeline([
        ('datetime_features', DateTimeFeatureExtractor(datetime_col=datetime_col)),
        ('aggregator', TransactionAggregator(customer_id_col=customer_id_col, amount_col=amount_col, datetime_col=datetime_col)),
        ('preprocessor', preprocessor)
    ])
    return pipeline

# Example usage (to be replaced with actual data loading in production)
if __name__ == '__main__':
    # df = pd.read_csv('path_to_raw_data.csv')
    # Define your categorical and numerical columns
    categorical_cols = ['category_col1', 'category_col2']  # replace with actual column names
    numerical_cols = ['total_transaction_amount', 'avg_transaction_amount', 'transaction_count', 'std_transaction_amount']
    pipeline = build_feature_engineering_pipeline(categorical_cols, numerical_cols)
    # X_processed = pipeline.fit_transform(df)
    pass
