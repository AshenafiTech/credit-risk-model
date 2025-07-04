import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


def train_and_evaluate(X, y, experiment_name='credit-risk-model', search_type='grid'):
    """
    Trains and evaluates multiple models with hyperparameter tuning and MLflow tracking.
    search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    """
    mlflow.set_experiment(experiment_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'DecisionTree': {'max_depth': [3, 5, 10, None]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
    }
    best_model = None
    best_score = 0
    best_run_id = None
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            if search_type == 'random':
                search = RandomizedSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_iter=5, random_state=42)
            else:
                search = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc')
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, name)
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = search.best_estimator_
                best_run_id = run.info.run_id
    # Register best model
    if best_model is not None:
        mlflow.sklearn.log_model(best_model, 'best_model')
        mlflow.register_model(f'runs:/{best_run_id}/best_model', 'CreditRiskBestModel')
    return best_model

# Example usage (replace with your processed data)
if __name__ == '__main__':
    # df = pd.read_csv('processed_data.csv')
    # X = df.drop(['is_high_risk', 'customer_id'], axis=1)
    # y = df['is_high_risk']
    # train_and_evaluate(X, y, search_type='grid')
    pass
