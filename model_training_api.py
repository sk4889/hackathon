import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import joblib

def train_model(data, model_name):
    """Train a machine learning model with the given data."""
    # Validate input
    if data.empty:
        print("Error: Input data is empty")
        return None, None, None

    if 'is_anomaly' not in data.columns:
        print("Error: 'is_anomaly' column is missing")
        return None, None, None

    if data['is_anomaly'].nunique() < 2:
        print(f"Error: Model {model_name} requires at least two classes in 'is_anomaly'")
        return None, None, None

    if data.isna().any().any():
        print(f"Error: Data contains missing values in columns: {data.columns[data.isna().any()].tolist()}")
        return None, None, None

    # Define features in fixed order to match preprocessing and inference
    feature_cols = [
        'transaction_type', 'transaction_channel', 'payment_method', 'originating_country',
        'destination_country', 'counterparty_country', 'transaction_amount', 'transaction_frequency',
        'sanctioned_country', 'deviation_from_profile', 'unusual_timing', 'structuring',
        'rapid_movement', 'sanctions_list_hit', 'pep_match', 'negative_media',
        'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel'
    ]
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing feature columns: {missing_cols}")
        return None, None, None

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"Error: Non-numeric values in column {col}: {data[col].dtype}")
            return None, None, None

    X = data[feature_cols]
    y = data['is_anomaly']

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None

    # Initialize and train model
    model = None
    try:
        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
        else:
            print(f"Error: Unsupported model: {model_name}")
            return None, None, None

        model.fit(X_train, y_train)

        # Save model with absolute path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'models', model_name.lower().replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model, X_test, y_test
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None, None, None
