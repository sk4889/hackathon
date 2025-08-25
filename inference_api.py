import pandas as pd
import numpy as np
import joblib
import os

def predict_realtime(data, model_name):
    """Predict anomalies for input data."""
    try:
        # Load model
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(project_root, 'models', model_name.lower().replace(' ', '_'), 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)

        # Load encoders and scaler
        categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method',
                           'originating_country', 'destination_country', 'counterparty_country']
        numerical_cols = ['transaction_amount', 'transaction_frequency', 'sanctioned_country',
                         'deviation_from_profile', 'unusual_timing', 'structuring', 'rapid_movement',
                         'sanctions_list_hit', 'pep_match', 'negative_media', 'account_age_days',
                         'last_update_days', 'failed_attempts', 'impossible_travel']
        feature_cols = categorical_cols + numerical_cols
        model_dir = os.path.join(project_root, 'models', model_name.lower().replace(' ', '_'))
        encoders = {}
        for col in categorical_cols:
            encoder_path = os.path.join(model_dir, f'encoder_{col}.pkl')
            if os.path.exists(encoder_path):
                encoders[col] = joblib.load(encoder_path)
            else:
                raise FileNotFoundError(f"Encoder for {col} not found")
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError("Scaler not found")

        # Preprocess data
        df = data.copy()
        for col in categorical_cols:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    raise ValueError(f"Invalid value for {col}. Must match training data categories.")
            else:
                raise ValueError(f"Column {col} missing in input data")
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Predict
        X = df[feature_cols]
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        raise Exception(f"Prediction error: {e}")
