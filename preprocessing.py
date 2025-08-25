import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def preprocess_data(data, model_name):
    """Preprocess transaction data for training or prediction."""
    required_cols = [
        'customer_id', 'timestamp', 'transaction_amount', 'transaction_type',
        'transaction_frequency', 'transaction_channel', 'counterparty_name',
        'counterparty_country', 'payment_method', 'originating_country',
        'destination_country', 'sanctioned_country', 'deviation_from_profile',
        'unusual_timing', 'structuring', 'rapid_movement', 'sanctions_list_hit',
        'pep_match', 'negative_media', 'ip_address', 'device_type',
        'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel',
        'is_anomaly'
    ]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None

    if data.empty:
        print("Error: No data provided")
        return None

    categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method',
                       'originating_country', 'destination_country', 'counterparty_country']
    numerical_cols = ['transaction_amount', 'transaction_frequency', 'sanctioned_country',
                     'deviation_from_profile', 'unusual_timing', 'structuring', 'rapid_movement',
                     'sanctions_list_hit', 'pep_match', 'negative_media', 'account_age_days',
                     'last_update_days', 'failed_attempts', 'impossible_travel']
    df_processed = data.copy()

    try:
        # Encode categorical columns
        encoders = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'models', model_name.lower().replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
            encoder_path = os.path.join(model_dir, f'encoder_{col}.pkl')
            joblib.dump(le, encoder_path)
            print(f"Saved encoder to {encoder_path}")

        # Scale numerical columns
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")

        return df_processed
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None