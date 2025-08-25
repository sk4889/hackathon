import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import streamlit as st
import os
import joblib

def preprocess_data(data, model_name):
    """Preprocess transaction data for training."""
    # Validate input data
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
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing columns: {[col for col in required_cols if col not in data.columns]}")
        return None

    if data.empty:
        st.error("No data provided")
        return None

    # Define feature columns
    categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method',
                       'originating_country', 'destination_country', 'counterparty_country']
    numerical_cols = ['transaction_amount', 'transaction_frequency', 'sanctioned_country',
                     'deviation_from_profile', 'unusual_timing', 'structuring', 'rapid_movement',
                     'sanctions_list_hit', 'pep_match', 'negative_media', 'account_age_days',
                     'last_update_days', 'failed_attempts', 'impossible_travel']
    df_processed = data.copy()

    try:
        # Encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            st.session_state.encoders[col] = le
            model_dir = f"models/{model_name.lower().replace(' ', '_')}"
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(le, f"{model_dir}/encoder_{col}.pkl")

        # Scale numerical columns
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        st.session_state.scaler = scaler
        model_dir = f"models/{model_name.lower().replace(' ', '_')}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

        # Apply SMOTE for class imbalance
        feature_cols = categorical_cols + numerical_cols
        X = df_processed[feature_cols]
        y = df_processed['is_anomaly']
        if len(X) >= 5 and y.nunique() >= 2:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
            df_resampled['is_anomaly'] = y_resampled
            return df_resampled
        else:
            st.warning("Not enough data or classes for SMOTE")
            return df_processed

    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None