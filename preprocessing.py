import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import streamlit as st

def preprocess_df(df):
    # Validate input DataFrame
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
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Missing required columns: {missing_cols}")
        return None
    
    if df.empty:
        st.error("Input DataFrame is empty")
        return None
    
    # Initialize session state
    if 'encoders' not in st.session_state:
        st.session_state.encoders = {}
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'customer_ids' not in st.session_state:
        st.session_state.customer_ids = None
    
    # Define columns
    exclude_cols = ['customer_id', 'timestamp', 'counterparty_name', 'ip_address', 'device_type']
    categorical_cols = [
        'transaction_type', 'transaction_channel', 'payment_method',
        'originating_country', 'destination_country', 'counterparty_country'
    ]
    numerical_cols = [
        col for col in df.columns
        if col not in exclude_cols + categorical_cols + ['is_anomaly']
    ]
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Store customer_ids in session state
    st.session_state.customer_ids = df_processed['customer_id'].values
    
    try:
        # Label encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            st.session_state.encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        st.session_state.scaler = scaler
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        feature_cols = categorical_cols + numerical_cols
        X = df_processed[feature_cols]
        y = df_processed['is_anomaly']
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Reconstruct DataFrame
        df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
        df_resampled['is_anomaly'] = y_resampled
        
        return df_resampled
    
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None