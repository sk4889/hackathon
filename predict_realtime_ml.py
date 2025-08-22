import streamlit as st
import pandas as pd
import numpy as np
import re

def predict_realtime_ml():
    if 'model' not in st.session_state or st.session_state.model is None:
        st.error("Please train a model first.")
        return
    
    if 'encoders' not in st.session_state or 'scaler' not in st.session_state:
        st.error("Encoders or scaler not found. Please preprocess data first.")
        return
    
    st.subheader("Real-Time Anomaly Detection")
    
    # Input fields for a single transaction
    with st.form("realtime_form"):
        customer_id = st.text_input("Customer ID", value="123e4567-e89b-12d3-a456-426614174000")
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        transaction_type = st.selectbox("Transaction Type", ['cash_deposit', 'wire_transfer', 'remittance', 'crypto', 'trade_finance'])
        transaction_frequency = st.number_input("Transaction Frequency", min_value=0, value=1)
        transaction_channel = st.selectbox("Transaction Channel", ['branch', 'online', 'ATM', 'mobile', 'SWIFT'])
        counterparty_name = st.text_input("Counterparty Name", value="Acme Corp")
        counterparty_country = st.selectbox("Counterparty Country", ['US', 'IN', 'UK', 'CA', 'RU', 'CN', 'NG'])
        payment_method = st.selectbox("Payment Method", ['cash', 'cheque', 'digital', 'prepaid_card'])
        originating_country = st.selectbox("Originating Country", ['US', 'IN', 'UK', 'CA', 'RU', 'CN', 'NG'])
        destination_country = st.selectbox("Destination Country", ['US', 'IN', 'UK', 'CA', 'RU', 'CN', 'NG'])
        sanctioned_country = st.checkbox("Sanctioned Country")
        deviation_from_profile = st.checkbox("Deviation from Profile")
        unusual_timing = st.checkbox("Unusual Timing")
        structuring = st.checkbox("Structuring")
        rapid_movement = st.checkbox("Rapid Movement")
        sanctions_list_hit = st.checkbox("Sanctions List Hit")
        pep_match = st.checkbox("PEP Match")
        negative_media = st.checkbox("Negative Media")
        ip_address = st.text_input("IP Address", value="192.10.123.45")
        device_type = st.selectbox("Device Type", ['desktop', 'laptop', 'tablet', 'phone'])
        account_age_days = st.number_input("Account Age (Days)", min_value=0, value=500)
        last_update_days = st.number_input("Last Update (Days)", min_value=0, value=30)
        failed_attempts = st.number_input("Failed Attempts", min_value=0, value=0)
        impossible_travel = st.checkbox("Impossible Travel")
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Validate inputs
        if not re.match(r'^192\.10\.\d{1,3}\.\d{1,3}$', ip_address):
            st.error("Invalid IP address format. Must be 192.10.x.y")
            return
        
        # Create transaction dictionary
        transaction = {
            'customer_id': customer_id,
            'timestamp': pd.Timestamp.now(),
            'transaction_amount': transaction_amount,
            'transaction_type': transaction_type,
            'transaction_frequency': transaction_frequency,
            'transaction_channel': transaction_channel,
            'counterparty_name': counterparty_name,
            'counterparty_country': counterparty_country,
            'payment_method': payment_method,
            'originating_country': originating_country,
            'destination_country': destination_country,
            'sanctioned_country': int(sanctioned_country),
            'deviation_from_profile': int(deviation_from_profile),
            'unusual_timing': int(unusual_timing),
            'structuring': int(structuring),
            'rapid_movement': int(rapid_movement),
            'sanctions_list_hit': int(sanctions_list_hit),
            'pep_match': int(pep_match),
            'negative_media': int(negative_media),
            'ip_address': ip_address,
            'device_type': device_type,
            'account_age_days': account_age_days,
            'last_update_days': last_update_days,
            'failed_attempts': failed_attempts,
            'impossible_travel': int(impossible_travel)
        }
        
        # Preprocess transaction
        df = pd.DataFrame([transaction])
        categorical_cols = [
            'transaction_type', 'transaction_channel', 'payment_method',
            'originating_country', 'destination_country', 'counterparty_country'
        ]
        numerical_cols = [
            'transaction_amount', 'transaction_frequency', 'sanctioned_country',
            'deviation_from_profile', 'unusual_timing', 'structuring', 'rapid_movement',
            'sanctions_list_hit', 'pep_match', 'negative_media', 'account_age_days',
            'last_update_days', 'failed_attempts', 'impossible_travel'
        ]
        
        # Apply label encoding
        try:
            for col in categorical_cols:
                if col in st.session_state.encoders:
                    try:
                        df[col] = st.session_state.encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        st.error(f"Invalid value for {col}. Must match training data categories.")
                        return
                else:
                    st.error(f"Encoder for {col} not found.")
                    return
        except Exception as e:
            st.error(f"Error encoding categorical variables: {str(e)}")
            return
        
        # Apply scaling
        try:
            df[numerical_cols] = st.session_state.scaler.transform(df[numerical_cols])
        except Exception as e:
            st.error(f"Error scaling numerical features: {str(e)}")
            return
        
        # Predict
        model = st.session_state.model
        model_choice = st.session_state.get('model_choice', 'Unknown')
        feature_cols = categorical_cols + numerical_cols
        X = df[feature_cols]
        
        try:
            if model_choice in ["Isolation Forest", "OneClassSVM"]:
                pred = model.predict(X)
                pred = np.where(pred == -1, 1, 0)[0]
            else:
                pred = model.predict(X)[0]
            st.write(f"Prediction: {'Anomaly' if pred == 1 else 'Normal'}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")