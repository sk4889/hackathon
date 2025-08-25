import streamlit as st
import pandas as pd
import numpy as np

def predict_realtime(column):
    """Handle real-time anomaly detection in Streamlit."""
    column.write(f"Debug: model_choice = {st.session_state.get('model_choice', 'None')}")
    if 'model' not in st.session_state or st.session_state.model is None:
        column.error("Please train a model first")
        return
    if 'encoders' not in st.session_state or 'scaler' not in st.session_state:
        column.error("Please preprocess data first")
        return
    if 'model_choice' not in st.session_state or st.session_state.model_choice is None:
        column.error("Please select a model in the sidebar and complete the ML flow (Generate Data → Preprocess Data → Train Model)")
        return

    column.subheader("Real-Time Anomaly Detection")
    with column.form("realtime_form"):
        customer_id = st.text_input("Customer ID", value="123e4567-e89b-12d3-a456-426614174000")
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
        transaction_type = st.selectbox("Transaction Type", ['cash_deposit', 'wire_transfer', 'remittance'])
        transaction_frequency = st.number_input("Transaction Frequency", min_value=0, value=1)
        transaction_channel = st.selectbox("Transaction Channel", ['branch', 'online', 'ATM'])
        counterparty_name = st.text_input("Counterparty Name", value="Acme Corp")
        counterparty_country = st.selectbox("Counterparty Country", ['US', 'IN', 'UK', 'CA', 'RU', 'CN', 'NG'])
        payment_method = st.selectbox("Payment Method", ['cash', 'digital'])
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
        account_age_days = st.number_input("Account Age (Days)", min_value=1, value=365)
        last_update_days = st.number_input("Last Update (Days)", min_value=1, value=10)
        failed_attempts = st.number_input("Failed Attempts", min_value=0, value=0)
        impossible_travel = st.checkbox("Impossible Travel")
        submit = st.form_submit_button("Predict")

        if submit:
            try:
                # Create transaction DataFrame
                transaction = {
                    'customer_id': customer_id, 'timestamp': pd.Timestamp.now(),
                    'transaction_amount': transaction_amount, 'transaction_type': transaction_type,
                    'transaction_frequency': transaction_frequency, 'transaction_channel': transaction_channel,
                    'counterparty_name': counterparty_name, 'counterparty_country': counterparty_country,
                    'payment_method': payment_method, 'originating_country': originating_country,
                    'destination_country': destination_country, 'sanctioned_country': sanctioned_country,
                    'deviation_from_profile': deviation_from_profile, 'unusual_timing': unusual_timing,
                    'structuring': structuring, 'rapid_movement': rapid_movement,
                    'sanctions_list_hit': sanctions_list_hit, 'pep_match': pep_match,
                    'negative_media': negative_media, 'ip_address': ip_address,
                    'device_type': device_type, 'account_age_days': account_age_days,
                    'last_update_days': last_update_days, 'failed_attempts': failed_attempts,
                    'impossible_travel': impossible_travel
                }
                df = pd.DataFrame([transaction])

                # Define feature columns in fixed order
                feature_cols = [
                    'transaction_type', 'transaction_channel', 'payment_method', 'originating_country',
                    'destination_country', 'counterparty_country', 'transaction_amount', 'transaction_frequency',
                    'sanctioned_country', 'deviation_from_profile', 'unusual_timing', 'structuring',
                    'rapid_movement', 'sanctions_list_hit', 'pep_match', 'negative_media',
                    'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel'
                ]

                # Preprocess transaction
                categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method',
                                   'originating_country', 'destination_country', 'counterparty_country']
                numerical_cols = ['transaction_amount', 'transaction_frequency', 'sanctioned_country',
                                 'deviation_from_profile', 'unusual_timing', 'structuring', 'rapid_movement',
                                 'sanctions_list_hit', 'pep_match', 'negative_media', 'account_age_days',
                                 'last_update_days', 'failed_attempts', 'impossible_travel']

                for col in categorical_cols:
                    if col in st.session_state.encoders:
                        try:
                            df[col] = st.session_state.encoders[col].transform(df[col].astype(str))
                        except ValueError:
                            column.error(f"Invalid value for {col}. Must match training data categories.")
                            return
                    else:
                        column.error(f"Encoder for {col} not found")
                        return

                df[numerical_cols] = st.session_state.scaler.transform(df[numerical_cols])

                # Predict
                model = st.session_state.model
                X = df[feature_cols]
                pred = model.predict(X)[0]
                column.write(f"Prediction: {'Anomaly' if pred == 1 else 'Normal'}")
            except Exception as e:
                column.error(f"Error predicting: {e}")