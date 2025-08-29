import streamlit as st
import pandas as pd

def display_data_dictionary():
    with st.expander("Data Dictionary", expanded=False):
        # Create tabs for Customer Profiles and Transactions
        customer_tab, transaction_tab = st.tabs(["Customer Profiles", "Transactions"])
        
        # Customer Profiles
        with customer_tab:
            st.subheader("Customer Profiles")
            customer_dict = pd.DataFrame([
                {"Field Name": "customer_id", "Data Type": "String", "Description": "Unique customer ID"},
                {"Field Name": "name", "Data Type": "String", "Description": "Customer’s name"},
                {"Field Name": "account_type", "Data Type": "String", "Description": "Account type (e.g., individual)"},
                {"Field Name": "risk_score", "Data Type": "Float", "Description": "Customer risk score"},
                {"Field Name": "account_age_days", "Data Type": "Integer", "Description": "Days since account creation"},
                {"Field Name": "country", "Data Type": "String", "Description": "Customer’s residence country"},
            ])
            search_customer = st.text_input("Search Customer Profiles Fields", key="search_customer", placeholder="e.g., customer_id")
            if search_customer:
                customer_dict = customer_dict[customer_dict.apply(lambda row: search_customer.lower() in row.to_string().lower(), axis=1)]
            styled_customer = customer_dict.style.set_properties(**{
                'text-align': 'left',
                'font-weight': ['bold' if col == 'Field Name' else 'normal' for col in customer_dict.columns]
            })
            st.table(styled_customer)
        
        # Transactions
        with transaction_tab:
            st.subheader("Transactions")
            transaction_dict = pd.DataFrame([
                {"Field Name": "transaction_id", "Data Type": "String", "Description": "Unique transaction ID"},
                {"Field Name": "customer_id", "Data Type": "String", "Description": "Customer ID for transaction"},
                {"Field Name": "transaction_amount", "Data Type": "Float", "Description": "Transaction amount"},
                {"Field Name": "transaction_type", "Data Type": "String", "Description": "Type (e.g., deposit)"},
                {"Field Name": "transaction_channel", "Data Type": "String", "Description": "Channel (e.g., online)"},
                {"Field Name": "payment_method", "Data Type": "String", "Description": "Method (e.g., card)"},
                {"Field Name": "originating_country", "Data Type": "String", "Description": "Transaction origin country"},
                {"Field Name": "destination_country", "Data Type": "String", "Description": "Transaction destination"},
                {"Field Name": "counterparty_country", "Data Type": "String", "Description": "Counterparty country"},
                {"Field Name": "transaction_frequency", "Data Type": "Integer", "Description": "Transactions in time window"},
                {"Field Name": "deviation_from_profile", "Data Type": "Float", "Description": "Behavior deviation"},
                {"Field Name": "unusual_timing", "Data Type": "Boolean", "Description": "Unusual timing flag"},
                {"Field Name": "structuring", "Data Type": "Boolean", "Description": "Structuring activity flag"},
                {"Field Name": "rapid_movement", "Data Type": "Boolean", "Description": "Rapid fund movement flag"},
                {"Field Name": "sanctions_list_hit", "Data Type": "Boolean", "Description": "Sanctions list hit flag"},
                {"Field Name": "pep_match", "Data Type": "Boolean", "Description": "PEP match flag"},
                {"Field Name": "negative_media", "Data Type": "Boolean", "Description": "Negative media flag"},
                {"Field Name": "failed_attempts", "Data Type": "Integer", "Description": "Failed attempt count"},
                {"Field Name": "impossible_travel", "Data Type": "Boolean", "Description": "Impossible travel flag"},
                {"Field Name": "last_update_days", "Data Type": "Integer", "Description": "Days since profile update"},
            ])
            search_transaction = st.text_input("Search Transactions Fields", key="search_transaction", placeholder="e.g., transaction_amount")
            if search_transaction:
                transaction_dict = transaction_dict[transaction_dict.apply(lambda row: search_transaction.lower() in row.to_string().lower(), axis=1)]
            styled_transaction = transaction_dict.style.set_properties(**{
                'text-align': 'left',
                'font-weight': ['bold' if col == 'Field Name' else 'normal' for col in transaction_dict.columns]
            })
            st.table(styled_transaction)