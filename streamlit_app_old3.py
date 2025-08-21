import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_df
from utils.model import train_model, evaluate_model, save_model_artifacts
from utils.explainer import explain_model
from utils.predict_realtime import predict_realtime
from faker import Faker
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Transaction Anomaly Detector", layout="wide")
st.title("Real-Time Financial Transaction Anomaly Detection")
st.success("Steps: Generate/Load Data > Preview Data > Preprocess Data (for ML) > Train Model (for ML) > Evaluate Model (for ML) > Explain Model > Detect Real-Time")

# Initialize Faker for synthetic data
Faker.seed(42)
fake = Faker()

# Synthetic data generation
def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    data = {
        'customer_id': [fake.uuid4() for _ in range(num_records)],
        'customer_type': np.random.choice(['individual', 'corporate', 'NGO', 'PEP'], num_records, p=[0.6, 0.2, 0.1, 0.1]),
        'risk_rating': np.random.choice(['low', 'medium', 'high'], num_records, p=[0.7, 0.2, 0.1]),
        'nationality': np.random.choice(['US', 'IN', 'UK', 'CA', 'RU'], num_records, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'occupation': [fake.job() for _ in range(num_records)],
        'transaction_amount': np.random.lognormal(mean=5, sigma=1, size=num_records).round(2),
        'transaction_type': np.random.choice(['cash_deposit', 'wire_transfer', 'remittance', 'crypto', 'trade_finance'], num_records),
        'transaction_frequency': np.random.randint(1, 10, num_records),
        'transaction_channel': np.random.choice(['branch', 'online', 'ATM', 'mobile', 'SWIFT'], num_records),
        'counterparty_name': [fake.company() for _ in range(num_records)],
        'counterparty_country': np.random.choice(['US', 'IN', 'UK', 'CN', 'NG'], num_records),
        'payment_method': np.random.choice(['cash', 'cheque', 'digital', 'prepaid_card'], num_records),
        'originating_country': np.random.choice(['US', 'IN', 'UK', 'CA', 'RU'], num_records),
        'destination_country': np.random.choice(['US', 'IN', 'UK', 'CN', 'NG'], num_records),
        'sanctioned_country': np.random.choice([0, 1], num_records, p=[0.95, 0.05]),
        'deviation_from_profile': np.random.choice([0, 1], num_records, p=[0.9, 0.1]),
        'unusual_timing': np.random.choice([0, 1], num_records, p=[0.85, 0.15]),
        'structuring': np.random.choice([0, 1], num_records, p=[0.95, 0.05]),
        'rapid_movement': np.random.choice([0, 1], num_records, p=[0.9, 0.1]),
        'sanctions_list_hit': np.random.choice([0, 1], num_records, p=[0.98, 0.02]),
        'pep_match': np.random.choice([0, 1], num_records, p=[0.95, 0.05]),
        'negative_media': np.random.choice([0, 1], num_records, p=[0.97, 0.03]),
        'ip_address': [fake.ipv4() for _ in range(num_records)],
        'device_id': [fake.uuid4() for _ in range(num_records)],
        'account_age_days': np.random.randint(1, 1000, num_records),
        'last_update_days': np.random.randint(1, 365, num_records),
        'failed_attempts': np.random.randint(0, 5, num_records),
        'is_anomaly': np.zeros(num_records, dtype=int)
    }
    df = pd.DataFrame(data)
    fraud_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
    df.loc[fraud_indices, 'is_anomaly'] = 1
    df.loc[fraud_indices, 'transaction_amount'] = np.random.uniform(1000, 5000, len(fraud_indices))
    df.loc[fraud_indices, 'sanctioned_country'] = 1
    df.loc[fraud_indices, 'deviation_from_profile'] = 1
    return df

# Data source selection
data_source = st.radio("Select Data Source", ["Upload CSV", "Generate Synthetic Data"])
row_limit = st.number_input("Limit to N rows (0 = full dataset)", min_value=0, max_value=10000, value=1000, step=50)
n_estimators = st.slider("Number of trees (for RandomForest)", min_value=50, max_value=300, value=100, step=10)
model_type = st.selectbox("Model Type", ["RandomForest (Supervised)", "OneClassSVM (Semi-Supervised)", "IsolationForest (Unsupervised)"])
use_rule_based = st.checkbox("Use Rule-Based Detection", value=True)

# Initialize session variables
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Load or generate data
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a transaction CSV file", type=["csv"])
    if uploaded_file is not None and st.button("Preview of uploaded data"):
        df = pd.read_csv(uploaded_file)
        if row_limit > 0:
            df = df.head(row_limit)
        st.session_state.df = df
        st.session_state.processed_df = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.subheader("Preview of uploaded data")
        st.info(f"Dataset before preprocessing contains total {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())
else:
    if st.button("Generate Synthetic Data"):
        df = generate_synthetic_data(num_records=row_limit if row_limit > 0 else 1000)
        st.session_state.df = df
        st.session_state.processed_df = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.subheader("Preview of synthetic data")
        st.info(f"Synthetic dataset contains total {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())

# Preprocessing (for ML)
if st.button("Run Preprocessing"):
    if st.session_state.df is not None:
        df = st.session_state.df
        processed_df = preprocess_df(df, skip_smote=use_rule_based)  # Skip SMOTE for rule-based
        st.session_state.processed_df = processed_df
        st.session_state.feature_columns = list(processed_df.columns)
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.subheader("Processed Data Preview")
        st.info(f"Dataset after preprocessing contains total {processed_df.shape[0]} rows and {processed_df.shape[1]} columns.")
        st.dataframe(processed_df.head())
    else:
        st.error("Please load or generate data first")

# Train model
if st.button("Train"):
    try:
        if st.session_state.processed_df is not None:
            processed_df = st.session_state.processed_df
            model, X_test, y_test = train_model(processed_df, n_estimators, model_type)
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        else:
            st.error("Please preprocess data first")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Evaluate model
if st.button("Evaluate"):
    try:
        if st.session_state.model is not None and st.session_state.X_test is not None and st.session_state.y_test is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            df = st.session_state.df
            preds_df, preds = evaluate_model(model, X_test, y_test, model_type, df, use_rule_based)
            if preds_df is not None:
                st.session_state.preds_df = preds_df
                st.subheader("Predictions and Risk Score")
                st.dataframe(preds_df.head(10))
            else:
                st.error("Evaluation failed due to sample mismatch")
        else:
            st.error("Please train the model first")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Save model
if st.button("Save Trained Model for API use"):
    try:
        msg = save_model_artifacts()
        st.success(msg)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Explain model
if st.button("Detected Patterns"):
    try:
        if st.session_state.model is not None and st.session_state.X_test is not None:
            st.success("Fraud patterns by feature importances or anomaly scores")
            explain_model(st.session_state.model, st.session_state.X_test, st.session_state.df, model_type, use_rule_based)
        else:
            st.error("Please train the model first")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Real-time prediction
show_form = st.checkbox("Show user form for real-time Fraud detection")
if show_form:
    st.subheader("Real-time Fraud Detection")
    try:
        predict_realtime(use_rule_based)
    except Exception as e:
        st.warning("Please complete: 1. Upload/Generate data 2. Preprocess (for ML) 3. Train (for ML) 4. Evaluate (for ML)")
