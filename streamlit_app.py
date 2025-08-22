import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from utils.preprocessing import preprocess_df
from utils.model_ml import train_ml_model, evaluate_ml_model
from utils.explainer_ml import explain_ml_model
from utils.predict_realtime_ml import predict_realtime_ml
from utils.rule_based_analytics import customer_centric_view, transactional_view
from math import radians, sin, cos, sqrt, atan2
import os
import joblib

st.set_page_config(page_title="Payment Anomaly Detection Demo", layout="wide")
st.title("Payment Anomaly Detection Demo")

# Initialize Faker
Faker.seed(42)
fake = Faker()

# Country coordinates (lat, long)
country_coords = {
    'US': (37.09024, -95.712891),
    'IN': (20.593684, 78.96288),
    'UK': (55.378051, -3.435973),
    'CA': (56.130366, -106.346771),
    'RU': (61.52401, 105.318756),
    'CN': (35.86166, 104.195397),
    'NG': (9.081999, 8.675277)
}

# Predefined lists for restricted fields
device_types = ['desktop', 'laptop', 'tablet', 'phone']
company_names = [
    'Acme Corp', 'Globex Inc', 'Soylent Solutions', 'Initech', 'Umbrella Corp',
    'Cyberdyne Systems', 'Wayne Enterprises', 'Stark Industries', 'LexCorp', 'Omni Consumer Products'
]
occupations = [
    'Software Engineer', 'Accountant', 'Marketing Manager', 'Sales Representative',
    'Teacher', 'Doctor', 'Consultant', 'Analyst'
]

def haversine(coord1, coord2):
    R = 6371.0  # Earth radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Generate custom IP address in 192.10.x.y format
def generate_custom_ip():
    return f"192.10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"

# Common data generation with timestamps and impossible travel simulation
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    num_customers = max(1, num_records // 10)

    customer_profiles = pd.DataFrame({
        'customer_id': [fake.uuid4() for _ in range(num_customers)],
        'customer_type': np.random.choice(['individual', 'corporate', 'NGO', 'PEP'], num_customers, p=[0.6, 0.2, 0.1, 0.1]),
        'risk_rating': np.random.choice(['low', 'medium', 'high'], num_customers, p=[0.7, 0.2, 0.1]),
        'nationality': np.random.choice(['US', 'IN', 'UK', 'CA', 'RU'], num_customers, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'occupation': np.random.choice(occupations, num_customers),
        'ip_address': [generate_custom_ip() for _ in range(num_customers)],
        'location': np.random.choice(list(country_coords.keys()), num_customers),
        'usual_transaction_time_hour': np.random.randint(0, 24, num_customers),
        'usual_amount_mean': np.random.lognormal(mean=5, sigma=1, size=num_customers).round(2),
        'usual_frequency_per_day': np.random.randint(1, 5, num_customers),
        'device_type': np.random.choice(device_types, num_customers),
        'account_age_days': np.random.randint(1, 1000, num_customers),
        'last_update_days': np.random.randint(1, 365, num_customers)
    })

    transactions = []
    customer_tx = customer_profiles.groupby('customer_id').groups
    for customer_id in customer_tx:
        customer = customer_profiles.loc[customer_tx[customer_id][0]]
        num_tx = np.random.randint(1, num_records // num_customers + 1)
        timestamps = pd.date_range(start='2025-01-01', periods=num_tx, freq='H')  # Hourly for demo
        prev_loc = customer['location']
        for i in range(num_tx):
            is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])
            amount = np.random.normal(customer['usual_amount_mean'], 100) if not is_fraud else np.random.uniform(1000, 5000)
            loc = customer['location'] if not is_fraud else np.random.choice(list(country_coords.keys()))
            impossible = 0
            if i > 0:
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                dist = haversine(country_coords[prev_loc], country_coords[loc])
                if dist / time_diff > 1000:  # Impossible speed
                    impossible = 1
            prev_loc = loc
            transaction = {
                'customer_id': customer_id,
                'timestamp': timestamps[i],
                'transaction_amount': round(amount, 2),
                'transaction_type': np.random.choice(['cash_deposit', 'wire_transfer', 'remittance', 'crypto', 'trade_finance']),
                'transaction_frequency': customer['usual_frequency_per_day'],
                'transaction_channel': np.random.choice(['branch', 'online', 'ATM', 'mobile', 'SWIFT']),
                'counterparty_name': np.random.choice(company_names),
                'counterparty_country': np.random.choice(list(country_coords.keys())),
                'payment_method': np.random.choice(['cash', 'cheque', 'digital', 'prepaid_card']),
                'originating_country': loc,
                'destination_country': np.random.choice(list(country_coords.keys())),
                'sanctioned_country': 1 if is_fraud else np.random.choice([0, 1], p=[0.95, 0.05]),
                'deviation_from_profile': 1 if is_fraud else 0,
                'unusual_timing': 1 if is_fraud and np.random.rand() > 0.5 else 0,
                'structuring': np.random.choice([0, 1], p=[0.95, 0.05]),
                'rapid_movement': np.random.choice([0, 1], p=[0.9, 0.1]),
                'sanctions_list_hit': np.random.choice([0, 1], p=[0.98, 0.02]),
                'pep_match': np.random.choice([0, 1], p=[0.95, 0.05]),
                'negative_media': np.random.choice([0, 1], p=[0.97, 0.03]),
                'ip_address': customer['ip_address'] if not is_fraud else generate_custom_ip(),
                'device_type': customer['device_type'] if not is_fraud else np.random.choice(device_types),
                'account_age_days': customer['account_age_days'],
                'last_update_days': customer['last_update_days'],
                'failed_attempts': np.random.randint(0, 5),
                'impossible_travel': impossible,
                'is_anomaly': is_fraud
            }
            transactions.append(transaction)

    transaction_df = pd.DataFrame(transactions)
    return customer_profiles, transaction_df

# Session state
if 'customer_profiles' not in st.session_state:
    st.session_state.customer_profiles = None
if 'transaction_df' not in st.session_state:
    st.session_state.transaction_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = None

# UI layout
left_col, right_col = st.columns([1, 4])

with left_col:
    st.subheader("Options")
    flow_choice = st.radio("Choose Flow", ["ML Based", "Rule Based"])
    row_limit = st.number_input("Number of Records", min_value=100, max_value=10000, value=1000, step=100)
    if st.button("Generate Data"):
        try:
            customer_profiles, transaction_df = generate_synthetic_data(row_limit)
            st.session_state.customer_profiles = customer_profiles
            st.session_state.transaction_df = transaction_df
            st.session_state.processed_df = None
            st.session_state.model = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.session_state.model_choice = None
            st.success("Data Generated")
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")

    if st.button("Save Profiles"):
        if st.session_state.customer_profiles is not None:
            try:
                st.session_state.customer_profiles.to_csv('customer_profiles.csv', index=False)
                with open('customer_profiles.csv', 'rb') as f:
                    st.download_button("Download Profiles CSV", f, file_name='customer_profiles.csv')
            except Exception as e:
                st.error(f"Error saving profiles: {str(e)}")
        else:
            st.error("Generate Data First")

    if flow_choice == "ML Based":
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Isolation Forest", "OneClassSVM", "XGBoost"])
        # Show estimators slider only for tree-based models
        if model_choice in ["Random Forest", "Isolation Forest", "XGBoost"]:
            n_estimators = st.slider("Estimators (for tree models)", 50, 300, 100, 10)
        else:
            n_estimators = None
        if st.button("Preprocess"):
            if st.session_state.transaction_df is not None:
                try:
                    processed_df = preprocess_df(st.session_state.transaction_df)
                    st.session_state.processed_df = processed_df
                    st.success("Preprocessed")
                except Exception as e:
                    st.error(f"Error preprocessing: {str(e)}")
            else:
                st.error("Generate Data First")
        if st.button("Train"):
            if st.session_state.processed_df is not None:
                try:
                    model, X_test, y_test = train_ml_model(st.session_state.processed_df, model_choice, n_estimators)
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.model_choice = model_choice
                    st.success("Trained")
                except Exception as e:
                    st.error(f"Error training: {str(e)}")
            else:
                st.error("Preprocess First")
        if st.button("Evaluate"):
            if st.session_state.model is not None:
                try:
                    evaluate_ml_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, right_col, model_choice)
                except Exception as e:
                    st.error(f"Error evaluating: {str(e)}")
            else:
                st.error("Train Model First")
        if st.button("Insights"):
            if st.session_state.model is not None:
                try:
                    explain_ml_model(st.session_state.model, st.session_state.X_test, right_col, model_choice)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
            else:
                st.error("Train Model First")
        show_realtime = st.checkbox("Real-Time Detection")

    elif flow_choice == "Rule Based":
        tab_choice = st.radio("View", ["Customer-Centric Fraudulent Transactions", "Transactional View"])
        if st.button("Analyze"):
            if st.session_state.customer_profiles is not None and st.session_state.transaction_df is not None:
                try:
                    if tab_choice == "Customer-Centric Fraudulent Transactions":
                        customer_centric_view(st.session_state.customer_profiles, st.session_state.transaction_df, right_col)
                    else:
                        transactional_view(st.session_state.customer_profiles, st.session_state.transaction_df, right_col)
                except Exception as e:
                    st.error(f"Error analyzing: {str(e)}")
            else:
                st.error("Generate Data First")

with right_col:
    st.subheader("Results and Insights")
    if 'transaction_df' in st.session_state and st.session_state.transaction_df is not None:
        try:
            st.dataframe(st.session_state.transaction_df.head(10))
        except Exception as e:
            st.error(f"Error displaying transaction data: {str(e)}")
    if flow_choice == "ML Based" and show_realtime and st.session_state.model is not None:
        try:
            predict_realtime_ml()
        except Exception as e:
            st.error(f"Error in real-time detection: {str(e)}")