import streamlit as st
import pandas as pd
#from common.data_generation import generate_data, save_data
#from streamlit_app.preprocessing import preprocess_data
#from streamlit_app.model_training import train_model, evaluate_model
#from streamlit_app.inference import predict_realtime
#from streamlit_app.rule_based import show_customer_centric, show_transactional
#from streamlit_app.insights import show_insights

from data_generation import generate_data, save_data
from preprocessing import preprocess_data
from model_training import train_model, evaluate_model
from inference import predict_realtime
from rule_based import show_customer_centric, show_transactional
from insights import show_insights

# Set page configuration
st.set_page_config(page_title="Payment Anomaly Detection", layout="wide")
st.title("Payment Anomaly Detection")

# Initialize session state
if 'customer_profiles' not in st.session_state:
    st.session_state.customer_profiles = None
if 'transaction_df' not in st.session_state:
    st.session_state.transaction_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "XGBoost"  # Default to XGBoost

# Sidebar for user inputs
st.sidebar.header("Settings")
flow_choice = st.sidebar.radio("Choose Analysis Type", ["ML Based", "Rule Based"])
num_records = st.sidebar.slider("Number of Transactions", 100, 1000, 500)
if flow_choice == "ML Based":
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "XGBoost"], index=2, key="model_select")  # Default to XGBoost
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice
        st.session_state.model = None  # Reset model if model_choice changes
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.session_state.encoders = {}
        st.session_state.scaler = None
else:
    st.session_state.model_choice = None  # Clear for Rule Based

# Create two columns: left (20%) for controls, right (80%) for results
left_col, right_col = st.columns([0.2, 0.8])

with left_col:
    st.header("Controls")
    if flow_choice == "ML Based":
        if st.button("Generate Data"):
            try:
                profiles, transactions = generate_data(num_records)
                st.session_state.customer_profiles = profiles
                st.session_state.transaction_df = transactions
                save_data(profiles, transactions)
                st.success(f"Generated {num_records} transactions")
            except Exception as e:
                st.error(f"Error generating data: {e}")

        if st.button("Preprocess Data"):
            if st.session_state.transaction_df is not None and st.session_state.model_choice:
                try:
                    st.session_state.transaction_df = preprocess_data(st.session_state.transaction_df, st.session_state.model_choice)
                    st.success("Data preprocessed successfully")
                except Exception as e:
                    st.error(f"Error preprocessing data: {e}")
            else:
                st.error("Generate data and select a model first")

        if st.button("Train Model"):
            if st.session_state.transaction_df is not None and st.session_state.model_choice:
                try:
                    model, X_test, y_test = train_model(st.session_state.transaction_df, st.session_state.model_choice)
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.success(f"Trained {st.session_state.model_choice} model")
                except Exception as e:
                    st.error(f"Error training model: {e}")
            else:
                st.error("Preprocess data and select a model first")

        if st.button("Evaluate Model"):
            if st.session_state.model is not None and st.session_state.model_choice:
                try:
                    evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, right_col, st.session_state.model_choice)
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")
            else:
                st.error("Train model first")

        if st.button("Show Insights"):
            if st.session_state.model is not None and st.session_state.model_choice:
                try:
                    show_insights(st.session_state.model, st.session_state.X_test, right_col, st.session_state.model_choice)
                except Exception as e:
                    st.error(f"Error showing insights: {e}")
            else:
                st.error("Train model first")

        show_realtime = st.checkbox("Real-Time Detection")

    elif flow_choice == "Rule Based":
        if st.button("Generate Data"):
            try:
                profiles, transactions = generate_data(num_records)
                st.session_state.customer_profiles = profiles
                st.session_state.transaction_df = transactions
                save_data(profiles, transactions)
                st.success(f"Generated {num_records} transactions")
            except Exception as e:
                st.error(f"Error generating data: {e}")

        tab_choice = st.radio("View", ["Customer-Centric", "Transactional"])
        if st.button("Analyze"):
            if st.session_state.customer_profiles is not None and st.session_state.transaction_df is not None:
                try:
                    if tab_choice == "Customer-Centric":
                        show_customer_centric(st.session_state.customer_profiles, st.session_state.transaction_df, right_col)
                    else:
                        show_transactional(st.session_state.customer_profiles, st.session_state.transaction_df, right_col)
                except Exception as e:
                    st.error(f"Error analyzing: {e}")
            else:
                st.error("Generate data first")

with right_col:
    st.header("Results and Insights")
    if st.session_state.transaction_df is not None:
        try:
            st.dataframe(st.session_state.transaction_df.head(10))
        except Exception as e:
            st.error(f"Error displaying data: {e}")
    if flow_choice == "ML Based" and show_realtime:
        try:
            st.write(f"Debug: model_choice = {st.session_state.get('model_choice', 'None')}")
            if st.session_state.model_choice:
                predict_realtime(right_col)
            else:
                st.error("Select a model in the sidebar to enable real-time detection")
        except Exception as e:
            st.error(f"Error in real-time detection: {e}")