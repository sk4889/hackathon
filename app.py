import streamlit as st
import pandas as pd
import os
import joblib
from data_generation import generate_data, save_data
from preprocessing import preprocess_data
from model_training import train_model, evaluate_model
from inference import predict_realtime
from rule_based import show_customer_centric, show_transactional
from insights import show_insights
from data_dictionary import display_data_dictionary

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
if 'models' not in st.session_state:
    st.session_state.models = {}  # Store multiple models for "All"
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
if 'best_model' not in st.session_state:
    st.session_state.best_model = None  # Store best model for "All"

# Function to load existing models, encoders, scalers, and test data
def load_existing_model(model_name):
    model_dir = f"models/{model_name.lower().replace(' ', '_')}"
    model_path = f"{model_dir}/best_model.pkl" if model_name == "All" else f"{model_dir}/model.pkl"
    scaler_path = f"{model_dir}/scaler.pkl"
    encoder_cols = ['transaction_type', 'transaction_channel', 'payment_method', 
                    'originating_country', 'destination_country', 'counterparty_country']
    encoder_paths = [f"{model_dir}/encoder_{col}.pkl" for col in encoder_cols]
    test_data_path = f"{model_dir}/X_test.pkl"
    
    try:
        # Check if all required files exist
        if not os.path.exists(model_path):
            st.warning(f"No pre-trained model found at {model_path}")
            return False
        for encoder_path in encoder_paths:
            if not os.path.exists(encoder_path):
                st.warning(f"Missing encoder at {encoder_path}")
                return False
        if not os.path.exists(scaler_path):
            st.warning(f"Missing scaler at {scaler_path}")
            return False
        
        # Load model
        model = joblib.load(model_path)
        if model_name == "All":
            st.session_state.best_model = model
            st.session_state.models[model_name] = model
            st.session_state.model = model  # Default for single-model operations
        else:
            st.session_state.model = model
            st.session_state.models[model_name] = model
        
        # Load scaler
        st.session_state.scaler = joblib.load(scaler_path)
        
        # Load encoders
        for encoder_path, col in zip(encoder_paths, encoder_cols):
            st.session_state.encoders[col] = joblib.load(encoder_path)
        
        # Load test data if available
        if os.path.exists(test_data_path):
            st.session_state.X_test = joblib.load(test_data_path)
            st.info(f"Loaded preprocessed test data from {test_data_path}")
        else:
            st.warning(f"No preprocessed test data found at {test_data_path}. Attempting to preprocess data if available.")
        
        st.success(f"Successfully loaded pre-trained {model_name} model and preprocessing artifacts")
        return True
    except Exception as e:
        st.error(f"Error loading {model_name} model or artifacts: {e}")
        return False

# Sidebar for user inputs with tooltips
st.sidebar.header("Settings")
flow_choice = st.sidebar.radio(
    "Choose Analysis Type",
    ["ML Based", "Rule Based"],
    key="flow_choice",
    help="Choose between machine learning-based or rule-based anomaly detection"
)
num_records = st.sidebar.slider(
    "Number of Transactions",
    100, 1000, 500,
    key="num_records",
    help="Select the number of synthetic transactions to generate"
)
if flow_choice == "ML Based":
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "XGBoost", "All"],
        index=3,
        key="model_select",
        help="Choose a machine learning model or compare all models"
    )
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice
        st.session_state.model = None
        st.session_state.models = {}
        st.session_state.encoders = {}
        st.session_state.scaler = None
        st.session_state.best_model = None
        load_existing_model(model_choice)  # Load pre-trained model on selection
else:
    st.session_state.model_choice = None
    st.session_state.best_model = None

# Create two columns: left (20%) for controls, right (80%) for results
left_col, right_col = st.columns([0.2, 0.8])

with left_col:
    st.header("Controls")
    
    # Display Data Dictionary
    display_data_dictionary()

    if flow_choice == "ML Based":
        st.subheader("ML Based Analysis")
        if st.button(
            "Generate Data",
            key="generate_data",
            help="Generate synthetic customer and transaction data"
        ):
            try:
                profiles, transactions = generate_data(num_records)
                st.session_state.customer_profiles = profiles
                st.session_state.transaction_df = transactions
                save_data(profiles, transactions)
                st.success(f"Generated {num_records} transactions")
            except Exception as e:
                st.error(f"Error generating data: {e}")

        if st.button(
            "Preprocess Data",
            key="preprocess_data",
            help="Prepare data for model training"
        ):
            if st.session_state.transaction_df is not None and st.session_state.model_choice:
                try:
                    st.session_state.transaction_df = preprocess_data(st.session_state.transaction_df, st.session_state.model_choice)
                    if st.session_state.transaction_df is None:
                        st.error("Preprocessing failed, please check data")
                    else:
                        st.success("Data preprocessed successfully")
                except Exception as e:
                    st.error(f"Error preprocessing data: {e}")
            else:
                st.error("Generate data and select a model first")

        if st.button(
            "Train Model",
            key="train_model",
            help="Train the selected machine learning model"
        ):
            if st.session_state.transaction_df is not None and st.session_state.model_choice:
                try:
                    if st.session_state.model_choice == "All":
                        models, X_test, y_test, best_model = train_model(st.session_state.transaction_df, st.session_state.model_choice)
                        st.session_state.models = models
                        st.session_state.model = models.get("XGBoost")  # Default to XGBoost for single-model operations
                        st.session_state.best_model = best_model
                        # Save X_test for future use
                        model_dir = f"models/{st.session_state.model_choice.lower().replace(' ', '_')}"
                        os.makedirs(model_dir, exist_ok=True)
                        joblib.dump(X_test, f"{model_dir}/X_test.pkl")
                    else:
                        model, X_test, y_test = train_model(st.session_state.transaction_df, st.session_state.model_choice)
                        st.session_state.model = model
                        st.session_state.models = {st.session_state.model_choice: model}
                        st.session_state.best_model = model
                        # Save X_test for future use
                        model_dir = f"models/{st.session_state.model_choice.lower().replace(' ', '_')}"
                        os.makedirs(model_dir, exist_ok=True)
                        joblib.dump(X_test, f"{model_dir}/X_test.pkl")
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.success(f"Trained {st.session_state.model_choice} model(s)")
                except Exception as e:
                    st.error(f"Error training model: {e}")
            else:
                st.error("Preprocess data and select a model first")

        if st.button(
            "Evaluate Model",
            key="evaluate_model",
            help="Evaluate the trained model's performance"
        ):
            if st.session_state.models and st.session_state.model_choice:
                try:
                    evaluate_model(st.session_state.models, st.session_state.X_test, st.session_state.y_test, right_col, st.session_state.model_choice)
                except Exception as e:
                    st.error(f"Error evaluating model: {e}")
            else:
                st.error("Train model first or ensure pre-trained models exist in the 'models' folder.")

        if st.button(
            "Show Insights",
            key="show_insights",
            help="Display feature importance and SHAP plots"
        ):
            if not st.session_state.models:
                load_existing_model(st.session_state.model_choice)
            if st.session_state.models and st.session_state.model_choice:
                try:
                    if st.session_state.X_test is None and st.session_state.transaction_df is not None:
                        st.info("No test data found. Attempting to preprocess existing data.")
                        st.session_state.transaction_df = preprocess_data(st.session_state.transaction_df, st.session_state.model_choice)
                        # Assume preprocess_data splits data and updates X_test
                        if st.session_state.X_test is None:
                            st.warning("Preprocessing did not generate test data. Insights may be limited.")
                    if st.session_state.X_test is None:
                        st.warning("No test data available for insights. Some features may be limited. Generate and preprocess data for full insights.")
                    show_insights(st.session_state.models, st.session_state.X_test, right_col, st.session_state.model_choice)
                except Exception as e:
                    st.error(f"Error showing insights: {e}")
            else:
                st.error("No trained model available. Train a model or ensure pre-trained models exist in the 'models' folder.")

        show_realtime = st.checkbox(
            "Real-Time Detection",
            key="realtime_detection",
            help="Enable real-time anomaly detection"
        )

    elif flow_choice == "Rule Based":
        if st.button(
            "Generate Data",
            key="rule_generate_data",
            help="Generate synthetic customer and transaction data"
        ):
            try:
                profiles, transactions = generate_data(num_records)
                st.session_state.customer_profiles = profiles
                st.session_state.transaction_df = transactions
                save_data(profiles, transactions)
                st.success(f"Generated {num_records} transactions")
            except Exception as e:
                st.error(f"Error generating data: {e}")

        tab_choice = st.radio(
            "View",
            ["Customer-Centric", "Transactional"],
            key="rule_tab_choice",
            help="Choose between customer-centric or transactional analysis"
        )
        if st.button(
            "Analyze",
            key="rule_analyze",
            help="Analyze data based on selected view"
        ):
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
                if not st.session_state.models:
                    load_existing_model(st.session_state.model_choice)
                if st.session_state.models:
                    predict_realtime(right_col)
                else:
                    st.error("No trained model available. Train a model or ensure pre-trained models exist in the 'models' folder.")
            else:
                st.error("Select a model in the sidebar to enable real-time detection")
        except Exception as e:
            st.error(f"Error in real-time detection: {e}")