import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import streamlit as st
import os
import joblib

def train_model(data, model_name):
    """Train a machine learning model with the given data."""
    # Check if data is valid
    if data.empty or 'is_anomaly' not in data.columns:
        st.error("Invalid data: Missing transactions or 'is_anomaly' column")
        return None, None, None

    # Define features in fixed order to match preprocessing and inference
    feature_cols = [
        'transaction_type', 'transaction_channel', 'payment_method', 'originating_country',
        'destination_country', 'counterparty_country', 'transaction_amount', 'transaction_frequency',
        'sanctioned_country', 'deviation_from_profile', 'unusual_timing', 'structuring',
        'rapid_movement', 'sanctions_list_hit', 'pep_match', 'negative_media',
        'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel'
    ]
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing feature columns: {missing_cols}")
        return None, None, None

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            st.error(f"Non-numeric values in column {col}: {data[col].dtype}")
            return None, None, None

    X = data[feature_cols]
    y = data['is_anomaly']

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error splitting data: {e}")
        return None, None, None

    # Store test indices for customer mapping
    st.session_state.test_indices = X_test.index.values

    # Initialize and train the model
    model = None
    try:
        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
        else:
            st.error(f"Unsupported model: {model_name}")
            return None, None, None

        model.fit(X_train, y_train)

        # Save model
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'models', model_name.lower().replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        joblib.dump(model, model_path)
        st.success(f"Model saved to {model_path}")

        return model, X_test, y_test
    except Exception as e:
        st.error(f"Error training {model_name}: {e}")
        return None, None, None

def evaluate_model(model, X_test, y_test, column, model_name):
    """Evaluate the model and display results in the Streamlit column."""
    if model is None or X_test is None or y_test is None:
        column.error("No model or test data available")
        return

    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Display classification report
        column.subheader("Model Evaluation")
        column.text("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

        # Display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ax.set_title('Confusion Matrix', fontsize=10)
        ax.tick_params(labelsize=7)
        column.pyplot(fig)
    except Exception as e:
        column.error(f"Error evaluating model: {e}")