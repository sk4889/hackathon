import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def show_insights(models, X_test, right_col, model_choice):
    """
    Display model insights including feature importance and SHAP values in the right column.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test data for SHAP and feature importance.
        right_col: Streamlit column for rendering outputs.
        model_choice (str): Selected model ("Random Forest", "Logistic Regression", "XGBoost", "All").
    """
    with right_col:
        st.subheader(f"Insights for {model_choice}")

        # Select model
        if model_choice == "All":
            model = models.get("XGBoost")  # Default to XGBoost for "All"
        else:
            model = models.get(model_choice)

        if model is None:
            st.error(f"No {model_choice} model available.")
            return

        # Check if X_test is available
        if X_test is None:
            st.warning("No test data available for SHAP plots. Generating sample data for visualization.")
            # Generate sample data (minimal example to avoid blank plots)
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'transaction_amount': np.random.uniform(100, 10000, 100),
                'transaction_frequency': np.random.randint(1, 10, 100),
                'deviation_from_profile': np.random.uniform(0, 1, 100),
                'unusual_timing': np.random.choice([0, 1], 100),
                'structuring': np.random.choice([0, 1], 100),
                'rapid_movement': np.random.choice([0, 1], 100),
                'sanctions_list_hit': np.random.choice([0, 1], 100),
                'pep_match': np.random.choice([0, 1], 100),
                'negative_media': np.random.choice([0, 1], 100),
                'failed_attempts': np.random.randint(0, 5, 100),
                'impossible_travel': np.random.choice([0, 1], 100),
                'account_age_days': np.random.randint(30, 1000, 100),
                'last_update_days': np.random.randint(1, 100, 100)
            })
            # Add categorical columns (minimal encoding)
            categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method', 
                              'originating_country', 'destination_country', 'counterparty_country']
            for col in categorical_cols:
                sample_data[col] = np.random.choice(['type1', 'type2', 'type3'], 100)
            X_test = sample_data

        try:
            # Feature Importance
            st.write("Feature Importance")
            if model_choice == "Logistic Regression":
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': abs(model.coef_[0])
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(feature_importance.set_index('Feature')['Importance'])
            else:
                # For tree-based models (Random Forest, XGBoost)
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(feature_importance.set_index('Feature')['Importance'])

            # SHAP Values Plot
            st.write("SHAP Values Summary Plot")
            if model_choice == "Random Forest":
                explainer = shap.TreeExplainer(model)
            elif model_choice == "XGBoost":
                explainer = shap.TreeExplainer(model)
            elif model_choice == "All":
                explainer = shap.TreeExplainer(model)  # Assuming best model is XGBoost
            else:
                explainer = shap.LinearExplainer(model, X_test)

            # Compute SHAP values
            shap_values = explainer.shap_values(X_test)
            
            # Handle multi-class or binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 for binary classification
            
            # Create SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(plt)
            plt.clf()

        except Exception as e:
            st.error(f"Error generating insights for {model_choice}: {e}")