import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

def show_insights(model, X_test, column, model_name):
    """Show model insights like feature importance."""
    if model is None or X_test.empty:
        column.error("No model or test data available")
        return

    try:
        # Limit to 100 rows for performance
        X_test_features = X_test.iloc[:100]

        # Feature importance
        if model_name in ["Random Forest", "XGBoost"]:
            imp = pd.Series(model.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            sns.barplot(x=imp.values[:10], y=imp.index[:10], ax=ax)
            ax.set_title('Feature Importance', fontsize=10)
            ax.set_xlabel('Importance', fontsize=8)
            ax.set_ylabel('Feature', fontsize=8)
            ax.tick_params(labelsize=7)
            column.pyplot(fig)
        elif model_name == "Logistic Regression":
            imp = pd.Series(np.abs(model.coef_[0]), index=X_test_features.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            sns.barplot(x=imp.values[:10], y=imp.index[:10], ax=ax)
            ax.set_title('Coefficient Magnitude', fontsize=10)
            ax.set_xlabel('Magnitude', fontsize=8)
            ax.set_ylabel('Feature', fontsize=8)
            ax.tick_params(labelsize=7)
            column.pyplot(fig)

        # SHAP analysis
        column.subheader("SHAP Feature Analysis")
        explainer = shap.TreeExplainer(model) if model_name in ["Random Forest", "XGBoost"] else shap.LinearExplainer(model, X_test_features)
        shap_values = explainer.shap_values(X_test_features)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        shap.summary_plot(shap_values, X_test_features, plot_type="bar", max_display=10, show=False)
        column.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        column.error(f"Error showing insights: {e}")