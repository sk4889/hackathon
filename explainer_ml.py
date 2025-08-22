import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

def explain_ml_model(model, X_test, col, model_choice):
    if model is None:
        col.error("Model is not initialized. Please train the model first.")
        return
    
    if X_test.empty:
        col.error("Test data is empty")
        return
    
    feature_cols = [col for col in X_test.columns]
    # Limit to 100 rows for performance
    X_test_features = X_test[feature_cols].astype(float).iloc[:100]
    
    try:
        # Standard feature importance or anomaly scores
        if model_choice in ["Random Forest", "Logistic Regression", "XGBoost"]:
            if model_choice == "Random Forest":
                imp = pd.Series(model.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
                title = "Feature Importance"
            elif model_choice == "Logistic Regression":
                imp = pd.Series(np.abs(model.coef_[0]), index=X_test_features.columns).sort_values(ascending=False)
                title = "Coefficient Magnitude"
            elif model_choice == "XGBoost":
                imp = pd.Series(model.feature_importances_, index=X_test_features.columns).sort_values(ascending=False)
                title = "Feature Importance"
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=imp.values[:10], y=imp.index[:10], ax=ax)
            ax.set_title(title)
            col.pyplot(fig)
        else:
            if model_choice == "Isolation Forest":
                scores = -model.score_samples(X_test_features)
            else:  # OneClassSVM
                scores = model.decision_function(X_test_features)
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(scores, bins=20, kde=True, ax=ax)
            ax.set_title("Anomaly Score Distribution")
            col.pyplot(fig)
        
        # SHAP explanations for supervised models
        if model_choice in ["Random Forest", "Logistic Regression", "XGBoost"]:
            col.subheader("SHAP Feature Contribution Analysis")
            try:
                explainer = shap.TreeExplainer(model) if model_choice in ["Random Forest", "XGBoost"] else shap.LinearExplainer(model, X_test_features)
                shap_values = explainer.shap_values(X_test_features)
                
                shap_values_array = np.array(shap_values)
                col.write(f"SHAP values shape: {shap_values_array.shape}")
                if shap_values_array.size == 0 or np.all(shap_values_array == 0):
                    col.warning("SHAP values are empty or all zeros. Displaying raw values instead.")
                    col.write(pd.DataFrame(shap_values_array, columns=X_test_features.columns).head())
                    return
                
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, X_test_features, plot_type="bar", max_display=10, show=False)
                col.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                col.error(f"SHAP analysis failed: {str(e)}")
    
    except Exception as e:
        col.error(f"Error generating insights: {str(e)}")