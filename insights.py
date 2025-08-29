import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

def show_insights(models, X_test, column, model_choice):
    if model_choice == "All":
        column.subheader("Model Comparison Insights")
        for name, model in models.items():
            column.subheader(f"Feature Importance: {name}")
            if name == "XGBoost":
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values(by='importance', ascending=False)
                column.bar_chart(feature_importance.set_index('feature'))
            elif name == "Random Forest":
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': model.feature_importances_
                }).sort_values(by='importance', ascending=False)
                column.bar_chart(feature_importance.set_index('feature'))
            elif name == "Logistic Regression":
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': abs(model.coef_[0])
                }).sort_values(by='importance', ascending=False)
                column.bar_chart(feature_importance.set_index('feature'))

            column.subheader(f"SHAP Values: {name}")
            try:
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, show=False)
                column.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                column.error(f"Error generating SHAP plot for {name}: {e}")
    else:
        model = models[model_choice]
        column.subheader(f"Feature Importance: {model_choice}")
        if model_choice in ["Random Forest", "XGBoost"]:
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            column.bar_chart(feature_importance.set_index('feature'))
        elif model_choice == "Logistic Regression":
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': abs(model.coef_[0])
            }).sort_values(by='importance', ascending=False)
            column.bar_chart(feature_importance.set_index('feature'))

        column.subheader(f"SHAP Values: {model_choice}")
        try:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, show=False)
            column.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            column.error(f"Error generating SHAP plot: {e}")