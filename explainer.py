import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def explain_model(model, X_test, df, model_type, use_rule_based):
    if use_rule_based:
        st.success("Rule-Based Fraud Detection Insights")
        risky = df[df['is_anomaly'] == 1]
        for col in ['transaction_type', 'originating_country', 'device_id']:
            st.subheader(f"Top Risky {col.replace('_', ' ').title()}")
            top = risky[col].value_counts().head(5)
            st.bar_chart(top)
    else:
        if model_type == "RandomForest (Supervised)":
            importances = model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by="Importance", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(3, 1.5))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
            ax.set_xlabel("Importance score", fontsize=3)
            ax.set_title("Top 15 Features by RandomForest Importance", fontsize=4)
            ax.tick_params(axis='y', labelsize=3)
            ax.tick_params(axis='x', labelsize=3)
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.success("Anomaly Score Distribution")
            scores = model.score_samples(X_test)
            fig, ax = plt.subplots(figsize=(3, 1.5))
            ax.hist(scores, bins=30, color='teal')
            ax.set_xlabel("Anomaly Score", fontsize=3)
            ax.set_title(f"{model_type} Anomaly Score Distribution", fontsize=4)
            ax.tick_params(axis='both', labelsize=3)
            st.pyplot(fig)