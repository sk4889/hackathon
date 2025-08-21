import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def apply_rule_based_prediction(input_df):
    rules = [
        (input_df['transaction_amount'] > 1000, "Unusual Amount"),
        (input_df['sanctioned_country'] == 1, "Sanctioned Country"),
        (input_df['deviation_from_profile'] == 1, "Deviation from Profile"),
        (input_df['unusual_timing'] == 1, "Unusual Timing"),
        (input_df['structuring'] == 1, "Structuring"),
        (input_df['rapid_movement'] == 1, "Rapid Movement"),
        (input_df['sanctions_list_hit'] == 1, "Sanctions List Hit"),
        (input_df['pep_match'] == 1, "PEP Match"),
        (input_df['negative_media'] == 1, "Negative Media"),
        (input_df['account_age_days'] < 30, "Dormant Account"),
        (input_df['last_update_days'] < 7, "Recently Updated Account"),
        (input_df['failed_attempts'] > 2, "Multiple Failed Attempts")
    ]
    is_anomaly = 0
    reasons = []
    for condition, reason in rules:
        if condition.iloc[0]:
            is_anomaly = 1
            reasons.append(reason)
    return is_anomaly, "; ".join(reasons) if reasons else "Legit Transaction"

def predict_realtime(use_rule_based):
    if 'df' not in st.session_state or 'processed_df' not in st.session_state or 'model' not in st.session_state:
        st.warning("Please complete: 1. Upload/Generate data 2. Preprocess 3. Train model")
        return
    
    df = st.session_state.df
    model = st.session_state.model
    st.subheader("Enter Transaction Details")
    with st.form("realtime_form", clear_on_submit=False):
        user_input = {}
        feature_columns = [col for col in df.columns if col not in ['customer_id', 'is_anomaly']]
        for col in feature_columns:
            if df[col].dtype == 'object':
                options = sorted(df[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(col, options)
            else:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                default_val = int(df[col].median())
                user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val)
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_df = pd.DataFrame([user_input])
        st.session_state['submitted_data'] = input_df
        if use_rule_based:
            pred, reason = apply_rule_based_prediction(input_df)
            prob = 1.0 if pred else 0.0
        else:
            cat_cols = input_df.select_dtypes(include='object').columns.tolist()
            enc = st.session_state.encoder
            ohe = pd.DataFrame(enc.transform(input_df[cat_cols]), columns=enc.get_feature_names_out(cat_cols), index=input_df.index)
            input_encoded = pd.concat([input_df.drop(columns=cat_cols), ohe], axis=1)
            input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            pred = model.predict(input_encoded)[0]
            prob = model.predict_proba(input_encoded)[0][1] if hasattr(model, 'predict_proba') else -model.score_samples(input_encoded)[0]
            prob = (prob - prob.min()) / (prob.max() - prob.min()) if not hasattr(model, 'predict_proba') else prob
            reason = "ML Prediction"
        
        show_risk_bar(prob)
        st.success(f"Prediction: {'Suspicious Transaction' if pred else 'Legit Transaction'}")
        if use_rule_based and reason:
            st.info(f"Reason: {reason}")

def show_risk_bar(prob):
    level = "LOW" if prob < 0.4 else "MEDIUM" if prob < 0.75 else "HIGH"
    color = "#006A4E" if prob < 0.4 else "#FFBF00" if prob < 0.75 else "#D2042D"
    st.markdown(f"<div style='font-size: 20px;'>Risk Score: <span style='color:{color}; font-weight:bold;'>{prob * 100:.1f}% ({level})</span></div>", unsafe_allow_html=True)
    filled_width = int(prob * 100)
    empty_width = 100 - filled_width
    st.markdown(f"""
    <div style='display: flex; height: 25px; border: 1px solid #ccc; border-radius: 5px; overflow: hidden;'>
    <div style='width: {filled_width}%; background-color: {color};'></div>
    <div style='width: {empty_width}%; background-color: #f0f0f0;'></div>
    </div>""", unsafe_allow_html=True)