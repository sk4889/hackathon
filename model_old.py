import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

def apply_rule_based_detection(df):
    rules = [
        (df['transaction_amount'] > 1000, "Unusual Amount"),
        (df['sanctioned_country'] == 1, "Sanctioned Country"),
        (df['deviation_from_profile'] == 1, "Deviation from Profile"),
        (df['unusual_timing'] == 1, "Unusual Timing"),
        (df['structuring'] == 1, "Structuring"),
        (df['rapid_movement'] == 1, "Rapid Movement"),
        (df['sanctions_list_hit'] == 1, "Sanctions List Hit"),
        (df['pep_match'] == 1, "PEP Match"),
        (df['negative_media'] == 1, "Negative Media"),
        (df['account_age_days'] < 30, "Dormant Account"),
        (df['last_update_days'] < 7, "Recently Updated Account"),
        (df['failed_attempts'] > 2, "Multiple Failed Attempts")
    ]
    df['is_anomaly'] = 0
    df['rule_triggered'] = ""
    for condition, reason in rules:
        df.loc[condition, 'is_anomaly'] = 1
        df.loc[condition, 'rule_triggered'] += reason + "; "
    return df

def train_model(df, n_estimators, model_type):
    with st.spinner("Model training in progress... Please wait"):
        X = df.drop(columns=['is_anomaly'])
        y = df['is_anomaly']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        if model_type == "RandomForest (Supervised)":
            model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=42)
            model.fit(X_train, y_train)
        elif model_type == "OneClassSVM (Semi-Supervised)":
            model = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
            model.fit(X_train[y_train == 0])  # Train on normal data
        else:  # IsolationForest
            model = IsolationForest(n_estimators=int(n_estimators), contamination=0.05, random_state=42)
            model.fit(X_train)
    
    st.success(f"{model_type} training completed, proceed with evaluation")
    return model, X_test, y_test

def plot_conf_matrix_from_values(tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['Legit', 'Fraud']
    cmap = sns.color_palette(["#FFBF00", "#006A4E"])
    fig, ax = plt.subplots(figsize=(2, 1.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar=False, linewidths=0.5, linecolor="gray", annot_kws={"size": 3})
    ax.set_ylabel("Predicted", fontsize=3)
    ax.set_xlabel("Actual", fontsize=3)
    ax.set_title("Confusion Matrix", fontsize=3)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=3)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=3)
    plt.tight_layout()
    st.pyplot(fig)

def evaluate_model(model, X_test, y_test, model_type, df, use_rule_based):
    if use_rule_based:
        result_df = apply_rule_based_detection(df.copy())
        preds = result_df['is_anomaly']
        result_df = result_df[['is_anomaly', 'rule_triggered'] + [col for col in result_df.columns if col not in ['is_anomaly', 'rule_triggered']]]
    else:
        if model_type == "RandomForest (Supervised)":
            preds = model.predict(X_test)
            risk_scores = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)
            preds = np.where(preds == -1, 1, 0)  # Convert -1 (anomaly) to 1, 1 (normal) to 0
            risk_scores = -model.score_samples(X_test) if model_type == "IsolationForest (Unsupervised)" else model.decision_function(X_test)
            risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())  # Normalize
        result_df = X_test.copy()
        result_df['predicted_anomaly'] = preds
        result_df['risk_score'] = risk_scores
        last_two_cols = result_df.columns[-2:].tolist()
        other_cols = result_df.columns[:-2].tolist()
        new_column_order = last_two_cols + other_cols
        result_df = result_df[new_column_order]
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    total = tn + fp + fn + tp
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (tp + fn) if (tp + fn) else 0
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    
    st.success("Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**FPR (False Positive Rate):** {fpr:.2f}")
    st.write(f"**FNR (False Negative Rate):** {fnr:.2f}")
    st.success("Model Performance Overview")
    plot_conf_matrix_from_values(tn, fp, fn, tp)
    
    to_csv_download(result_df, filename='data.csv')
    return result_df, preds

def to_csv_download(result_df, filename="data.csv"):
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Result", data=csv, file_name=filename, mime="text/csv")

def save_model_artifacts():
    if "model" not in st.session_state or "encoder" not in st.session_state or "feature_columns" not in st.session_state:
        st.warning("Train the model before saving artifacts")
        return "No artifacts saved"
    os.makedirs("exported_model", exist_ok=True)
    joblib.dump(st.session_state.model, "exported_model/model.pkl")
    joblib.dump(st.session_state.encoder, "exported_model/encoder.pkl")
    joblib.dump(st.session_state.feature_columns, "exported_model/feature.pkl")
    return "Model, Encoder, and features saved in 'exported_model/' directory."