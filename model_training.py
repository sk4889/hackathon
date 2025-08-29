import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
from imblearn.over_sampling import SMOTE

def train_model(df, model_choice):
    feature_cols = [
        'transaction_type', 'transaction_channel', 'payment_method', 'originating_country',
        'destination_country', 'counterparty_country', 'transaction_amount', 'transaction_frequency',
        'sanctioned_country', 'deviation_from_profile', 'unusual_timing', 'structuring',
        'rapid_movement', 'sanctions_list_hit', 'pep_match', 'negative_media',
        'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel'
    ]
    X = df[feature_cols]
    y = df['is_anomaly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model_dir = f"models/{model_choice.lower().replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    if model_choice == "All":
        trained_models = {}
        metrics = {}
        best_model_name = None
        best_f1 = -1

        for name, model in models.items():
            model.fit(X_train_resampled, y_train_resampled)
            trained_models[name] = model
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_score = report['weighted avg']['f1-score']
            metrics[name] = {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': f1_score,
                'accuracy': report['accuracy']
            }
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_name = name

        model_path = os.path.join("models", "all", "best_model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(trained_models[best_model_name], model_path)
        return trained_models, X_test, y_test, trained_models[best_model_name]

    else:
        model = models[model_choice]
        model.fit(X_train_resampled, y_train_resampled)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        return model, X_test, y_test

def evaluate_model(models, X_test, y_test, column, model_choice):
    if model_choice == "All":
        column.subheader("Model Comparison")
        metrics_df = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1-Score", "Accuracy"])
        best_model_name = None
        best_f1 = -1

        for name, model in models.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics_df = metrics_df.append({
                "Model": name,
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-Score": report['weighted avg']['f1-score'],
                "Accuracy": report['accuracy']
            }, ignore_index=True)
            if report['weighted avg']['f1-score'] > best_f1:
                best_f1 = report['weighted avg']['f1-score']
                best_model_name = name

        column.write("### Performance Metrics")
        column.dataframe(metrics_df.style.highlight_max(subset=["F1-Score", "Accuracy"], color='lightgreen'))
        column.write(f"**Best Model (based on F1-Score):** {best_model_name}")

        for name, model in models.items():
            column.subheader(f"Confusion Matrix: {name}")
            cm = confusion_matrix(y_test, model.predict(X_test))
            cm_df = pd.DataFrame(cm, index=['Normal', 'Anomaly'], columns=['Predicted Normal', 'Predicted Anomaly'])
            column.dataframe(cm_df)
    else:
        model = models[model_choice]
        column.subheader(f"Evaluation: {model_choice}")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        column.text("Classification Report:")
        column.text(report)
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_df = pd.DataFrame(cm, index=['Normal', 'Anomaly'], columns=['Predicted Normal', 'Predicted Anomaly'])
        column.dataframe(cm_df)