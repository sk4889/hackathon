from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import streamlit as st
import os
import joblib

def train_ml_model(df, model_choice, n_estimators=None):
    # Validate input DataFrame
    if df.empty:
        st.error("Input DataFrame is empty")
        return None, None, None
    if 'is_anomaly' not in df.columns:
        st.error("Input DataFrame missing 'is_anomaly' column")
        return None, None, None
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['is_anomaly']]
    X = df[feature_cols]
    y = df['is_anomaly']
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error splitting data: {str(e)}")
        return None, None, None
    
    # Store X_test indices for customer_id mapping
    st.session_state.test_indices = X_test.index.values
    
    # Initialize model
    model = None
    try:
        if model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators or 100, random_state=42)
        elif model_choice == "Isolation Forest":
            model = IsolationForest(n_estimators=n_estimators or 100, contamination=0.1, random_state=42)
        elif model_choice == "OneClassSVM":
            model = OneClassSVM(gamma='auto', nu=0.1)
        elif model_choice == "XGBoost":
            model = xgb.XGBClassifier(n_estimators=n_estimators or 100, eval_metric='logloss', random_state=42)
        else:
            st.error(f"Unknown model choice: {model_choice}")
            return None, None, None
        
        # Fit the model
        if model_choice in ["Isolation Forest", "OneClassSVM"]:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
            
        # Save model to dedicated folder
        model_dir = f"models/{model_choice.replace(' ', '_').lower()}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/model.pkl"
        joblib.dump(model, model_path)
        st.success(f"Model saved to {model_path}")
        
        return model, X_test, y_test
    
    except Exception as e:
        st.error(f"Error training {model_choice}: {str(e)}")
        return None, None, None

def evaluate_ml_model(model, X_test, y_test, col, model_choice):
    if model is None:
        col.error("No model available for evaluation")
        return
    
    feature_cols = [col for col in X_test.columns]
    X_test_features = X_test[feature_cols]
    
    try:
        if model_choice in ["Isolation Forest", "OneClassSVM"]:
            y_pred = model.predict(X_test_features)
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1, 1 (normal) to 0
        else:
            y_pred = model.predict(X_test_features)
        
        col.subheader("Model Evaluation")
        col.text("Classification Report:\n" + classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        col.pyplot(fig)
    
    except Exception as e:
        col.error(f"Error evaluating model: {str(e)}")