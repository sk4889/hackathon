# Hackathon Reviewer: Payment Anomaly Detection

## Slide 1: Title & Project Overview
- **Content**:
  - Title: "Payment Anomaly Detection: AI Meets Fintech"
  - Subtitle: "A Dual-Mode System for Fraud Detection"
  - Overview:
    - Streamlit web app for interactive fraud analysis.
    - FastAPI for programmatic predictions and retraining.
    - Combines ML (XGBoost, Random Forest, Logistic Regression) and rule-based analysis.
  - Visual: Project logo or split image (Streamlit UI + API docs).
- **Notes**:
  - Start with impact: “We built a system to catch financial fraud in real-time.”
  - Mention hackathon relevance: “Showcases AI, web dev, and API integration.”

## Slide 2: Problem Statement
- **Content**:
  - Header: "Why Fraud Detection Matters"
  - Bullet Points:
    - Financial fraud costs $5T annually (ACFE).
    - Challenges: Complex fraud patterns, imbalanced data, real-time needs.
    - Current solutions: Slow manual reviews or rigid rule-based systems.
  - Visual: Bar chart of fraud losses or example fraud case.
- **Notes**:
  - Highlight technical challenge: “Imbalanced data makes ML tricky.”

## Slide 3: System Architecture
- **Content**:
  - Header: "How It Works"
  - Diagram: Streamlit app + FastAPI server.
    - Streamlit: Data generation → Preprocessing → Training → Insights → Real-time.
    - FastAPI: Predict (single/batch) → Retrain.
  - Shared components: `models/` folder for artifacts.
  - Tech Stack: Python, Streamlit, FastAPI, Pandas, Scikit-learn, XGBoost, SHAP, Folium.
  - Visual: Architecture diagram (Streamlit + FastAPI with arrows to `models/`).
- **Notes**:
  - Explain modularity: “Shared preprocessing and inference ensure consistency.”

## Slide 4: Data Generation
- **Content**:
  - Header: "Synthetic Data for Testing"
  - Bullet Points:
    - File: `data_generation.py`.
    - Uses Faker for realistic data (e.g., customer IDs, IPs).
    - Generates customer profiles and transactions (e.g., `transaction_amount`, `impossible_travel`).
    - Saves to `data/raw/` (CSV) and `data/retrain/` (JSON).
  - Visual: Sample DataFrame output or code snippet:
    ```python
    customers = pd.DataFrame({
        'customer_id': [fake.uuid4() for _ in range(100)],
        'nationality': np.random.choice(['US', 'IN', 'UK'], 100)
    })
    ```
- **Notes**:
  - Emphasize realism: “Mimics real-world banking data.”

## Slide 5: Preprocessing
- **Content**:
  - Header: "Data Prep for ML"
  - Bullet Points:
    - File: `preprocessing.py`.
    - LabelEncoder for categorical columns (e.g., `transaction_type`).
    - StandardScaler for numerical columns (e.g., `transaction_amount`).
    - Saves encoders/scalers to `models/` for reuse.
  - Visual: Before/after DataFrame (raw vs. encoded/scaled).
- **Notes**:
  - Highlight robustness: “Handles missing columns and invalid data.”

## Slide 6: Model Training
- **Content**:
  - Header: "Training ML Models"
  - Bullet Points:
    - File: `model_training.py`.
    - Models: Random Forest, Logistic Regression, XGBoost.
    - Splits data (80% train, 20% test).
    - Saves models to `models/{model_name}/model.pkl`.
  - Visual: Code snippet:
    ```python
    model = XGBClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/xgboost/model.pkl')
    ```
- **Notes**:
  - Mention validation: “Checks for missing or non-numeric data.”

## Slide 7: Inference & API
- **Content**:
  - Header: "Real-Time Predictions"
  - Bullet Points:
    - File: `inference.py` (used by both Streamlit and API).
    - Streamlit: Interactive form for single predictions.
    - FastAPI: Endpoints `/predict/{model_name}`, `/predict_batch/{model_name}`.
    - Loads models/encoders from `models/`.
  - Visual: API curl example:
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict/XGBoost" -d '{...}'
    ```
- **Notes**:
  - Stress flexibility: “API enables integration with any system.”

## Slide 8: Visualizations & Insights
- **Content**:
  - Header: "Understanding Fraud Patterns"
  - Bullet Points:
    - Files: `rule_based.py`, `insights.py`.
    - Rule-Based: Charts (Seaborn) and maps (Folium) for fraud analysis.
    - ML Insights: Feature importance and SHAP plots.
  - Visual: Screenshot of Streamlit charts or SHAP plot.
- **Notes**:
  - Highlight dual approach: “Rule-based for quick insights, ML for precision.”

## Slide 9: API Retraining
- **Content**:
  - Header: "Keeping Models Fresh"
  - Bullet Points:
    - File: `api.py` (`/retrain/{model_name}`).
    - Takes new data (transactions + labels), preprocesses, and retrains.
    - Saves updated model to `models/`.
  - Visual: JSON payload example:
    ```json
    {"X": [{"customer_id": "...", ...}], "y": [0, 1, ...]}
    ```
- **Notes**:
  - Emphasize scalability: “Adapts to new fraud patterns.”

## Slide 10: Conclusion & Future Work
- **Content**:
  - Header: "Why This Wins"
  - Bullet Points:
    - Technical: Robust ML pipeline, modular code, API integration.
    - Functional: User-friendly UI, real-time detection, actionable insights.
    - Future: Add authentication, database integration, cloud deployment.
  - Visual: Demo screenshot or team photo.
- **Notes**:
  - End with impact: “This is a scalable, real-world solution for fintech.”
  - Invite feedback: “What features would you add?”