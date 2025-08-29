# API Guide for Payment Anomaly Detection Project

This guide is tailored for Python beginners to understand the API component of the Payment Anomaly Detection project, which complements the previously discussed Streamlit web app. The API, built with **FastAPI**, allows external systems to interact with the anomaly detection system by sending transactions for prediction or retraining models. Below, I'll explain the API files (`api.py`, updated `model_training.py`, `preprocessing.py`, `inference.py`), their connections to the original project, and provide a workflow diagram for the API. The guide assumes basic Python knowledge and familiarity with the original project's structure.

## 1. API Overview

The API provides three endpoints:
- **Single Prediction (`/predict/{model_name}`)**: Predicts if one transaction is anomalous using a specified ML model.
- **Batch Prediction (`/predict_batch/{model_name}`)**: Predicts for multiple transactions.
- **Model Retraining (`/retrain/{model_name}`)**: Retrains a model with new data.

It integrates with the original project's `preprocessing.py`, `model_training.py`, and `inference.py`, reusing their logic for data preprocessing, model training, and predictions. The API uses **FastAPI** for handling HTTP requests and **Pydantic** for data validation.

**Goal**: Enable external applications (e.g., banking systems) to use the anomaly detection system programmatically.

## 2. Prerequisites

- **Python 3.8+**: Installed from python.org.
- **Additional Libraries**: Install FastAPI and Uvicorn (for running the API server):
  ```bash
  pip install fastapi uvicorn
  ```
  Ensure other libraries from the original project are installed:
  ```bash
  pip install pandas numpy scikit-learn imbalanced-learn joblib xgboost
  ```
- **Project Structure**: Place `api.py`, `preprocessing.py`, `model_training.py`, and `inference.py` in the same folder as the original project (with `data/` and `models/` subfolders).
- **Pre-trained Models**: Ensure models and preprocessing artifacts (encoders, scalers) exist in `models/` from the Streamlit app's training process.

## 3. File-by-File Explanation

### 3.1 `api.py`
- **Purpose**: Defines the FastAPI application with endpoints for prediction and retraining.
- **Key Concepts**:
  - Uses **Pydantic** (`BaseModel`) to define the structure of input data (transactions and retraining data).
  - Handles HTTP POST requests for predictions and retraining.
  - Integrates with `preprocessing.py`, `model_training.py`, and `inference.py`.
- **Main Components**:
  - **Pydantic Models**:
    - `Transaction`: Defines a single transaction's fields (e.g., `customer_id`, `transaction_amount`).
    - `BatchTransaction`: A list of transactions for batch predictions.
    - `RetrainData`: Transaction data (`X`) and labels (`y`) for retraining.
  - **Endpoints**:
    - `/predict/{model_name}`: Takes one transaction, returns a prediction (0 = normal, 1 = anomaly).
    - `/predict_batch/{model_name}`: Takes multiple transactions, returns a list of predictions.
    - `/retrain/{model_name}`: Retrains a model with new data.
- **Code Snippet Example** (Simplified):
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel

  app = FastAPI()

  class Transaction(BaseModel):
      customer_id: str
      transaction_amount: float
      # ... other fields

  @app.post("/predict/{model_name}")
  async def predict(model_name: str, transaction: Transaction):
      data = pd.DataFrame([transaction.dict()])
      prediction = predict_realtime(data, model_name)
      return {"prediction": int(prediction[0])}
  ```
- **Beginner Tip**: FastAPI's `@app.post` creates an endpoint. `async` allows handling multiple requests efficiently. Pydantic ensures input data matches the expected format.

### 3.2 `preprocessing.py`
- **Purpose**: Preprocesses transaction data (encodes categorical variables, scales numerical ones) for training or prediction.
- **Key Updates**:
  - Modified to work with the API (removes Streamlit dependencies like `st.error`).
  - Uses absolute paths for saving encoders/scalers to ensure compatibility.
  - Prints errors instead of using Streamlit UI.
- **Main Function**: `preprocess_data(data, model_name)`: Validates data, encodes categories, scales numbers, and saves artifacts.
- **Code Snippet**:
  ```python
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  import joblib

  def preprocess_data(data, model_name):
      categorical_cols = ['transaction_type', 'transaction_channel', 'payment_method']
      numerical_cols = ['transaction_amount', 'transaction_frequency']
      df_processed = data.copy()
      for col in categorical_cols:
          le = LabelEncoder()
          df_processed[col] = le.fit_transform(df_processed[col].astype(str))
          joblib.dump(le, f'models/{model_name}/encoder_{col}.pkl')
      scaler = StandardScaler()
      df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
      joblib.dump(scaler, f'models/{model_name}/scaler.pkl')
      return df_processed
  ```
- **Tip**: This is the same preprocessing logic as the Streamlit app but adapted for API use (no UI).

### 3.3 `model_training.py`
- **Purpose**: Trains ML models (Logistic Regression, Random Forest, XGBoost) and saves them.
- **Key Updates**:
  - Removes Streamlit dependencies and SMOTE (class balancing is now handled in `preprocessing.py`).
  - Adds strict validation for data (non-empty, correct columns, numeric types).
  - Uses absolute paths for saving models.
- **Main Function**: `train_model(data, model_name)`: Trains a model, saves it, and returns the model with test data.
- **Code Snippet**:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  import joblib

  def train_model(data, model_name):
      feature_cols = ['transaction_type', 'transaction_amount']  # Simplified
      X = data[feature_cols]
      y = data['is_anomaly']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      model = RandomForestClassifier()
      model.fit(X_train, y_train)
      joblib.dump(model, f'models/{model_name}/model.pkl')
      return model, X_test, y_test
  ```
- **Tip**: The model is saved as a `.pkl` file, which the API loads for predictions.

### 3.4 `inference.py`
- **Purpose**: Handles predictions for the API by loading models and preprocessing artifacts.
- **Key Updates**:
  - Removes Streamlit form/UI logic.
  - Loads encoders/scalers and model from `models/` folder.
  - Validates input data and applies preprocessing before prediction.
- **Main Function**: `predict_realtime(data, model_name)`: Preprocesses input data and returns predictions.
- **Code Snippet**:
  ```python
  import joblib

  def predict_realtime(data, model_name):
      model = joblib.load(f'models/{model_name}/model.pkl')
      encoders = {col: joblib.load(f'models/{model_name}/encoder_{col}.pkl') for col in categorical_cols}
      scaler = joblib.load(f'models/{model_name}/scaler.pkl')
      df = data.copy()
      for col in categorical_cols:
          df[col] = encoders[col].transform(df[col].astype(str))
      df[numerical_cols] = scaler.transform(df[numerical_cols])
      return model.predict(df)
  ```
- **Tip**: This function is reused by both the Streamlit app and API, ensuring consistent predictions.

## 4. How the Files Connect

- **api.py** is the entry point for the API, defining endpoints that external systems call.
  - **Prediction Endpoints** (`/predict`, `/predict_batch`): Call `inference.py`'s `predict_realtime` to preprocess and predict.
  - **Retrain Endpoint** (`/retrain`): Calls `preprocessing.py` to preprocess data, then `model_training.py` to train and save the model.
- **preprocessing.py**: Prepares data for both training (retrain endpoint) and prediction (inference).
- **model_training.py**: Used by the retrain endpoint to train and save models.
- **inference.py**: Used by prediction endpoints to load models and preprocess data.
- **Connection to Original Project**:
  - The API reuses the same `preprocessing.py`, `model_training.py`, and `inference.py` logic as the Streamlit app, but without UI components.
  - Models and artifacts (encoders, scalers) are stored in the same `models/` folder, allowing the API to use models trained by the Streamlit app.
  - The API does not use `data_generation.py`, `rule_based.py`, `insights.py`, or `data_dictionary.py`, as these are specific to the Streamlit UI.

## 5. Workflow Diagram

Below is a text-based (ASCII) diagram of the API workflow. Arrows show the flow of data and requests.

```
+-------------------+     +-------------------+
| Client Sends      |     | FastAPI Server    |
| HTTP Request      |---->| (api.py)          |
| (Postman, curl)   |     +-------------------+
+-------------------+               |
                                   v
+-------------------+     +-------------------+
| /predict          |     | /predict_batch    |
| Single Transaction|     | Multiple Trans.   |
| (inference.py)    |     | (inference.py)    |
+-------------------+     +-------------------+
           |                        |
           v                        v
+-------------------+     +-------------------+
| Preprocess Data   |<----| Preprocess Data   |
| (preprocessing.py)|     | (preprocessing.py)|
+-------------------+     +-------------------+
           |                        |
           v                        v
+-------------------+     +-------------------+
| Load Model &      |     | Load Model &      |
| Predict           |     | Predict           |
| (inference.py)    |     | (inference.py)    |
+-------------------+     +-------------------+
           |                        |
           v                        v
+-------------------+     +-------------------+
| Return Prediction |     | Return Predictions|
| (0 or 1)          |     | (List of 0/1)    |
+-------------------+     +-------------------+

+-------------------+
| /retrain          |
| (api.py)          |
+-------------------+
           |
           v
+-------------------+
| Preprocess Data   |
| (preprocessing.py)|
+-------------------+
           |
           v
+-------------------+
| Train Model       |
| (model_training)  |
+-------------------+
           |
           v
+-------------------+
| Save Model        |
| (models/ folder)  |
+-------------------+
           |
           v
+-------------------+
| Return Success    |
| Message           |
+-------------------+
```

- **Prediction Flow**: Client → API Endpoint → Preprocess → Predict → Response.
- **Retrain Flow**: Client → API Endpoint → Preprocess → Train → Save → Response.
- **Shared Artifacts**: Models, encoders, and scalers in `models/` folder are used by both API and Streamlit app.

## 6. Running the API

1. **Save Files**: Ensure `api.py`, `preprocessing.py`, `model_training.py`, and `inference.py` are in the project folder with `models/` and `data/` subfolders.
2. **Train a Model First** (if not already done):
   - Run the Streamlit app (`streamlit run app.py`) to generate data, preprocess, and train a model (e.g., XGBoost), which saves artifacts to `models/`.
   - Alternatively, use a script to call `generate_data`, `preprocess_data`, and `train_model` to create artifacts.
3. **Run the API**:
   - In the terminal, navigate to the project folder.
   - Run: `uvicorn api:app --reload`
   - The API starts at `http://127.0.0.1:8000`.
4. **Test Endpoints**:
   - Use **Postman**, **curl**, or the auto-generated FastAPI docs at `http://127.0.0.1:8000/docs`.
   - Example curl for single prediction:
     ```bash
     curl -X POST "http://127.0.0.1:8000/predict/XGBoost" -H "Content-Type: application/json" -d '{
         "customer_id": "123e4567-e89b-12d3-a456-426614174000",
         "timestamp": "2025-08-29T12:00:00",
         "transaction_amount": 1000.0,
         "transaction_type": "wire_transfer",
         "transaction_frequency": 1.0,
         "transaction_channel": "online",
         "counterparty_name": "Acme Corp",
         "counterparty_country": "US",
         "payment_method": "digital",
         "originating_country": "US",
         "destination_country": "UK",
         "sanctioned_country": false,
         "deviation_from_profile": 0.1,
         "unusual_timing": false,
         "structuring": false,
         "rapid_movement": false,
         "sanctions_list_hit": false,
         "pep_match": false,
         "negative_media": false,
         "ip_address": "192.10.123.45",
         "device_type": "laptop",
         "account_age_days": 365,
         "last_update_days": 10,
         "failed_attempts": 0,
         "impossible_travel": false
     }'
     ```
   - Response: `{"prediction": 0}` (0 = normal, 1 = anomaly).
5. **Retrain Example**:
   - Send a POST request to `/retrain/XGBoost` with a JSON payload containing `X` (list of transactions) and `y` (list of labels).

## 7. Tips for Beginners

- **Debugging**: Check terminal logs for errors (e.g., missing model files). Add `print` statements in `preprocessing.py` or `inference.py` to trace issues.
- **Common Issues**:
  - **Model Not Found**: Ensure `models/{model_name}/model.pkl` exists (run Streamlit app to train first).
  - **Invalid Categories**: Input values (e.g., `transaction_type`) must match those used during training (e.g., `wire_transfer`, not `wire`).
  - **File Paths**: Absolute paths in `preprocessing.py` and `inference.py` prevent path issues.
- **Testing the API**:
  - Use FastAPI's `/docs` for an interactive UI to test endpoints.
  - Start with a single prediction to verify setup before trying batch or retrain.
- **Learning Resources**:
  - FastAPI: `fastapi.tiangolo.com` (tutorials, docs).
  - Pydantic: `pydantic-docs.helpmanual.io` (data validation).
  - Uvicorn: `www.uvicorn.org` (running FastAPI).
- **Extensions**:
  - Add authentication to secure endpoints.
  - Integrate with a database for real transaction data.
  - Deploy to a cloud service (e.g., Heroku, AWS) for production use.

## 8. Integration with Original Project

- **Shared Components**: The API uses the same `preprocessing.py`, `model_training.py`, and `inference.py` as the Streamlit app, ensuring consistent data handling and predictions.
- **Differences**:
  - The API removes Streamlit dependencies and UI logic.
  - It adds HTTP endpoints for external access, unlike the Streamlit app's browser-based UI.
  - Retraining is API-specific; the Streamlit app trains models interactively.
- **Workflow**:
  - Use the Streamlit app to generate data and train initial models.
  - Use the API to serve predictions or retrain models programmatically.
  - Both share the `models/` folder for storing/loading artifacts.

This API extends the Payment Anomaly Detection project to support programmatic access, making it versatile for integration with other systems. If you have questions or want to test specific endpoints, let me know! 🚀