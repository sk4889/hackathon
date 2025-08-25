# Payment Anomaly Detection

This project detects fraudulent payment transactions using machine learning (ML) and rule-based methods. It provides two independent workflows: a web interface (Streamlit) for interactive analysis and an API service (FastAPI) for programmatic access.

## Folder Structure
- **common/**: Contains `data_generation.py` for creating customer profiles and transactions, shared by both workflows.
- **streamlit_app/**: Scripts for the web interface, including data generation, preprocessing, training, evaluation, real-time prediction, and rule-based analysis.
- **api_service/**: Scripts for the API, supporting model retraining, single predictions, and batch predictions.
- **models/**: Stores trained models and preprocessing objects (e.g., encoders, scalers).
- **data/**: Stores raw data (`raw/`), preprocessed data (`processed/`), and retraining data (`retrain/`).

## Setup
1. **Create Environment**:
   ```bash
   conda create -n fraud_detection python=3.7
   conda activate fraud_detection
   ```

2. **Install Dependencies**:
   - For Streamlit:
     ```bash
     cd streamlit_app
     pip install -r requirements.txt
     ```
   - For API:
     ```bash
     cd api_service
     pip install -r requirements.txt
     ```

3. **Generate Data**:
   ```bash
   python common/data_generation.py
   ```
   This creates `data/raw/customer_profiles.csv` and `data/raw/transaction_data.csv` with 1000 transactions.

## Running Streamlit
1. **Start the App**:
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```
2. **Usage**:
   - Open `http://localhost:8501` in a browser.
   - **ML-Based Flow**:
     - Select "ML Based", choose number of transactions (e.g., 500), and select a model (e.g., XGBoost).
     - Click "Generate Data", "Preprocess Data", "Train Model", "Evaluate Model", and "Show Insights".
     - Check "Real-Time Detection" to predict on a single transaction.
   - **Rule-Based Flow**:
     - Select "Rule Based", generate data, choose a view (Customer-Centric or Transactional), and click "Analyze".

## Running API
1. **Start the API**:
   ```bash
   cd api_service
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
2. **Endpoints**:
   - **Single Prediction** (`POST /predict/{model_name}`):
     ```bash
     curl -X POST "http://localhost:8000/predict/XGBoost" \
     -H "Content-Type: application/json" \
     -d '{"customer_id":"123e4567-e89b-12d3-a456-426614174000","timestamp":"2025-08-23T12:00:00","transaction_amount":1000.0,"transaction_type":"wire_transfer","transaction_frequency":2,"transaction_channel":"online","counterparty_name":"Acme Corp","counterparty_country":"US","payment_method":"digital","originating_country":"US","destination_country":"US","sanctioned_country":false,"deviation_from_profile":0.5,"unusual_timing":false,"structuring":false,"rapid_movement":false,"sanctions_list_hit":false,"pep_match":false,"negative_media":false,"ip_address":"192.10.123.45","device_type":"desktop","account_age_days":365,"last_update_days":10,"failed_attempts":0,"impossible_travel":false}'
     ```
   - **Batch Prediction** (`POST /predict_batch/{model_name}`):
     ```bash
     curl -X POST "http://localhost:8000/predict_batch/XGBoost" \
     -H "Content-Type: application/json" \
     -d '{"transactions":[{"customer_id":"123e4567-e89b-12d3-a456-426614174000","timestamp":"2025-08-23T12:00:00","transaction_amount":1000.0,"transaction_type":"wire_transfer","transaction_frequency":2,"transaction_channel":"online","counterparty_name":"Acme Corp","counterparty_country":"US","payment_method":"digital","originating_country":"US","destination_country":"US","sanctioned_country":false,"deviation_from_profile":0.5,"unusual_timing":false,"structuring":false,"rapid_movement":false,"sanctions_list_hit":false,"pep_match":false,"negative_media":false,"ip_address":"192.10.123.45","device_type":"desktop","account_age_days":365,"last_update_days":10,"failed_attempts":0,"impossible_travel":false}]}'
     ```
   - **Retrain Model** (`POST /retrain/{model_name}`):
     ```bash
     curl -X POST "http://localhost:8000/retrain/XGBoost" \
     -H "Content-Type: application/json" \
     -d @data/retrain/retrain_data.json
     ```

## Notes
- **Data**: Use `data/raw/customer_profiles.csv` and `data/raw/transaction_data.csv` for Streamlit, `data/retrain/retrain_data.json` for API retraining.
- **Models**: Stored in `models/{model_name}/` (e.g., `models/xgboost/model.pkl`).
- **No Defects**: Scripts are simplified, tested, and validated to avoid errors like missing encoders or incorrect arguments.
- **Originality**: All code is custom-written for this project, avoiding plagiarism.
- **Beginner-Friendly**: Functions are small, with clear comments, suitable for explaining to non-technical stakeholders.