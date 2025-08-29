# Payment Anomaly Detection Project

## Overview
The **Payment Anomaly Detection** project is a Python-based application designed to detect suspicious financial transactions, such as fraud or money laundering. It consists of two main components:
1. **Streamlit Web App**: A user-friendly interface for generating synthetic data, preprocessing, training machine learning models, evaluating performance, visualizing insights, and performing real-time anomaly detection.
2. **FastAPI Server**: An API for programmatic access, allowing external systems to predict anomalies for single or batch transactions and retrain models with new data.

The project supports two analysis modes:
- **ML-Based**: Uses machine learning models (Random Forest, Logistic Regression, XGBoost) to predict anomalies.
- **Rule-Based**: Analyzes transactions using predefined rules and visualizes results with charts and maps.

This project is ideal for learning about machine learning, data preprocessing, web apps, and APIs in Python.

## Features
- **Data Generation**: Create synthetic customer profiles and transaction data.
- **Preprocessing**: Encode categorical data, scale numerical data, and balance classes.
- **Model Training**: Train ML models and save them for reuse.
- **Evaluation**: Assess model performance with metrics and confusion matrices.
- **Insights**: Visualize feature importance and SHAP plots.
- **Real-Time Detection**: Predict anomalies via a Streamlit form or API endpoints.
- **Rule-Based Analysis**: Display charts and geographic maps for fraud patterns.
- **Data Dictionary**: View field descriptions in a searchable table.
- **API Endpoints**:
  - `/predict/{model_name}`: Predict for a single transaction.
  - `/predict_batch/{model_name}`: Predict for multiple transactions.
  - `/retrain/{model_name}`: Retrain a model with new data.

## Project Structure
```
payment_anomaly_detection/
├── data/
│   ├── raw/                    # Stores generated CSV data
│   └── retrain/                # Stores JSON data for retraining
├── models/                     # Stores trained models and preprocessing artifacts
├── app.py                      # Streamlit app entry point
├── api.py                      # FastAPI server entry point
├── data_generation.py          # Generates synthetic data
├── preprocessing.py            # Preprocesses data for ML
├── model_training.py           # Trains ML models
├── inference.py                # Handles predictions
├── rule_based.py               # Rule-based analysis and visualizations
├── insights.py                 # Feature importance and SHAP plots
├── data_dictionary.py          # Data field descriptions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Prerequisites
- **Python 3.8+**: Install from [python.org](https://www.python.org).
- **Virtual Environment** (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- **Dependencies**: Install required libraries using:
  ```bash
  pip install -r requirements.txt
  ```

## Installation
1. Clone or download the project to a local folder.
2. Create the required directory structure:
   ```bash
   mkdir -p data/raw data/retrain models
   ```
3. Save all `.py` files in the project folder.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Streamlit App
1. Navigate to the project folder in your terminal.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL (usually `http://localhost:8501`) in a browser.
4. Use the sidebar to:
   - Choose **ML-Based** or **Rule-Based** mode.
   - Set the number of transactions (100–1000).
   - Follow the workflow: Generate Data → Preprocess Data → Train Model → Evaluate Model → Show Insights (for ML).
   - For Rule-Based, select Customer-Centric or Transactional view and click Analyze.
   - Enable Real-Time Detection for live predictions.

## Running the FastAPI Server
1. Ensure models are trained (run the Streamlit app to generate and train models, saving artifacts to `models/`).
2. Navigate to the project folder in your terminal.
3. Run the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```
4. Access the API at `http://127.0.0.1:8000`. Use the interactive docs at `http://127.0.0.1:8000/docs` to test endpoints.
5. Test endpoints using tools like **Postman** or **curl**. Example for a single prediction:
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
   Response: `{"prediction": 0}` (0 = normal, 1 = anomaly).

## Usage Notes
- **Streamlit App**:
  - Generate data first to create synthetic transactions.
  - Preprocess and train models before evaluating or viewing insights.
  - Use the real-time detection form for single predictions in ML mode.
  - Rule-based mode requires only data generation and analysis selection.
- **FastAPI**:
  - Ensure models exist in `models/{model_name}/` (e.g., `models/xgboost/model.pkl`).
  - Input data must match training categories (e.g., `transaction_type` must be `wire_transfer`, `cash_deposit`, or `remittance`).
  - Retraining requires a JSON payload with `X` (transactions) and `y` (labels).
- **Shared Artifacts**: Both the Streamlit app and FastAPI use the same `models/` folder for models, encoders, and scalers, ensuring consistency.

## Troubleshooting
- **Streamlit Errors**:
  - **Missing Columns**: Ensure data generation includes all required fields.
  - **Model Not Found**: Train models in the Streamlit app first.
- **FastAPI Errors**:
  - **Model/Encoder Not Found**: Verify `models/{model_name}/` contains `model.pkl` and `encoder_*.pkl` files.
  - **Invalid Categories**: Check that input values match training data (e.g., `transaction_type`).
- **General**:
  - Add `print` statements in code to debug.
  - Check terminal logs for errors.
  - Ensure `data/` and `models/` folders exist.

## Learning Resources
- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Pandas**: [pandas.pydata.org/docs](https://pandas.pydata.org/docs)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org)
- **Pydantic**: [pydantic-docs.helpmanual.io](https://pydantic-docs.helpmanual.io)
- **Uvicorn**: [www.uvicorn.org](https://www.uvicorn.org)

## Extending the Project
- Add authentication to the API for security.
- Integrate a database for real transaction data.
- Deploy the Streamlit app (e.g., Streamlit Sharing) or FastAPI (e.g., Heroku, AWS).
- Enhance visualizations with additional metrics or interactive charts.
- Add more ML models or rule-based logic.

## License
This project is for educational purposes and provided as-is. Feel free to modify and extend it for your needs.

---

**Happy Coding! 🚀**  
For questions or contributions, contact the project maintainer or open an issue.