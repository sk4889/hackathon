from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
#from api_service.preprocessing import preprocess_data
#from api_service.model_training import train_model
#from api_service.inference import predict_realtime

from preprocessing import preprocess_data
from model_training import train_model
from inference import predict_realtime

app = FastAPI()

class Transaction(BaseModel):
    customer_id: str
    timestamp: str
    transaction_amount: float
    transaction_type: str
    transaction_frequency: float
    transaction_channel: str
    counterparty_name: str
    counterparty_country: str
    payment_method: str
    originating_country: str
    destination_country: str
    sanctioned_country: bool
    deviation_from_profile: float
    unusual_timing: bool
    structuring: bool
    rapid_movement: bool
    sanctions_list_hit: bool
    pep_match: bool
    negative_media: bool
    ip_address: str
    device_type: str
    account_age_days: int
    last_update_days: int
    failed_attempts: int
    impossible_travel: bool

class BatchTransaction(BaseModel):
    transactions: list[Transaction]

class RetrainData(BaseModel):
    X: list[dict]
    y: list[int]

@app.post("/predict/{model_name}")
async def predict(model_name: str, transaction: Transaction):
    try:
        data = pd.DataFrame([transaction.dict()])
        prediction = predict_realtime(data, model_name)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict_batch/{model_name}")
async def predict_batch(model_name: str, batch: BatchTransaction):
    try:
        data = pd.DataFrame([t.dict() for t in batch.transactions])
        predictions = predict_realtime(data, model_name)
        return {"predictions": [int(pred) for pred in predictions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")

@app.post("/retrain/{model_name}")
async def retrain(model_name: str, data: RetrainData):
    try:
        X = pd.DataFrame(data.X)
        X['is_anomaly'] = data.y
        df_processed = preprocess_data(X, model_name)
        if df_processed is None:
            raise Exception("Preprocessing failed")
        model, _, _ = train_model(df_processed, model_name)
        if model is None:
            raise Exception("Training failed")
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")