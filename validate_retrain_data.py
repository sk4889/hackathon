import json
import pandas as pd

def validate_retrain_data(file_path='data/retrain/retrain_data.json'):
    """Validate the retrain_data.json file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'X' not in data or 'y' not in data:
            print("Error: JSON missing 'X' or 'y' keys")
            return False
        
        X = pd.DataFrame(data['X'])
        y = data['y']
        
        required_cols = [
            'customer_id', 'timestamp', 'transaction_amount', 'transaction_type',
            'transaction_frequency', 'transaction_channel', 'counterparty_name',
            'counterparty_country', 'payment_method', 'originating_country',
            'destination_country', 'sanctioned_country', 'deviation_from_profile',
            'unusual_timing', 'structuring', 'rapid_movement', 'sanctions_list_hit',
            'pep_match', 'negative_media', 'ip_address', 'device_type',
            'account_age_days', 'last_update_days', 'failed_attempts', 'impossible_travel'
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            print(f"Error: Missing columns in X: {missing_cols}")
            return False
        
        if len(X) != len(y):
            print(f"Error: Mismatch between X ({len(X)}) and y ({len(y)}) lengths")
            return False
        
        if len(X) < 5:
            print("Error: Too few samples (< 5)")
            return False
        
        if pd.Series(y).nunique() < 2:
            print("Error: y contains only one class")
            return False
        
        if X.isna().any().any():
            print(f"Error: Missing values in columns: {X.columns[X.isna().any()].tolist()}")
            return False
        
        print("Validation successful: retrain_data.json is valid")
        return True
    except Exception as e:
        print(f"Error validating retrain_data.json: {e}")
        return False

if __name__ == "__main__":
    validate_retrain_data()