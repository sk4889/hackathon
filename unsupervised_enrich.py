import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# -------------------------------
# Parameters
# -------------------------------
N_CUSTOMERS = 500
N_TRANSACTIONS = 10000
FRAUD_RATIO = 0.05
SANCTIONED_COUNTRIES = ["IR", "KP", "SY", "CU", "RU"]
COUNTRIES = ["US", "IN", "GB", "DE", "FR", "SG", "CN", "JP", "BR", "ZA"] + SANCTIONED_COUNTRIES

np.random.seed(42)

# -------------------------------
# Helper Functions
# -------------------------------
def random_date(start, end):
    """Generate a random datetime between two datetime objects."""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

# -------------------------------
# Generate Customers
# -------------------------------
customers = []
for cid in range(N_CUSTOMERS):
    base_amount = np.random.randint(50, 2000)  # avg spend per transaction
    base_country = np.random.choice(COUNTRIES, p=[0.15,0.15,0.1,0.1,0.1,0.05,0.1,0.05,0.1,0.1,0.0,0.0,0.0,0.0,0.0])
    customers.append({
        "customer_id": cid,
        "base_amount": base_amount,
        "home_country": base_country
    })

df_customers = pd.DataFrame(customers)

# -------------------------------
# Generate Transactions
# -------------------------------
transactions = []
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

for tid in range(N_TRANSACTIONS):
    cust = df_customers.sample(1).iloc[0]
    tx_date = random_date(start_date, end_date)
    amount = np.random.normal(loc=cust.base_amount, scale=cust.base_amount*0.3)
    amount = max(1, round(amount, 2))
    country = np.random.choice(COUNTRIES)
    channel = np.random.choice(["POS", "WEB", "MOBILE", "ATM", "SWIFT"])

    # Fraud label (only label we expose)
    fraud_flag = 0

    # --- Fraud Injection ---
    if np.random.rand() < FRAUD_RATIO:
        fraud_flag = 1
        # unusual location or device
        if country == cust.home_country:
            country = np.random.choice([c for c in COUNTRIES if c != cust.home_country])
        # spike in amount
        amount = amount * np.random.randint(3, 10)

    # Note: Operational anomaly, sanction, and AML-like signals will be embedded
    # in the raw data (e.g., high amounts, sanctioned countries, round figures),
    # but NOT directly labeled.

    transactions.append({
        "transaction_id": tid,
        "customer_id": cust.customer_id,
        "timestamp": tx_date,
        "amount": round(amount,2),
        "currency": "USD",
        "channel": channel,
        "country": country,
        "fraud_flag": fraud_flag
    })

# -------------------------------
# Final Dataset
# -------------------------------
df_transactions = pd.DataFrame(transactions)

# Sort by timestamp
df_transactions = df_transactions.sort_values("timestamp").reset_index(drop=True)

# -------------------------------
# Feature Engineering Function
# -------------------------------
def add_derived_features(df_tx, df_cust):
    merged = df_tx.merge(df_cust, on="customer_id", how="left")
    
    # Operational anomaly flag (derived)
    merged["derived_op_anomaly"] = (merged["amount"] > 5 * merged["base_amount"]).astype(int)
    
    # Sanctioned transaction flag (derived)
    merged["derived_sanction"] = merged["country"].isin(SANCTIONED_COUNTRIES).astype(int)
    
    # AML-like flags (derived)
    merged["derived_aml_round"] = (merged["amount"] % 1000 == 0).astype(int)
    merged["derived_aml_structured"] = ((merged["amount"] > 9000) & (merged["amount"] < 10000)).astype(int)
    
    return merged

# Apply feature engineering
df_transactions_enriched = add_derived_features(df_transactions, df_customers)

# Save enriched dataset
df_transactions_enriched.to_csv("synthetic_transactions_enriched.csv", index=False)

print("Dataset created: synthetic_transactions_enriched.csv with derived features")
print(df_transactions_enriched.head(10))
