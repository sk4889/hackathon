# ===============================
# 1. Setup & Download PaySim Data
# ===============================

!pip install kaggle scikit-learn pandas matplotlib seaborn -q

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# Kaggle API download (requires kaggle.json in ~/.kaggle/)
if not os.path.exists("./data"):
    os.makedirs("./data")

# Download dataset via kagglehub
path = kagglehub.dataset_download("ealaxi/paysim1")
print("Path to dataset files:", path)

# Copy downloaded files into ./data
for file in os.listdir(path):
    full_src = os.path.join(path, file)
    full_dst = os.path.join("./data", file)
    if not os.path.exists(full_dst):
        shutil.copy(full_src, full_dst)

# Load dataset
df = pd.read_csv("./data/PS_20174392719_1491204439457_log.csv")
print("Data loaded:", df.shape)
print(df.head())

# ===============================
# 2. Feature Engineering
# ===============================

# Create derived flags
df['op_anomaly_flag'] = (df['amount'] > df['amount'].mean() + 3*df['amount'].std()).astype(int)

# Compliance / Sanction check (simulate unusual counterparty by random selection)
np.random.seed(42)
high_risk_entities = np.random.choice(df['nameDest'].unique(), size=500, replace=False)
df['compliance_flag'] = df['nameDest'].isin(high_risk_entities).astype(int)

# AML suspicious behavior (round-figure structuring: multiples of 1000)
df['aml_flag'] = ((df['amount'] % 1000 == 0) & (df['amount'] > 1000)).astype(int)

# Keep the original fraud label
df['fraud_flag'] = df['isFraud']

# ===============================
# 3. Prepare Features & Labels
# ===============================

# Select numeric features for models
features = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
X = df[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define multiple tasks
tasks = {
    "fraud_detection": df['fraud_flag'],
    "op_anomaly": df['op_anomaly_flag'],
    "aml_detection": df['aml_flag'],
    "compliance_risk": df['compliance_flag']
}

# ===============================
# 4. Train Models per Use Case
# ===============================

results = {}

for task_name, y in tasks.items():
    print(f"\n--- Training for {task_name} ---")
    
    # Handle extreme imbalance (fraud detection especially)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

    # Try two models: Logistic Regression & Random Forest
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    task_results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
        
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        
        task_results[model_name] = {
            "ClassificationReport": report,
            "ROC_AUC": auc
        }
        
        print(f"\nModel: {model_name}")
        print("ROC AUC:", auc)
        print(classification_report(y_test, y_pred))
        
    results[task_name] = task_results

# ===============================
# 5. Save Enriched Dataset
# ===============================

df.to_csv("./data/PaySim_enriched.csv", index=False)
print("\n✅ Enriched dataset saved as PaySim_enriched.csv")
