# dags/data_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor

def collect_and_preprocess():
    # 1. Collect Data
    collector = DataCollector()
    df = collector.from_sql(
        db_type='mysql',
        host='localhost',
        port=3306,
        database='sales',
        user='root',
        password='password',
        query='SELECT * FROM customers'
    )

    # 2. Preprocess Data
    processor = (
        DataPreprocessor(df)
        .standardize_column_names()
        .handle_missing_values(strategy='mean')
        .encode_categorical(method='label')
        .normalize(method='standard')
    )
    clean_df = processor.get_processed_data()

    # 3. Save to file
    clean_df.to_csv('/tmp/clean_data.csv', index=False)

default_args = {
    'start_date': datetime(2025, 8, 5),
    'catchup': False,
}

with DAG("data_ingestion_preprocessing_pipeline",
         default_args=default_args,
         schedule_interval="@daily",
         tags=["ai_pipeline"]) as dag:

    task = PythonOperator(
        task_id="collect_and_preprocess_data",
        python_callable=collect_and_preprocess
    )

    task





1. Customer Information (KYC-related)
	•	Customer ID / Account number
	•	Customer type (individual, corporate, NGO, PEP – Politically Exposed Person)
	•	Risk rating (low, medium, high)
	•	Nationality / Country of residence (high-risk jurisdiction)
	•	Occupation / Business type
2. Transaction Information
	•	Transaction amount
	•	Transaction type (cash deposit, wire transfer, remittance, crypto, trade finance, etc.)
	•	Transaction frequency (number of transactions per day/week/month)
	•	Transaction channel (branch, online, ATM, mobile, SWIFT, etc.)
	•	Counterparty details (beneficiary/sender name, country, bank, account)
	•	Payment method (cash, cheque, digital, prepaid card, etc.)
3. Geographical Attributes
	•	Originating country
	•	Destination country
	•	High-risk or sanctioned country (FATF, OFAC, EU blacklist, etc.)
4. Behavioral Attributes
	•	Deviation from customer profile (e.g., sudden high-value transfers)
	•	Unusual timing (odd hours, end of reporting cycles)
	•	Structuring/smurfing (many small transactions under reporting threshold)
	•	Rapid movement of funds (in-and-out within short time = layering)
5. Regulatory & Watchlist Attributes
	•	Sanctions list hits (OFAC, UN, EU, local regulator)
	•	PEP list match
	•	Negative media flags
