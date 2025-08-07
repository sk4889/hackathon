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
