import sys
import os
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.getcwd())

from data_processor import process_data, feature_engineering
from model_trainer import train_model
from visualizer import visualize_results_pre_modeling, visualize_results_post_modeling

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime.now() - timedelta(days=1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "trigger_rule": "all_success",
}

dag = DAG(
    "ml_pipeline",
    default_args=default_args,
    description="A simple ML pipeline",
    schedule_interval="@daily",
)

process_data_task = PythonOperator(
    task_id="process_data",
    python_callable=process_data,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

visualize_results_pre_model_task = PythonOperator(
    task_id="visualize_results_pre_modeling",
    python_callable=visualize_results_pre_modeling,
    dag=dag,
)

visualize_results_post_model_task = PythonOperator(
    task_id="visualize_results_post_modeling",
    python_callable=visualize_results_post_modeling,
    dag=dag,
)

(
    process_data_task
    >> feature_engineering_task
    >> visualize_results_pre_model_task
    >> train_model_task
    >> visualize_results_post_model_task
)
