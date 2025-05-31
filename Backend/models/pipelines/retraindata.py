from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import joblib
import numpy as np
import json

def retrain_model():
    # Charger les données driftées
    with open('drifted_data.json', 'r') as f:
        drifted_data = json.load(f)
    
    # Charger le modèle existant
    model = joblib.load("./models/data/diabetes_model.pkl")
    
    # Préparer les données
    X_train = np.array([entry["data"] for entry in drifted_data])
    
    # Réentraîner le modèle
    model.fit(X_train, [entry["result"] for entry in drifted_data])
    
    # Sauvegarder le modèle réentrainé
    joblib.dump(model, './models/data/diabetes_model.pkl')

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_model',
    default_args=default_args,
    description='Réentraîner le modèle chaque semaine',
    schedule_interval=timedelta(weeks=1),
    start_date=datetime(2025, 4, 17),
    catchup=False,
)

retrain_task = PythonOperator(
    task_id='retrain_model_task',
    python_callable=retrain_model,
    dag=dag,
)
