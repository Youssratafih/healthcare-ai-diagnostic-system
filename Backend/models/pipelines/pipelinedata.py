from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from airflow.configuration import conf
from mlflow.models import infer_signature

# Activer le pickling des objets dans XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Définir le propriétaire du DAG et les arguments par défaut
dag_owner = 'chaimae'
default_args = {
    'owner': dag_owner,
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5)
}

# Initialisation du DAG
with DAG(
    dag_id='diabetes_prediction_dag',
    default_args=default_args,
    description='A DAG for training and monitoring the diabetes prediction model',
    start_date=datetime(2025, 4, 15),
    schedule_interval='@weekly',
    catchup=False,
    tags=['diabetes', 'ml']
) as dag:

    # Task 1: Préparer les données
    def prepare_data(**kwargs):
        # Chargement du jeu de données
        diabetes_dataset = pd.read_csv('./data/diabetes.csv')

        # Séparer les données et les étiquettes
        X = diabetes_dataset.drop(columns='Outcome', axis=1)
        Y = diabetes_dataset['Outcome']

        # Standardisation des données
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # Séparer les données en ensembles de formation et de test
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

        # Passer les résultats à la tâche suivante via XCom
        kwargs['ti'].xcom_push(key='X_train', value=X_train)
        kwargs['ti'].xcom_push(key='X_test', value=X_test)
        kwargs['ti'].xcom_push(key='Y_train', value=Y_train)
        kwargs['ti'].xcom_push(key='Y_test', value=Y_test)
        kwargs['ti'].xcom_push(key='scaler', value=scaler)

    # Task 2: Entraîner le modèle
    def train_model(**kwargs):
        # Récupérer les données de la tâche précédente via XCom
        ti = kwargs['ti']
        X_train = ti.xcom_pull(task_ids='prepare_data', key='X_train')
        Y_train = ti.xcom_pull(task_ids='prepare_data', key='Y_train')

        # Initialiser le modèle SVM
        classifier = svm.SVC(kernel='linear')

        # Entraîner le modèle
        classifier.fit(X_train, Y_train)

        # Enregistrer le modèle avec Joblib
        joblib.dump(classifier, './data/diabetes_model.pkl')

        # Passer le modèle à la tâche suivante via XCom
        kwargs['ti'].xcom_push(key='classifier', value=classifier)

    # Task 3: Évaluer et enregistrer les performances avec MLflow
    def evaluate_and_log_model(**kwargs):
        # Récupérer les données et le modèle depuis les tâches précédentes via XCom
        ti = kwargs['ti']
        X_train = ti.xcom_pull(task_ids='prepare_data', key='X_train')
        Y_train = ti.xcom_pull(task_ids='prepare_data', key='Y_train')
        X_test = ti.xcom_pull(task_ids='prepare_data', key='X_test')
        Y_test = ti.xcom_pull(task_ids='prepare_data', key='Y_test')
        classifier = ti.xcom_pull(task_ids='train_model', key='classifier')

        # Prédiction et évaluation
        X_train_prediction = classifier.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        
        X_test_prediction = classifier.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

        print(f'Accuracy on training data: {training_data_accuracy}')
        print(f'Accuracy on test data: {test_data_accuracy}')

        # Initialiser MLflow
       # Set our tracking server uri for logging
        mlflow.set_tracking_uri("http://mlflow:5000")


        # Create a new MLflow Experiment
        mlflow.set_experiment("MLflow Quickstart")

        # Start an MLflow run
        with mlflow.start_run():
            # Log the hyperparameters
            

            # Log the loss metric
            mlflow.log_metric("accuracy", test_data_accuracy)

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")

            # Infer the model signature
            signature = infer_signature(X_train, classifier.predict(X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=classifier,
                artifact_path="iris_model",
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-quickstart",
            )

        # Passer l'exactitude pour la prédiction à la tâche suivante via XCom
        kwargs['ti'].xcom_push(key='test_data_accuracy', value=test_data_accuracy)

    # Task 4: Prédiction avec le modèle enregistré
    def predict(**kwargs):
        # Récupérer les données et le modèle depuis les tâches précédentes via XCom
        ti = kwargs['ti']
        classifier = ti.xcom_pull(task_ids='train_model', key='classifier')

        # Exemple de données d'entrée
        input_data = [6, 130, 85, 32, 0, 0, 33.6, 0.627]
        
        # Transformer l'entrée en tableau numpy et standardiser
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        scaler = ti.xcom_pull(task_ids='prepare_data', key='scaler')
        std_data = scaler.transform(input_data_as_numpy_array)

        # Prédiction avec le modèle
        prediction = classifier.predict(std_data)
        if prediction[0] == 0:
            prediction_result = "The person is not diabetic"
        else:
            prediction_result = "The person is diabetic"

        # Passer la prédiction à la tâche suivante via XCom
        kwargs['ti'].xcom_push(key='prediction', value=prediction_result)

    # Définir l'ordre d'exécution des tâches
    start_task = EmptyOperator(task_id="start")
    end_task = EmptyOperator(task_id="end")

    # Créer les tâches Python avec PythonOperator
    prepare_data_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        provide_context=True,
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
        dag=dag
    )

    evaluate_and_log_model_task = PythonOperator(
        task_id='evaluate_and_log_model',
        python_callable=evaluate_and_log_model,
        provide_context=True,
        dag=dag
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict,
        provide_context=True,
        dag=dag
    )

    # Définir l'ordre des tâches
    start_task >> prepare_data_task >> train_model_task >> evaluate_and_log_model_task >> predict_task >> end_task
