# pipelinedata.py
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
# Removed: from airflow.configuration import conf
from mlflow.models import infer_signature

# Removed: Activer le pickling des objets dans XCom
# Removed: conf.set('core', 'enable_xcom_pickling', 'True')
# This configuration is typically managed in airflow.cfg or via environment variables
# (e.g., AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True) when running Airflow.
# It should not be at the module level if the module is imported for testing.

# Définir le propriétaire du DAG et les arguments par défaut
dag_owner = 'chaimae'
default_args = {
    'owner': dag_owner,
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5)
}

# --- Core Data Preprocessing Function (Testable Unit) ---
def preprocess_dataframe(df: pd.DataFrame):
    """
    Handles missing values and standardizes features in a pandas DataFrame.
    This function is designed to be reusable and testable independently of Airflow.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
        tuple: A tuple containing:
            - StandardScaler: The fitted StandardScaler object.
            - pd.DataFrame: The DataFrame with processed features and original target.
    """
    # Make a copy to avoid modifying the original DataFrame passed in
    processed_df = df.copy()

    # Define columns that might contain 0s or NaNs that need imputation
    # These are typically features where 0 is not a valid physiological value.
    columns_to_impute_with_mean = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for column in columns_to_impute_with_mean:
        if column in processed_df.columns:
            # Replace 0s with NaN first, then fill NaNs with the mean of the column.
            # This handles both explicit NaNs/None and 0s that represent missing data.
            processed_df[column] = processed_df[column].replace(0, np.nan)
            processed_df[column] = processed_df[column].fillna(processed_df[column].mean())

    # Separate features (X) and target (Y)
    # Ensure 'Outcome' column exists before attempting to drop it
    if 'Outcome' in processed_df.columns:
        X = processed_df.drop(columns='Outcome', axis=1)
        Y = processed_df['Outcome']
    else:
        # If 'Outcome' is not present, assume all columns are features
        X = processed_df
        Y = pd.Series([], dtype='int64') # Create an empty Series for Y if no outcome

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled features back to a DataFrame to maintain column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Combine processed features with the original target column (if it existed)
    if 'Outcome' in processed_df.columns:
        final_df = pd.concat([X_scaled_df, Y], axis=1)
    else:
        final_df = X_scaled_df # If no Outcome, the final_df is just the scaled features

    return scaler, final_df

# --- Airflow DAG Definition ---
with DAG(
    dag_id='diabetes_prediction_dag',
    default_args=default_args,
    description='A DAG for training and monitoring the diabetes prediction model',
    start_date=datetime(2025, 4, 15),
    schedule_interval='@weekly',
    catchup=False,
    tags=['diabetes', 'ml']
) as dag:

    # Task 1: Prepare Data (Airflow Callable)
    def prepare_data(**kwargs):
        """
        Airflow callable for the 'prepare_data' task.
        Loads the raw dataset, preprocesses it using `preprocess_dataframe`,
        splits it into training/testing sets, and pushes results to Airflow XCom.
        """
        # Load the raw dataset from the specified path
        # This path is relative to the Airflow worker's execution environment.
        diabetes_dataset = pd.read_csv('./data/diabetes.csv')

        # Use the reusable preprocessing function
        scaler, processed_df = preprocess_dataframe(diabetes_dataset)

        # Separate features (X) and target (Y) from the processed DataFrame
        # Convert to numpy arrays as expected by scikit-learn models
        X_scaled = processed_df.drop(columns='Outcome', axis=1).values
        Y = processed_df['Outcome'].values

        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
        )

        # Push the prepared data and the fitted scaler to XCom for downstream tasks
        ti = kwargs['ti']
        ti.xcom_push(key='X_train', value=X_train)
        ti.xcom_push(key='X_test', value=X_test)
        ti.xcom_push(key='Y_train', value=Y_train)
        ti.xcom_push(key='Y_test', value=Y_test)
        ti.xcom_push(key='scaler', value=scaler) # Push the scaler for prediction task

    # Task 2: Train the Model
    def train_model(**kwargs):
        """
        Airflow callable for the 'train_model' task.
        Pulls training data from XCom, initializes and trains an SVM classifier,
        and saves the trained model using joblib.
        """
        # Retrieve training data from XCom
        ti = kwargs['ti']
        X_train = ti.xcom_pull(task_ids='prepare_data', key='X_train')
        Y_train = ti.xcom_pull(task_ids='prepare_data', key='Y_train')

        # Initialize the SVM classifier
        classifier = svm.SVC(kernel='linear')

        # Train the model
        classifier.fit(X_train, Y_train)

        # Save the trained model using joblib
        # This path assumes a 'data' directory exists where the model will be stored
        joblib.dump(classifier, './data/diabetes_model.pkl')

        # Push the trained classifier to XCom for the next task
        kwargs['ti'].xcom_push(key='classifier', value=classifier)

    # Task 3: Evaluate and Log Model Performance with MLflow
    def evaluate_and_log_model(**kwargs):
        """
        Airflow callable for the 'evaluate_and_log_model' task.
        Pulls data and the trained model from XCom, evaluates its performance,
        and logs metrics and the model to MLflow.
        """
        # Retrieve data and classifier from XCom
        ti = kwargs['ti']
        X_train = ti.xcom_pull(task_ids='prepare_data', key='X_train')
        Y_train = ti.xcom_pull(task_ids='prepare_data', key='Y_train')
        X_test = ti.xcom_pull(task_ids='prepare_data', key='X_test')
        Y_test = ti.xcom_pull(task_ids='prepare_data', key='Y_test')
        classifier = ti.xcom_pull(task_ids='train_model', key='classifier')

        # Make predictions on training and test data
        X_train_prediction = classifier.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

        X_test_prediction = classifier.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

        print(f'Accuracy on training data: {training_data_accuracy}')
        print(f'Accuracy on test data: {test_data_accuracy}')

        # Initialize MLflow tracking
        # Set the MLflow tracking server URI (e.g., a local server or remote)
        mlflow.set_tracking_uri("http://mlflow:5000")

        # Create or set an MLflow Experiment
        mlflow.set_experiment("MLflow Quickstart")

        # Start an MLflow run to log artifacts
        with mlflow.start_run():
            # Log the accuracy metric
            mlflow.log_metric("accuracy", test_data_accuracy)

            # Set a tag to categorize or describe the run
            mlflow.set_tag("Training Info", "Basic LR model for iris data")

            # Infer the model signature for better model serving/deployment
            signature = infer_signature(X_train, classifier.predict(X_train))

            # Log the scikit-learn model to MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=classifier,
                artifact_path="iris_model", # Path within the MLflow run artifact store
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-quickstart", # Register the model in MLflow Model Registry
            )

        # Push the test data accuracy to XCom for potential future use
        kwargs['ti'].xcom_push(key='test_data_accuracy', value=test_data_accuracy)

    # Task 4: Make Prediction with the Trained Model
    def predict(**kwargs):
        """
        Airflow callable for the 'predict' task.
        Pulls the trained model and scaler from XCom, takes a sample input,
        preprocesses it, makes a prediction, and pushes the result to XCom.
        """
        # Retrieve classifier and scaler from XCom
        ti = kwargs['ti']
        classifier = ti.xcom_pull(task_ids='train_model', key='classifier')
        scaler = ti.xcom_pull(task_ids='prepare_data', key='scaler')

        # Example input data for prediction
        # This should match the expected features order after preprocessing
        input_data = [6, 130, 85, 32, 0, 0, 33.6, 0.627] # Example values

        # Convert input data to a numpy array and reshape for single sample prediction
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Standardize the input data using the previously fitted scaler
        std_data = scaler.transform(input_data_as_numpy_array)

        # Make a prediction using the trained classifier
        prediction = classifier.predict(std_data)

        # Determine the prediction result message
        if prediction[0] == 0:
            prediction_result = "The person is not diabetic"
        else:
            prediction_result = "The person is diabetic"

        # Push the prediction result to XCom
        kwargs['ti'].xcom_push(key='prediction', value=prediction_result)

    # Define the task dependencies and execution order within the DAG
    start_task = EmptyOperator(task_id="start")
    end_task = EmptyOperator(task_id="end")

    # Create PythonOperator tasks for each step
    prepare_data_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data, # This now calls the Airflow callable
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

    # Set the task execution order
    start_task >> prepare_data_task >> train_model_task >> evaluate_and_log_model_task >> predict_task >> end_task
