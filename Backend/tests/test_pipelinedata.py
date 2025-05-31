import pytest
import pandas as pd
import numpy as np
from Backend.pipelinedata import prepare_data

@pytest.fixture
def sample_data(tmp_path):
    data = {
        "Pregnancies": [1, 2, None, 4],
        "Glucose": [85, 90, 95, 100],
        "BloodPressure": [66, None, 70, 80],
        "BMI": [26.6, 27.1, 28.0, 29.5],
        "Age": [22, 25, 30, 35],
        "Outcome": [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_diabetes.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_preprocess_handles_missing_values(sample_data):
    output_path = sample_data.parent / "processed_test.csv"
    preprocess_data(sample_data, output_path)
    processed_df = pd.read_csv(output_path)
    assert not processed_df.isnull().sum().sum(), "Les valeurs manquantes n'ont pas été gérées correctement"

def test_preprocess_normalization(sample_data):
    output_path = sample_data.parent / "processed_test.csv"
    preprocess_data(sample_data, output_path)
    processed_df = pd.read_csv(output_path)
    mean_glucose = processed_df["Glucose"].mean()
    std_glucose = processed_df["Glucose"].std()
    assert abs(mean_glucose) < 0.01, f"Moyenne de Glucose après normalisation devrait être proche de 0, mais est {mean_glucose}"
    assert abs(std_glucose - 1) < 0.01, f"Écart-type de Glucose après normalisation devrait être proche de 1, mais est {std_glucose}"

def test_preprocess_output_file_exists(sample_data):
    output_path = sample_data.parent / "processed_test.csv"
    preprocess_data(sample_data, output_path)
    assert output_path.exists(), "Le fichier de sortie n'a pas été créé"

def test_preprocess_target_column_preserved(sample_data):
    output_path = sample_data.parent / "processed_test.csv"
    preprocess_data(sample_data, output_path)
    processed_df = pd.read_csv(output_path)
    assert "Outcome" in processed_df.columns, "La colonne cible 'Outcome' n'est pas présente dans les données prétraitées"