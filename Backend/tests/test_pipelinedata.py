# test_pipelinedata.py
import pytest
import pandas as pd
import numpy as np
# Corrected import path based on the file location:
from models.pipelines.pipelinedata import preprocess_dataframe

@pytest.fixture
def sample_data():
    """
    Provides a sample DataFrame for testing preprocessing logic.
    It includes None values to test missing data handling.
    """
    data = {
        "Pregnancies": [1, 2, None, 4], # Contains None
        "Glucose": [85, 90, 95, 100],
        "BloodPressure": [66, None, 70, 80], # Contains None
        "SkinThickness": [20, 25, 30, 35], # Added for more realistic imputation test
        "Insulin": [0, 100, 150, 200], # Contains 0, which should be imputed
        "BMI": [26.6, 27.1, 28.0, 29.5],
        "Age": [22, 25, 30, 35],
        "Outcome": [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    return df

def test_preprocess_handles_missing_values(sample_data, tmp_path):
    """
    Tests if the preprocess_dataframe function correctly handles missing values
    (both None and 0s in specific columns) by imputing them.
    """
    # Call the preprocessing function with the sample DataFrame
    scaler, processed_df = preprocess_dataframe(sample_data)

    # Save the processed DataFrame to a temporary file for verification
    output_path = tmp_path / "processed_test_missing.csv"
    processed_df.to_csv(output_path, index=False)

    # Reload the DataFrame to ensure the file content is correct for assertion
    reloaded_df = pd.read_csv(output_path)

    # Assert that there are no missing values (NaNs) in the processed DataFrame
    assert not reloaded_df.isnull().sum().sum(), \
        "Missing values (NaNs) were not handled correctly in the processed DataFrame."

def test_preprocess_normalization(sample_data, tmp_path):
    """
    Tests if numerical features are correctly normalized (mean close to 0, std dev close to 1).
    """
    # Call the preprocessing function
    scaler, processed_df = preprocess_dataframe(sample_data)

    # Save the processed DataFrame to a temporary file for verification
    output_path = tmp_path / "processed_test_norm.csv"
    processed_df.to_csv(output_path, index=False)

    # Reload the DataFrame for assertion
    reloaded_df = pd.read_csv(output_path)

    # Identify feature columns (all except 'Outcome')
    features_to_check = [col for col in reloaded_df.columns if col != 'Outcome']

    for col in features_to_check:
        mean_val = reloaded_df[col].mean()
        # Calculate population standard deviation (ddof=0) to match StandardScaler's behavior
        std_val = reloaded_df[col].std(ddof=0) 
        
        # Assert mean is close to 0 (using a small tolerance for floating point precision)
        assert abs(mean_val) < 1e-9, \
            f"Mean of '{col}' after normalization should be close to 0, but is {mean_val}"
        
        # Assert standard deviation is close to 1 (using a small tolerance)
        assert abs(std_val - 1) < 1e-9, \
            f"Standard deviation of '{col}' after normalization should be close to 1, but is {std_val}"

def test_preprocess_output_file_can_be_created(sample_data, tmp_path):
    """
    Tests that the processed DataFrame can be successfully saved to a file.
    """
    # Call the preprocessing function
    scaler, processed_df = preprocess_dataframe(sample_data)
    
    # Define a temporary output path
    output_path = tmp_path / "processed_test_exists.csv"
    
    # Save the processed DataFrame to the temporary path
    processed_df.to_csv(output_path, index=False)
    
    # Assert that the output file now exists
    assert output_path.exists(), "The output file was not created by saving the processed DataFrame."

def test_preprocess_target_column_preserved(sample_data):
    """
    Tests if the 'Outcome' (target) column is preserved in the processed DataFrame
    and its values remain unchanged.
    """
    # Call the preprocessing function
    scaler, processed_df = preprocess_dataframe(sample_data)
    
    # Assert that the 'Outcome' column is present
    assert "Outcome" in processed_df.columns, \
        "The target column 'Outcome' is not present in the processed data."
    
    # Assert that the values in the 'Outcome' column are identical to the original
    pd.testing.assert_series_equal(
        sample_data['Outcome'], processed_df['Outcome'],
        check_dtype=False, # Allow for potential dtype changes if Pandas infers differently
        check_names=False  # Don't check series names
    )
