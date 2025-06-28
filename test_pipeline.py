# test_pipeline.py
import pytest
import pandas as pd
import joblib
import json
import os
import subprocess

@pytest.fixture(scope="module")
def raw_iris_data_from_csv():
    """Provides the raw IRIS dataset loaded from CSV for data validation tests."""
    csv_path = os.path.join('data', 'iris.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        pytest.fail(f"Error: '{csv_path}' not found. Ensure it's in the 'data' directory for testing.")
    
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    return X, y

@pytest.fixture(scope="module")
def trained_model_and_metrics():
    """
    Runs train.py to produce a model and metrics, then loads them.
    Ensures a fresh model and metrics for testing.
    """
    # Clean up any existing artifacts before running train.py
    if os.path.exists('iris_model.pkl'):
        os.remove('iris_model.pkl')
    if os.path.exists('metrics.json'):
        os.remove('metrics.json')

    print("\nRunning train.py to generate model and metrics for tests...")
    # Run train.py as a subprocess
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True, check=False)
    print("train.py stdout:\n", result.stdout)
    print("train.py stderr:\n", result.stderr)

    if result.returncode != 0:
        pytest.fail(f"train.py failed with exit code {result.returncode}. Error:\n{result.stderr}")

    # Load the generated model and metrics
    try:
        model = joblib.load('iris_model.pkl')
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError as e:
        pytest.fail(f"Could not find generated files (iris_model.pkl or metrics.json): {e}")

    yield model, metrics

# --- Data Validation Tests ---

def test_data_has_expected_columns(raw_iris_data_from_csv):
    X, _ = raw_iris_data_from_csv
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    assert list(X.columns) == expected_columns, "Data does not have expected columns."

def test_data_has_no_missing_values(raw_iris_data_from_csv):
    X, y = raw_iris_data_from_csv
    assert not X.isnull().any().any(), "Features contain missing values."
    assert not y.isnull().any(), "Target contains missing values."

def test_target_values_are_valid(raw_iris_data_from_csv):
    _, y = raw_iris_data_from_csv
    expected_species = {'setosa', 'versicolor', 'virginica'}
    assert set(y.unique()).issubset(expected_species), "Target values are not valid (expected setosa, versicolor, or virginica)."

# --- Model Evaluation Tests ---

def test_model_accuracy_above_threshold(trained_model_and_metrics):
    model, metrics = trained_model_and_metrics
    accuracy = metrics.get("accuracy")
    assert accuracy is not None, "Accuracy not found in metrics."
    assert accuracy >= 0.80, f"Model accuracy {accuracy:.4f} is below threshold (0.80)."

def test_metrics_json_structure(trained_model_and_metrics):
    _, metrics = trained_model_and_metrics
    expected_keys = ["accuracy", "precision", "recall", "f1_score"]
    assert all(key in metrics for key in expected_keys), "Metrics.json is missing expected keys."
    assert all(isinstance(metrics[key], (float, int)) for key in expected_keys), "Metrics values are not numeric."