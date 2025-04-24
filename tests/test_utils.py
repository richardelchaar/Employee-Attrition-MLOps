# tests/test_utils.py
import pytest
import json
import joblib
import os
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

# Assuming your utils module is importable like this
# Adjust the import path if your project structure is different
from src.employee_attrition_mlops import utils

# --- Fixtures ---

@pytest.fixture
def sample_dict_data():
    """Provides sample dictionary data for JSON tests."""
    return {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}

@pytest.fixture
def sample_object_data():
    """Provides a sample Python object for joblib tests."""
    class SimpleObject:
        def __init__(self, name):
            self.name = name
        def __eq__(self, other):
            return isinstance(other, SimpleObject) and self.name == other.name
    return SimpleObject("test_object")

# --- Tests for JSON functions ---

def test_save_json_success(tmp_path, sample_dict_data):
    """Tests successful saving of data to a JSON file."""
    file_path = tmp_path / "subdir" / "test.json"
    utils.save_json(sample_dict_data, str(file_path))

    # Assert the directory and file were created
    assert file_path.parent.exists()
    assert file_path.exists()

    # Assert the content is correct
    with open(file_path, 'r') as f:
        loaded_data = json.load(f)
    assert loaded_data == sample_dict_data

@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('src.employee_attrition_mlops.utils.logger')
def test_save_json_exception(mock_logger, mock_makedirs, mock_open_file, sample_dict_data):
    """Tests error handling during JSON saving."""
    file_path = "/fake/path/test.json"
    # Simulate an exception during file writing
    mock_open_file.side_effect = IOError("Disk full")

    utils.save_json(sample_dict_data, file_path)

    # Assert os.makedirs was called correctly
    mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
    # Assert open was called correctly
    mock_open_file.assert_called_once_with(file_path, 'w')
    # Assert error was logged
    mock_logger.error.assert_called_once()
    assert "Error saving JSON" in mock_logger.error.call_args[0][0]
    assert "Disk full" in str(mock_logger.error.call_args[0][1])

def test_load_json_success(tmp_path, sample_dict_data):
    """Tests successful loading of data from a JSON file."""
    file_path = tmp_path / "test.json"
    # Create a dummy JSON file first
    with open(file_path, 'w') as f:
        json.dump(sample_dict_data, f)

    loaded_data = utils.load_json(str(file_path))
    assert loaded_data == sample_dict_data

def test_load_json_file_not_found(tmp_path):
    """Tests handling FileNotFoundError when loading JSON."""
    file_path = tmp_path / "non_existent.json"
    loaded_data = utils.load_json(str(file_path))
    assert loaded_data is None

@patch('builtins.open', new_callable=mock_open, read_data='invalid json')
@patch('src.employee_attrition_mlops.utils.logger')
def test_load_json_exception(mock_logger, mock_open_file):
    """Tests error handling for invalid JSON content."""
    file_path = "/fake/path/invalid.json"
    loaded_data = utils.load_json(file_path)

    mock_open_file.assert_called_once_with(file_path, 'r')
    assert loaded_data is None
    mock_logger.error.assert_called_once()
    assert "Error loading JSON" in mock_logger.error.call_args[0][0]

# --- Tests for Joblib functions ---

def test_save_object_success(tmp_path, sample_object_data):
    """Tests successful saving of a Python object using joblib."""
    file_path = tmp_path / "subdir" / "test.joblib"
    utils.save_object(sample_object_data, str(file_path))

    # Assert the directory and file were created
    assert file_path.parent.exists()
    assert file_path.exists()

    # Assert the content can be loaded correctly (optional, but good check)
    loaded_obj = joblib.load(file_path)
    assert loaded_obj == sample_object_data # Requires __eq__ method in SimpleObject

@patch('joblib.dump')
@patch('os.makedirs')
@patch('src.employee_attrition_mlops.utils.logger')
def test_save_object_exception(mock_logger, mock_makedirs, mock_dump, sample_object_data):
    """Tests error handling during object saving."""
    file_path = "/fake/path/test.joblib"
    # Simulate an exception during joblib dump
    mock_dump.side_effect = IOError("Cannot write")

    utils.save_object(sample_object_data, file_path)

    mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
    mock_dump.assert_called_once_with(sample_object_data, file_path)
    mock_logger.error.assert_called_once()
    assert "Error saving object" in mock_logger.error.call_args[0][0]
    assert "Cannot write" in str(mock_logger.error.call_args[0][1])

def test_load_object_success(tmp_path, sample_object_data):
    """Tests successful loading of a Python object using joblib."""
    file_path = tmp_path / "test.joblib"
    # Create a dummy joblib file first
    joblib.dump(sample_object_data, file_path)

    loaded_obj = utils.load_object(str(file_path))
    assert loaded_obj == sample_object_data # Requires __eq__ method in SimpleObject

def test_load_object_file_not_found(tmp_path):
    """Tests handling FileNotFoundError when loading an object."""
    file_path = tmp_path / "non_existent.joblib"
    loaded_obj = utils.load_object(str(file_path))
    assert loaded_obj is None

@patch('joblib.load')
@patch('src.employee_attrition_mlops.utils.logger')
def test_load_object_exception(mock_logger, mock_load):
    """Tests error handling during object loading (e.g., corrupted file)."""
    file_path = "/fake/path/corrupted.joblib"
    mock_load.side_effect = EOFError("Unexpected end of file")

    loaded_obj = utils.load_object(file_path)

    mock_load.assert_called_once_with(file_path)
    assert loaded_obj is None
    mock_logger.error.assert_called_once()
    assert "Error loading object" in mock_logger.error.call_args[0][0]
    assert "Unexpected end of file" in str(mock_logger.error.call_args[0][1])


# --- Tests for MLflow functions ---

@patch('src.employee_attrition_mlops.utils.MlflowClient')
def test_get_production_model_run_id_success(MockMlflowClient):
    """Tests successfully getting a production model run_id."""
    mock_client_instance = MockMlflowClient.return_value
    mock_version = MagicMock()
    mock_version.run_id = "test_run_id_123"
    mock_client_instance.get_latest_versions.return_value = [mock_version]

    model_name = "my_model"
    stage = "Production"
    run_id = utils.get_production_model_run_id(model_name, stage)

    assert run_id == "test_run_id_123"
    mock_client_instance.get_latest_versions.assert_called_once_with(model_name, stages=[stage])

@patch('src.employee_attrition_mlops.utils.MlflowClient')
@patch('src.employee_attrition_mlops.utils.logger')
def test_get_production_model_run_id_not_found(mock_logger, MockMlflowClient):
    """Tests the case where no model version is found for the stage."""
    mock_client_instance = MockMlflowClient.return_value
    mock_client_instance.get_latest_versions.return_value = [] # Simulate no versions found

    model_name = "my_model"
    stage = "Staging"
    run_id = utils.get_production_model_run_id(model_name, stage)

    assert run_id is None
    mock_client_instance.get_latest_versions.assert_called_once_with(model_name, stages=[stage])
    mock_logger.warning.assert_called_once()
    assert "No model version found" in mock_logger.warning.call_args[0][0]

@patch('src.employee_attrition_mlops.utils.MlflowClient')
@patch('src.employee_attrition_mlops.utils.logger')
def test_get_production_model_run_id_exception(mock_logger, MockMlflowClient):
    """Tests error handling during MLflow client interaction."""
    mock_client_instance = MockMlflowClient.return_value
    mock_client_instance.get_latest_versions.side_effect = Exception("MLflow connection error")

    model_name = "my_model"
    stage = "Production"
    run_id = utils.get_production_model_run_id(model_name, stage)

    assert run_id is None
    mock_client_instance.get_latest_versions.assert_called_once_with(model_name, stages=[stage])
    mock_logger.error.assert_called_once()
    assert "Error fetching production model run_id" in mock_logger.error.call_args[0][0]
    assert "MLflow connection error" in str(mock_logger.error.call_args[0][1])


@patch('src.employee_attrition_mlops.utils.MlflowClient')
def test_download_mlflow_artifact_success(MockMlflowClient):
    """Tests successful download of an MLflow artifact."""
    mock_client_instance = MockMlflowClient.return_value
    expected_local_path = "/tmp/downloaded/artifact.pkl"
    mock_client_instance.download_artifacts.return_value = expected_local_path

    run_id = "test_run_id_456"
    artifact_path = "models/model.pkl"
    dst_path = "/tmp/downloaded" # Optional destination path

    local_path = utils.download_mlflow_artifact(run_id, artifact_path, dst_path)

    assert local_path == expected_local_path
    mock_client_instance.download_artifacts.assert_called_once_with(run_id, artifact_path, dst_path)

@patch('src.employee_attrition_mlops.utils.MlflowClient')
def test_download_mlflow_artifact_success_no_dst(MockMlflowClient):
    """Tests successful download without specifying a destination path."""
    mock_client_instance = MockMlflowClient.return_value
    expected_local_path = "./mlruns_download/artifact.pkl" # Example default path
    mock_client_instance.download_artifacts.return_value = expected_local_path

    run_id = "test_run_id_789"
    artifact_path = "data/results.csv"

    local_path = utils.download_mlflow_artifact(run_id, artifact_path) # dst_path=None

    assert local_path == expected_local_path
    # When dst_path is None, the third argument to download_artifacts is None
    mock_client_instance.download_artifacts.assert_called_once_with(run_id, artifact_path, None)


@patch('src.employee_attrition_mlops.utils.MlflowClient')
@patch('src.employee_attrition_mlops.utils.logger')
def test_download_mlflow_artifact_exception(mock_logger, MockMlflowClient):
    """Tests error handling during artifact download."""
    mock_client_instance = MockMlflowClient.return_value
    mock_client_instance.download_artifacts.side_effect = Exception("Artifact not found")

    run_id = "test_run_id_abc"
    artifact_path = "non_existent/artifact"

    local_path = utils.download_mlflow_artifact(run_id, artifact_path)

    assert local_path is None
    mock_client_instance.download_artifacts.assert_called_once_with(run_id, artifact_path, None)
    mock_logger.error.assert_called_once()
    assert "Failed to download artifact" in mock_logger.error.call_args[0][0]
    assert "Artifact not found" in str(mock_logger.error.call_args[0][1])

