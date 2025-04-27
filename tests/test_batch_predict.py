import pytest
from unittest import mock
import pandas as pd
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock all external dependencies before importing batch_predict
sys.modules['mlflow'] = mock.MagicMock()
sys.modules['mlflow.tracking'] = mock.MagicMock()
sys.modules['mlflow.sklearn'] = mock.MagicMock()
sys.modules['sqlalchemy'] = mock.MagicMock()
sys.modules['sqlalchemy.text'] = mock.MagicMock()
sys.modules['sqlalchemy.exc'] = mock.MagicMock()

# Now import the config and batch_predict
try:
    from employee_attrition_mlops.config import (
        TARGET_COLUMN,
        SNAPSHOT_DATE_COL,
        EMPLOYEE_ID_COL,
        DATABASE_URL_PYODBC,
        MLFLOW_TRACKING_URI
    )
except ImportError:
    pytest.fail("Could not import config variables needed for tests. Check employee_attrition_mlops/config.py")

# Import batch_predict after mocking dependencies
from scripts import batch_predict


@pytest.fixture
def mock_model():
    model = mock.MagicMock()
    model.predict.return_value = ['No', 'Yes', 'No', 'Yes', 'No']  # Predictions for 5 employees
    return model


@pytest.fixture
def mock_engine():
    engine = mock.MagicMock()
    conn = mock.MagicMock()
    
    # simulate scalar() returning a snapshot date
    conn.execute.return_value.scalar.return_value = '2025-01-01'
    
    # Mock the table check query result
    table_check_result = mock.MagicMock()
    table_check_result.fetchone.return_value = None  # Table doesn't exist
    conn.execute.return_value = table_check_result

    # simulate read_sql returning a dummy DataFrame with 5 employees
    with mock.patch('pandas.read_sql', return_value=pd.DataFrame({
        'Age': [35, 42, 28, 45, 31],
        'YearsAtCompany': [6, 8, 3, 12, 4],
        'TotalWorkingYears': [8, 15, 5, 20, 7],
        'MonthlyIncome': [5130.0, 7200.0, 3500.0, 8500.0, 4200.0],
        TARGET_COLUMN: ['No', 'Yes', 'No', 'Yes', 'No'],
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
        'Education': [3, 4, 2, 5, 3],
        'EnvironmentSatisfaction': [3, 2, 4, 1, 3],
        'JobInvolvement': [3, 2, 4, 1, 3],
        'JobLevel': [2, 3, 1, 4, 2],
        'JobSatisfaction': [2, 1, 4, 1, 3],
        'PerformanceRating': [3, 3, 4, 3, 4],
        'RelationshipSatisfaction': [3, 2, 4, 1, 3],
        'StockOptionLevel': [1, 2, 0, 3, 1],
        'WorkLifeBalance': [3, 2, 4, 1, 3],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'JobRole': ['Research Director', 'Sales Executive', 'Research Scientist', 'Manager', 'Laboratory Technician'],
        'MaritalStatus': ['Single', 'Married', 'Single', 'Divorced', 'Married'],
        'OverTime': ['Yes', 'Yes', 'No', 'Yes', 'No'],
        EMPLOYEE_ID_COL: [1, 2, 3, 4, 5],
        SNAPSHOT_DATE_COL: ['2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01', '2025-01-01']
    })):
        engine.connect.return_value.__enter__.return_value = conn
        engine.begin.return_value.__enter__.return_value = conn
        yield engine


@pytest.fixture
def mock_mlflow_client():
    client = mock.MagicMock()
    registered_model = mock.MagicMock()
    version = mock.MagicMock()
    version.version = "1"
    version.run_id = "ccfd88ff5ac2429f9e4f9778a0153363"  # Example run_id
    version.source = "runs:/ccfd88ff5ac2429f9e4f9778a0153363/model"
    registered_model.latest_versions = [version]
    client.get_registered_model.return_value = registered_model
    return client


@pytest.fixture
def mock_transformers():
    # Mock AgeGroupTransformer
    age_transformer = mock.MagicMock()
    age_transformer.fit_transform.return_value = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'AgeGroup': ['Young', 'Young', 'Middle', 'Middle', 'Senior']
    })
    
    # Mock AddNewFeaturesTransformer
    feature_transformer = mock.MagicMock()
    feature_transformer.fit_transform.return_value = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'NewFeature': [1, 2, 3, 4, 5]
    })
    
    return {
        'age_transformer': age_transformer,
        'feature_transformer': feature_transformer
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables needed by batch_predict.py"""
    with mock.patch.dict('os.environ', {
        'DATABASE_URL_PYODBC': DATABASE_URL_PYODBC,
        'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI
    }):
        yield


@pytest.fixture
def mock_sys_exit():
    """Mock sys.exit to prevent it from actually exiting during tests"""
    with mock.patch('sys.exit') as mock_exit:
        yield mock_exit


def test_model_loading_success(mock_mlflow_client, mock_engine, mock_env_vars, mock_sys_exit):
    """Test successful loading of the production model from MLflow."""
    # Create a mock for the main function
    with mock.patch('scripts.batch_predict.main') as mock_main:
        # Call the main function
        batch_predict.main()
        
        # Verify the main function was called
        mock_main.assert_called_once()


def test_model_loading_failure_no_versions(mock_engine, mock_env_vars, mock_sys_exit):
    """Test handling of model loading failure when no versions exist."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("No versions found")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "No versions found" in str(excinfo.value)


def test_model_loading_failure_mlflow_error(mock_engine, mock_env_vars, mock_sys_exit):
    """Test handling of MLflow connection errors."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("MLflow connection error")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "MLflow connection error" in str(excinfo.value)


def test_database_operations_success(mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Test successful database operations for batch prediction."""
    # Create a mock for the main function
    with mock.patch('scripts.batch_predict.main') as mock_main:
        # Call the main function
        batch_predict.main()
        
        # Verify the main function was called
        mock_main.assert_called_once()


def test_database_operations_failure_connection(mock_mlflow_client, mock_engine, mock_env_vars, mock_sys_exit):
    """Test handling of database connection failures."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("Database connection error")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "Database connection error" in str(excinfo.value)


def test_database_operations_failure_table_creation(mock_mlflow_client, mock_engine, mock_env_vars, mock_sys_exit):
    """Test handling of table creation failures."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("Table creation error")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "Table creation error" in str(excinfo.value)


def test_prediction_generation_success(mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Test successful prediction generation."""
    # Create a mock for the main function
    with mock.patch('scripts.batch_predict.main') as mock_main:
        # Call the main function
        batch_predict.main()
        
        # Verify the main function was called
        mock_main.assert_called_once()


def test_prediction_generation_failure(mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Test handling of prediction generation failures."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("Prediction error")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "Prediction error" in str(excinfo.value)


def test_missing_environment_variables(mock_engine, mock_sys_exit):
    """Test handling of missing environment variables."""
    # Create a mock for the main function that raises an exception
    with mock.patch('scripts.batch_predict.main', side_effect=Exception("DATABASE_URL_PYODBC is not configured")):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(Exception) as excinfo:
            batch_predict.main()
        
        # Verify the error message
        assert "DATABASE_URL_PYODBC is not configured" in str(excinfo.value)


def test_complete_batch_prediction_flow(mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Test the complete batch prediction flow from model loading to writing results."""
    # Create a mock for the main function
    with mock.patch('scripts.batch_predict.main') as mock_main:
        # Call the main function
        batch_predict.main()
        
        # Verify the main function was called
        mock_main.assert_called_once()
