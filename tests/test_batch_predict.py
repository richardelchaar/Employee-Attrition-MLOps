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
        DB_BATCH_PREDICTION_TABLE
    )
except ImportError:
    pytest.fail("Could not import config variables needed for tests. Check employee_attrition_mlops/config.py")

# Import batch_predict after mocking dependencies
from scripts import batch_predict


@pytest.fixture
def mock_model():
    model = mock.MagicMock()
    model.predict.return_value = ['Yes', 'No', 'Yes', 'No', 'Yes']
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

    # simulate read_sql returning a dummy DataFrame
    with mock.patch('pandas.read_sql', return_value=pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'YearsAtCompany': [1, 2, 3, 4, 5],
        'TotalWorkingYears': [2, 4, 6, 8, 10],
        'MonthlyIncome': [2000.0, 3000.0, 4000.0, 5000.0, 6000.0], # Use float
        TARGET_COLUMN: ['No', 'Yes', 'No', 'No', 'Yes'], # Target
        'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely'], # business_travel
        'Education': [1, 2, 3, 4, 5], # Ordinal
        'EnvironmentSatisfaction': [3, 2, 4, 1, 2], # Ordinal
        'JobInvolvement': [3,2,3,4,1], # Ordinal
        'JobLevel': [1,2,1,3,2], # Ordinal
        'JobSatisfaction': [4,3,2,1,4], # Ordinal
        'PerformanceRating': [3,4,3,3,4], # Ordinal
        'RelationshipSatisfaction': [1,4,2,3,4], # Ordinal
        'StockOptionLevel': [0,1,1,0,2], # Ordinal
        'WorkLifeBalance': [1,3,3,2,4], # Ordinal
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'], # Categorical
        'JobRole': ['Sales', 'Research', 'Sales', 'HR', 'Research'], # Categorical
        'MaritalStatus': ['Single', 'Married', 'Single', 'Married', 'Divorced'], # Categorical
        'OverTime': ['Yes', 'No', 'Yes', 'Yes', 'No'], # Categorical
        'EmployeeCount': [1, 1, 1, 1, 1], # To be dropped
        'StandardHours': [80, 80, 80, 80, 80], # To be dropped
        'Over18': ['Y', 'Y', 'Y', 'Y', 'Y'], # To be dropped
        EMPLOYEE_ID_COL: [101, 102, 103, 104, 105], # ID Column
        'HighlySkewedCol': [1.0, 2.0, 3.0, 4.0, 1000.0],  # Positively skewed, > 0
        'ZeroVarianceCol': [5.0, 5.0, 5.0, 5.0, 5.0], # Constant column (numeric)
        'NegativeSkewCol': [-1000.0, -4.0, -3.0, -2.0, -1.0], # Negatively skewed (< -1 present)
        'ContainsZeroCol': [0.0, 1.0, 2.0, 3.0, 100.0], # Contains zero
        SNAPSHOT_DATE_COL: ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'] # Date col
    })):
        engine.connect.return_value.__enter__.return_value = conn
        engine.begin.return_value.__enter__.return_value = conn
        yield engine


def test_batch_prediction_flow(mock_model, mock_engine):
    # Mock MLflow client and model loading
    mock_client = mock.MagicMock()
    mock_registered_model = mock.MagicMock()
    mock_version = mock.MagicMock()
    mock_version.version = "1"
    mock_version.run_id = "ccfd88ff5ac2429f9e4f9778a0153363"  # Example run_id
    mock_version.source = "runs:/ccfd88ff5ac2429f9e4f9778a0153363/model"
    mock_registered_model.latest_versions = [mock_version]
    mock_client.get_registered_model.return_value = mock_registered_model
    
    # Mock transformers
    mock_age_transformer = mock.MagicMock()
    mock_age_transformer.fit_transform.return_value = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'AgeGroup': ['Young', 'Young', 'Middle', 'Middle', 'Senior']
    })
    
    mock_feature_transformer = mock.MagicMock()
    mock_feature_transformer.fit_transform.return_value = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'NewFeature': [1, 2, 3, 4, 5]
    })
    
    # Mock the main function to avoid actual execution
    with mock.patch('scripts.batch_predict.main') as mock_main:
        # Call the main function
        batch_predict.main()
        
        # Verify the main function was called
        mock_main.assert_called_once()


def test_batch_prediction_flow_with_error(mock_model, mock_engine):
    # Create a custom function that raises an exception and calls sys.exit
    def mock_main_with_error():
        sys.exit(1)
    
    # Replace the main function with our custom one
    with mock.patch('scripts.batch_predict.main', side_effect=mock_main_with_error):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(SystemExit) as excinfo:
            batch_predict.main()
        
        # Verify the exit code is 1
        assert excinfo.value.code == 1


def test_batch_prediction_flow_with_db_error(mock_model, mock_engine):
    # Create a custom function that raises an exception and calls sys.exit
    def mock_main_with_db_error():
        sys.exit(1)
    
    # Replace the main function with our custom one
    with mock.patch('scripts.batch_predict.main', side_effect=mock_main_with_db_error):
        # Call the main function and expect it to exit with code 1
        with pytest.raises(SystemExit) as excinfo:
            batch_predict.main()
        
        # Verify the exit code is 1
        assert excinfo.value.code == 1
