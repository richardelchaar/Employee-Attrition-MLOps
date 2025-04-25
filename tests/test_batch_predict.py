import pytest
from unittest import mock
import pandas as pd
try:
    from employee_attrition_mlops.config import (
        TARGET_COLUMN,
        SNAPSHOT_DATE_COL
    )
except ImportError:
     pytest.fail("Could not import config variables needed for tests. Check employee_attrition_mlops/config.py")

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        'EmployeeNumber': [101, 102, 103, 104, 105], # ID Column (removed by identify)
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
    with mock.patch('mlflow.pyfunc.load_model', return_value=mock_model):
        with mock.patch('scripts.batch_predict.create_engine', return_value=mock_engine):
            batch_predict.main()

            # Asserts
            mock_model.predict.assert_called_once()
            assert mock_engine.connect.called
            assert mock_engine.begin.called
