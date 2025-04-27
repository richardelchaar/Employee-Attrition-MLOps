#!/usr/bin/env python
# tests/test_production_automation.py
import os
import sys
import logging
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, Mock
import argparse

# Mock Evidently imports before they're used
sys.modules['evidently'] = Mock()
sys.modules['evidently.report'] = Mock()
sys.modules['evidently.metric_preset'] = Mock()
sys.modules['evidently.metrics'] = Mock()

# Add src to Python path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Import fixtures from test_batch_predict
from tests.test_batch_predict import (
    mock_mlflow_client,
    mock_engine,
    mock_transformers,
    mock_env_vars,
    mock_sys_exit
)

from employee_attrition_mlops.data_processing import load_and_clean_data
from employee_attrition_mlops.config import (
    TARGET_COLUMN, DB_HISTORY_TABLE, DATABASE_URL_PYMSSQL,
    MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_production_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_data():
    """Create synthetic test data."""
    np.random.seed(42)
    
    data = {
        'EmployeeNumber': range(1000, 1100),
        'Age': np.random.randint(18, 65, 100),
        'Gender': np.random.choice(['Female', 'Male'], 100),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], 100),
        'Department': np.random.choice(['Research & Development', 'Sales', 'Human Resources'], 100),
        'EducationField': np.random.choice(['Medical', 'Life Sciences', 'Technical'], 100),
        'JobLevel': np.random.randint(1, 5, 100),
        'JobRole': np.random.choice(['Research Scientist', 'Sales Executive', 'Manager', 'Laboratory Technician'], 100),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 100),
        'DistanceFromHome': np.random.randint(1, 30, 100),
        'Education': np.random.randint(1, 6, 100),
        'DailyRate': np.random.randint(800, 1200, 100),
        'HourlyRate': np.random.randint(75, 100, 100),
        'MonthlyIncome': np.random.randint(3500, 9500, 100),
        'MonthlyRate': np.random.randint(18000, 22000, 100),
        'PercentSalaryHike': np.random.randint(10, 15, 100),
        'StockOptionLevel': np.random.randint(0, 4, 100),
        'OverTime': np.random.choice(['Yes', 'No'], 100),
        'NumCompaniesWorked': np.random.randint(0, 10, 100),
        'TotalWorkingYears': np.random.randint(0, 40, 100),
        'TrainingTimesLastYear': np.random.randint(0, 6, 100),
        'YearsAtCompany': np.random.randint(0, 40, 100),
        'YearsInCurrentRole': np.random.randint(0, 40, 100),
        'YearsSinceLastPromotion': np.random.randint(0, 40, 100),
        'YearsWithCurrManager': np.random.randint(0, 40, 100),
        'EnvironmentSatisfaction': np.random.randint(1, 5, 100),
        'JobInvolvement': np.random.randint(1, 5, 100),
        'JobSatisfaction': np.random.randint(1, 5, 100),
        'PerformanceRating': np.random.randint(1, 5, 100),
        'RelationshipSatisfaction': np.random.randint(1, 5, 100),
        'WorkLifeBalance': np.random.randint(1, 5, 100),
        'SnapshotDate': [datetime.now().strftime('%Y-%m-%d')] * 100,
        TARGET_COLUMN: np.random.choice(['Yes', 'No'], 100, p=[0.2, 0.8])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def drifted_data(test_data):
    """Create drifted test data."""
    drifted_data = test_data.copy()
    
    # Select numeric columns for drift
    numeric_cols = drifted_data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['EmployeeNumber', 'SnapshotDate']]
    
    if len(numeric_cols) == 0:
        # If no numeric columns, create some drift in categorical columns
        categorical_cols = drifted_data.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:  # Apply drift to first 2 categorical columns
            drifted_data[col] = drifted_data[col].map(lambda x: f"{x}_drifted")
    else:
        # Apply drift to random subset of numeric columns
        n_cols_to_drift = max(1, len(numeric_cols) // 3)
        cols_to_drift = np.random.choice(numeric_cols, n_cols_to_drift, replace=False)
        
        for col in cols_to_drift:
            # Apply drift by multiplying values
            drifted_data[col] = drifted_data[col] * 2.0
    
    return drifted_data

@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_artifact') as mock_log_artifact:
        
        # Setup mock context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        yield {
            'start_run': mock_start_run,
            'log_metric': mock_log_metric,
            'log_artifact': mock_log_artifact,
            'run': mock_run
        }

def test_normal_run(test_data, mock_mlflow, tmp_path):
    """Test normal run of production automation."""
    # Save test data to temporary CSV
    test_data_path = tmp_path / "test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    
    # Mock the drift detection results
    mock_drift_results = {
        'dataset_drift': False,
        'drift_share': 0.0,
        'drifted_features': [],
        'n_drifted_features': 0,
        'report': {
            'metrics': [
                {'result': {'dataset_drift': False}},
                {'result': {'drift_by_columns': {}}}
            ]
        }
    }
    
    # Mock the prediction results
    mock_prediction_results = {
        'predictions': [0] * 100,  # 100 predictions
        'attrition_rate': 0.2
    }
    
    # Create a mock reference data DataFrame
    mock_reference_data = test_data.copy()
    
    # Patch the necessary functions
    with patch('employee_attrition_mlops.drift_detection.check_drift', return_value=mock_drift_results), \
         patch('employee_attrition_mlops.drift_detection.should_trigger_retraining', return_value=False), \
         patch('scripts.batch_predict.main', return_value=mock_prediction_results), \
         patch('employee_attrition_mlops.drift_detection.Report', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftPreset', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftTable', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DatasetDriftMetric', Mock()), \
         patch('employee_attrition_mlops.drift_detection.get_production_model_run_id', return_value='test-run-id'), \
         patch('employee_attrition_mlops.drift_detection.get_baseline_artifacts', return_value=('profile.json', 'reference.parquet', 'features.json')), \
         patch('pandas.read_parquet', return_value=mock_reference_data):
        
        # Run production automation
        from scripts.run_production_automation import run_production_automation
        run_production_automation(argparse.Namespace(csv_path=str(test_data_path), force_retrain=False))
        
        # Verify MLflow logging
        mock_mlflow['log_metric'].assert_any_call("num_predictions", 100)
        mock_mlflow['log_metric'].assert_any_call("attrition_rate", 0.2)
        mock_mlflow['log_metric'].assert_any_call("was_retrained", 0)

def test_drift_detection(drifted_data, mock_mlflow, tmp_path, mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Test drift detection in production automation."""
    # Save drifted data to temporary CSV
    drifted_data_path = tmp_path / "drifted_test_data.csv"
    drifted_data.to_csv(drifted_data_path, index=False)
    
    # Mock the drift detection results
    mock_drift_results = {
        'dataset_drift': True,
        'drift_share': 0.33,
        'drifted_features': ['Age', 'MonthlyIncome', 'YearsAtCompany'],
        'n_drifted_features': 3,
        'report': {
            'metrics': [
                {'result': {'dataset_drift': True}},
                {'result': {'drift_by_columns': {
                    'Age': {'drift_detected': True},
                    'MonthlyIncome': {'drift_detected': True},
                    'YearsAtCompany': {'drift_detected': True}
                }}}
            ]
        }
    }
    
    # Create a mock reference data DataFrame
    mock_reference_data = drifted_data.copy()
    
    # Mock the prediction results
    mock_prediction_results = {
        'predictions': [0] * 100,  # 100 predictions
        'attrition_rate': 0.2
    }
    
    # Patch the necessary functions
    with patch('employee_attrition_mlops.drift_detection.check_drift', return_value=mock_drift_results), \
         patch('employee_attrition_mlops.drift_detection.should_trigger_retraining', return_value=True), \
         patch('employee_attrition_mlops.drift_detection.Report', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftPreset', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftTable', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DatasetDriftMetric', Mock()), \
         patch('employee_attrition_mlops.drift_detection.get_production_model_run_id', return_value='test-run-id'), \
         patch('employee_attrition_mlops.drift_detection.get_baseline_artifacts', return_value=('profile.json', 'reference.parquet', 'features.json')), \
         patch('pandas.read_parquet', return_value=mock_reference_data), \
         patch('scripts.optimize_train_select.optimize_select_and_train', return_value=None) as mock_train, \
         patch('scripts.batch_predict.main', return_value=mock_prediction_results) as mock_predict:
        
        # Run production automation
        from scripts.run_production_automation import run_production_automation
        run_production_automation(argparse.Namespace(csv_path=str(drifted_data_path), force_retrain=False))
        
        # Verify drift metrics were logged
        mock_mlflow['log_metric'].assert_any_call("dataset_drift", 1)
        mock_mlflow['log_metric'].assert_any_call("drift_share", 0.33)
        mock_mlflow['log_metric'].assert_any_call("n_drifted_features", 3)
        
        # Verify that optimize_select_and_train was called
        mock_train.assert_called_once()
        
        # Verify that batch prediction was called
        mock_predict.assert_called_once()

def test_force_retrain(test_data, mock_mlflow, tmp_path):
    """Test force retraining in production automation."""
    # Save test data to temporary CSV
    test_data_path = tmp_path / "test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    
    # Mock the drift detection results
    mock_drift_results = {
        'dataset_drift': False,
        'drift_share': 0.0,
        'drifted_features': [],
        'n_drifted_features': 0,
        'report': {
            'metrics': [
                {'result': {'dataset_drift': False}},
                {'result': {'drift_by_columns': {}}}
            ]
        }
    }
    
    # Mock the prediction results
    mock_prediction_results = {
        'predictions': [0] * 100,  # 100 predictions
        'attrition_rate': 0.2
    }
    
    # Create a mock reference data DataFrame
    mock_reference_data = test_data.copy()
    
    # Patch the necessary functions
    with patch('employee_attrition_mlops.drift_detection.check_drift', return_value=mock_drift_results), \
         patch('employee_attrition_mlops.drift_detection.should_trigger_retraining', return_value=False), \
         patch('scripts.batch_predict.main', return_value=mock_prediction_results), \
         patch('scripts.optimize_train_select.optimize_select_and_train') as mock_train, \
         patch('scripts.promote_model.promote_model_to_production') as mock_promote, \
         patch('employee_attrition_mlops.drift_detection.Report', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftPreset', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftTable', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DatasetDriftMetric', Mock()), \
         patch('employee_attrition_mlops.drift_detection.get_production_model_run_id', return_value='test-run-id'), \
         patch('employee_attrition_mlops.drift_detection.get_baseline_artifacts', return_value=('profile.json', 'reference.parquet', 'features.json')), \
         patch('pandas.read_parquet', return_value=mock_reference_data):
        
        # Run production automation with force retrain
        from scripts.run_production_automation import run_production_automation
        run_production_automation(argparse.Namespace(csv_path=str(test_data_path), force_retrain=True))
        
        # Verify retraining was triggered
        mock_train.assert_called_once()
        mock_promote.assert_called_once()
        mock_mlflow['log_metric'].assert_any_call("was_retrained", 1)

@pytest.mark.integration
def test_full_pipeline(test_data, drifted_data, tmp_path, mock_mlflow_client, mock_engine, mock_transformers, mock_env_vars, mock_sys_exit):
    """Integration test of the full production pipeline."""
    # Save both datasets
    test_data_path = tmp_path / "test_data.csv"
    drifted_data_path = tmp_path / "drifted_test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    drifted_data.to_csv(drifted_data_path, index=False)
    
    # Mock the drift detection results for normal data
    mock_normal_drift_results = {
        'dataset_drift': False,
        'drift_share': 0.0,
        'drifted_features': [],
        'n_drifted_features': 0,
        'report': {
            'metrics': [
                {'result': {'dataset_drift': False}},
                {'result': {'drift_by_columns': {}}}
            ]
        }
    }
    
    # Mock the drift detection results for drifted data
    mock_drifted_results = {
        'dataset_drift': True,
        'drift_share': 0.33,
        'drifted_features': ['Age', 'MonthlyIncome', 'YearsAtCompany'],
        'n_drifted_features': 3,
        'report': {
            'metrics': [
                {'result': {'dataset_drift': True}},
                {'result': {'drift_by_columns': {
                    'Age': {'drift_detected': True},
                    'MonthlyIncome': {'drift_detected': True},
                    'YearsAtCompany': {'drift_detected': True}
                }}}
            ]
        }
    }
    
    # Mock the prediction results
    mock_prediction_results = {
        'predictions': [0] * 100,  # 100 predictions
        'attrition_rate': 0.2
    }
    
    # Create mock reference data
    mock_reference_data = test_data.copy()
    
    # Patch the necessary functions
    with patch('employee_attrition_mlops.drift_detection.check_drift', side_effect=[mock_normal_drift_results, mock_drifted_results, mock_normal_drift_results]), \
         patch('employee_attrition_mlops.drift_detection.should_trigger_retraining', side_effect=[False, True, False]), \
         patch('employee_attrition_mlops.drift_detection.Report', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftPreset', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DataDriftTable', Mock()), \
         patch('employee_attrition_mlops.drift_detection.DatasetDriftMetric', Mock()), \
         patch('employee_attrition_mlops.drift_detection.get_production_model_run_id', return_value='test-run-id'), \
         patch('employee_attrition_mlops.drift_detection.get_baseline_artifacts', return_value=('profile.json', 'reference.parquet', 'features.json')), \
         patch('pandas.read_parquet', return_value=mock_reference_data), \
         patch('scripts.optimize_train_select.optimize_select_and_train', return_value=None) as mock_train, \
         patch('scripts.batch_predict.main', return_value=mock_prediction_results) as mock_predict, \
         patch('scripts.promote_model.promote_model_to_production') as mock_promote:
        
        # Run normal pipeline
        from scripts.run_production_automation import run_production_automation
        run_production_automation(argparse.Namespace(csv_path=str(test_data_path), force_retrain=False))
        
        # Run with drifted data
        run_production_automation(argparse.Namespace(csv_path=str(drifted_data_path), force_retrain=False))
        
        # Run with force retrain
        run_production_automation(argparse.Namespace(csv_path=str(test_data_path), force_retrain=True))
        
        # Verify that optimize_select_and_train was called twice (once for drift, once for force retrain)
        assert mock_train.call_count == 2
        
        # Verify that promote_model_to_production was called twice
        assert mock_promote.call_count == 2
        
        # Verify that batch prediction was called three times
        assert mock_predict.call_count == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 