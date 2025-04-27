import pytest
from fastapi.testclient import TestClient
from src.employee_attrition_mlops.api import app
import mlflow
from unittest.mock import patch, MagicMock
import json

client = TestClient(app=app)

def test_health_endpoint_model_not_loaded():
    """Test health endpoint when model is not loaded."""
    with patch('src.employee_attrition_mlops.api.model', None):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["model_loaded"] is False
        assert "error" in data

def test_health_endpoint_with_model():
    """Test the health check endpoint when model is loaded."""
    with patch('src.employee_attrition_mlops.api.model', MagicMock()), \
         patch('mlflow.tracking.MlflowClient') as mock_client:
        # Mock MLflow client response
        mock_model = MagicMock()
        mock_model.latest_versions = [MagicMock(version="1", run_id="abc123")]
        mock_client.return_value.get_registered_model.return_value = mock_model
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "registered_model_name" in data
        assert "loaded_model_version" in data
        assert "loaded_model_run_id" in data

def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input data."""
    test_data = {
        "EmployeeNumber": 12345,
        "SnapshotDate": "2024-04-26",
        "Age": 30,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "EducationField": "Life Sciences",
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 5,
        "Education": 3,
        "DailyRate": 1102,
        "HourlyRate": 94,
        "MonthlyIncome": 7000,
        "MonthlyRate": 21410,
        "PercentSalaryHike": 12,
        "StockOptionLevel": 1,
        "OverTime": "No",
        "NumCompaniesWorked": 2,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2,
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobSatisfaction": 3,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "WorkLifeBalance": 3,
        "AgeGroup": "18-30"
    }
    
    with patch('src.employee_attrition_mlops.api.model') as mock_model:
        mock_model.predict.return_value = [0]  # Mock prediction
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]

def test_predict_endpoint_invalid_input():
    """Test prediction endpoint with invalid input data."""
    # First, we need to mock the model to be loaded
    with patch('src.employee_attrition_mlops.api.model', MagicMock()):
        # Test missing required fields
        test_data = {
            "EmployeeNumber": 12345,
            # Missing SnapshotDate
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 400
        assert "SnapshotDate are required" in response.json()["detail"]

        # Test invalid employee number (should return 400, not 422)
        test_data = {
            "EmployeeNumber": "not_a_number",
            "SnapshotDate": "2024-04-26"
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 400
        assert "Invalid value for EmployeeNumber" in response.json()["detail"]

def test_predict_endpoint_model_not_loaded():
    """Test prediction endpoint when model is not loaded."""
    with patch('src.employee_attrition_mlops.api.model', None):
        test_data = {
            "EmployeeNumber": 12345,
            "SnapshotDate": "2024-04-26"
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 503
        assert "Model is not loaded" in response.json()["detail"]

def test_model_info_endpoint_with_model():
    """Test the model info endpoint when model is loaded."""
    with patch('src.employee_attrition_mlops.api.model', MagicMock()), \
         patch('mlflow.tracking.MlflowClient') as mock_client:
        # Mock MLflow client response
        mock_model = MagicMock()
        mock_model.latest_versions = [MagicMock(
            version="1",
            run_id="abc123",
            status="READY",
            creation_timestamp=1234567890
        )]
        mock_client.return_value.get_registered_model.return_value = mock_model
        
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "registered_model_name" in data
        assert "latest_registered_version" in data
        assert "latest_registered_run_id" in data
        assert "latest_registered_status" in data
        assert "latest_registered_creation_timestamp" in data

def test_model_info_endpoint_no_model():
    """Test model info endpoint when model is not loaded."""
    with patch('src.employee_attrition_mlops.api.model', None):
        response = client.get("/model-info")
        assert response.status_code == 503
        assert "Model not loaded yet" in response.json()["detail"]

def test_predict_endpoint_database_logging():
    """Test that predictions are logged to database when DB is available."""
    test_data = {
        "EmployeeNumber": 12345,
        "SnapshotDate": "2024-04-26",
        "Age": 30,
        "Gender": "Male",
        "MaritalStatus": "Single",
        "Department": "Sales",
        "EducationField": "Life Sciences",
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 5,
        "Education": 3,
        "DailyRate": 1102,
        "HourlyRate": 94,
        "MonthlyIncome": 7000,
        "MonthlyRate": 21410,
        "PercentSalaryHike": 12,
        "StockOptionLevel": 1,
        "OverTime": "No",
        "NumCompaniesWorked": 2,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2,
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobSatisfaction": 3,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "WorkLifeBalance": 3,
        "AgeGroup": "18-30"
    }
    
    # Create a mock engine with a begin method that returns a context manager
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.execute.return_value = MagicMock(scalar=lambda: 0)  # No existing prediction
    
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn
    
    with patch('src.employee_attrition_mlops.api.model', MagicMock()) as mock_model, \
         patch('src.employee_attrition_mlops.api.engine', mock_engine):
        mock_model.predict.return_value = [0]
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        # Check if the engine was used
        mock_engine.begin.assert_called_once()
        # Check if the execute method was called
        assert mock_conn.execute.call_count >= 1 