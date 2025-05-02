import pytest
from unittest.mock import patch
import requests
from datetime import datetime

def test_api_integration_success():
    """Test successful API integration for predictions."""
    payload = {
        "EmployeeNumber": 12345,
        "SnapshotDate": datetime.now().strftime("%Y-%m-%d"),
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
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"prediction": 1}
        response = requests.post("http://localhost:8000/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["prediction"] == 1

def test_api_integration_error():
    """Test API error handling."""
    payload = {
        "EmployeeNumber": 12345,
        "SnapshotDate": datetime.now().strftime("%Y-%m-%d")
    }
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 500
        response = requests.post("http://localhost:8000/predict", json=payload)
        assert response.status_code == 500

def test_model_info_fetch():
    """Test fetching model information."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "latest_registered_version": "1",
            "latest_registered_run_id": "abc123",
            "latest_registered_creation_timestamp": 1234567890
        }
        response = requests.get("http://localhost:8000/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "latest_registered_version" in data
        assert "latest_registered_run_id" in data

def test_health_check():
    """Test health check endpoint."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "ok",
            "model_loaded": True
        }
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True 