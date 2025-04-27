# API Documentation

## Overview
The Employee Attrition API provides endpoints for making predictions and accessing model information. The API is built with FastAPI and automatically generates OpenAPI/Swagger documentation at `/docs` and `/redoc`.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication.

## Endpoints

### 1. Health Check
```http
GET /health
```
Checks the health of the API and model status.

**Response**
```json
{
    "status": "ok",
    "model_loaded": true,
    "registered_model_name": "AttritionProductionModel",
    "loaded_model_version": "1",
    "loaded_model_run_id": "abc123"
}
```

### 2. Model Information
```http
GET /model-info
```
Returns detailed information about the latest registered model version.

**Response**
```json
{
    "registered_model_name": "AttritionProductionModel",
    "latest_registered_version": "1",
    "latest_registered_run_id": "abc123",
    "latest_registered_status": "READY",
    "latest_registered_creation_timestamp": 1647123456789
}
```

### 3. Prediction
```http
POST /predict
```
Makes predictions for employee attrition.

**Request Body**
```json
{
    "EmployeeNumber": 12345,
    "SnapshotDate": "2024-04-26",
    "Age": 35,
    "Gender": "Male",
    "MaritalStatus": "Married",
    "Department": "Sales",
    "EducationField": "Marketing",
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "BusinessTravel": "Travel_Rarely",
    "DistanceFromHome": 10,
    "Education": 3,
    "DailyRate": 800,
    "HourlyRate": 50,
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "PercentSalaryHike": 15,
    "StockOptionLevel": 1,
    "OverTime": "No",
    "NumCompaniesWorked": 2,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 2,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 2,
    "EnvironmentSatisfaction": 4,
    "JobInvolvement": 3,
    "JobSatisfaction": 4,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 4,
    "WorkLifeBalance": 3,
    "AgeGroup": "35-40"
}
```

**Response**
```json
{
    "EmployeeNumber": 12345,
    "SnapshotDate": "2024-04-26",
    "prediction": 0
}
```

## Error Responses

### 400 Bad Request
```json
{
    "detail": "EmployeeNumber and SnapshotDate are required in request."
}
```

### 503 Service Unavailable
```json
{
    "detail": "Model is not loaded. Cannot make predictions."
}
```

### 500 Internal Server Error
```json
{
    "detail": "Internal server error: [error message]"
}
```

## Data Types

### Employee Data
| Field | Type | Description |
|-------|------|-------------|
| EmployeeNumber | integer | Unique employee identifier |
| SnapshotDate | string | Date of the data snapshot |
| Age | integer | Employee's age |
| Gender | string | Employee's gender |
| MaritalStatus | string | Employee's marital status |
| Department | string | Employee's department |
| EducationField | string | Field of education |
| JobLevel | integer | Job level (1-5) |
| JobRole | string | Employee's job role |
| BusinessTravel | string | Frequency of business travel |
| DistanceFromHome | integer | Distance from home in miles |
| Education | integer | Education level (1-5) |
| DailyRate | integer | Daily rate of pay |
| HourlyRate | integer | Hourly rate of pay |
| MonthlyIncome | integer | Monthly income |
| MonthlyRate | integer | Monthly rate of pay |
| PercentSalaryHike | integer | Percentage of salary hike |
| StockOptionLevel | integer | Stock option level (0-3) |
| OverTime | string | Whether employee works overtime |
| NumCompaniesWorked | integer | Number of companies worked for |
| TotalWorkingYears | integer | Total years of work experience |
| TrainingTimesLastYear | integer | Number of training times last year |
| YearsAtCompany | integer | Years at current company |
| YearsInCurrentRole | integer | Years in current role |
| YearsSinceLastPromotion | integer | Years since last promotion |
| YearsWithCurrManager | integer | Years with current manager |
| EnvironmentSatisfaction | integer | Environment satisfaction (1-4) |
| JobInvolvement | integer | Job involvement (1-4) |
| JobSatisfaction | integer | Job satisfaction (1-4) |
| PerformanceRating | integer | Performance rating (1-4) |
| RelationshipSatisfaction | integer | Relationship satisfaction (1-4) |
| WorkLifeBalance | integer | Work-life balance (1-4) |
| AgeGroup | string | Age group category |

## Rate Limiting
Currently, there are no rate limits implemented.

## Versioning
The API version is included in the response headers as `X-API-Version`. 