# Drift Detection Guide

This guide provides detailed information about the drift detection system in the Employee Attrition MLOps project.

## Table of Contents

- [Overview](#overview)
- [Drift Detection Parameters](#drift-detection-parameters)
  - [Configuration Settings](#configuration-settings)
  - [Feature Drift Detection](#feature-drift-detection)
  - [Prediction Drift Detection](#prediction-drift-detection)
  - [Interpreting Drift Results](#interpreting-drift-results)
  - [Common Drift Patterns](#common-drift-patterns)
- [Using the Drift Detection API](#using-the-drift-detection-api)
  - [Starting the API Server](#starting-the-api-server)
  - [API Endpoints](#api-endpoints)
  - [Example API Usage](#example-api-usage)
- [Automated Drift Detection](#automated-drift-detection)
  - [Production Monitoring](#production-monitoring)
  - [GitHub Actions Integration](#github-actions-integration)
- [MLflow Integration](#mlflow-integration)
  - [Drift Metrics in MLflow](#drift-metrics-in-mlflow)
  - [Viewing Drift Metrics](#viewing-drift-metrics)
- [Reference Data Management](#reference-data-management)
  - [Setting Up Reference Data](#setting-up-reference-data)
  - [Reference Data Structure](#reference-data-structure)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Logging](#logging)

## Overview

The drift detection system monitors both feature drift and prediction drift using statistical tests. It integrates with MLflow for tracking metrics and can trigger automated retraining when significant drift is detected.

## Drift Detection Parameters

### Configuration Settings
```python
# Default drift detection parameters
DRIFT_CONFIDENCE = 0.95  # Confidence level for statistical tests
DRIFT_STATTEST_THRESHOLD = 0.05  # Statistical test threshold
RETRAIN_TRIGGER_FEATURE_COUNT = 3  # Number of drifted features to trigger retraining
RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE = 0.05  # P-value threshold for dataset drift
```

### Feature Drift Detection
- Uses Evidently's statistical tests
- Compares current data against reference data
- Monitors both numerical and categorical features
- Generates detailed drift reports

### Prediction Drift Detection
- Monitors model predictions over time
- Compares prediction distributions
- Tracks prediction drift scores
- Triggers alerts when thresholds are exceeded

### Interpreting Drift Results

Drift detection outputs several key metrics that help you understand the state of your model:

1. **Drift Score (0-1)**
   - Higher scores indicate more significant drift
   - Default threshold is 0.05
   - Scores above threshold trigger alerts
   - Consider both magnitude and trend over time

2. **Drifted Features**
   - List of features showing significant drift
   - Number of drifted features indicates severity
   - 3+ drifted features trigger retraining
   - Check feature importance for context

3. **Drift Share**
   - Percentage of features showing drift
   - Helps assess overall data stability
   - High drift share suggests systematic changes
   - Consider business context for interpretation

4. **Statistical Test Results**
   - P-values for each feature
   - Confidence levels for drift detection
   - Helps identify most significant changes
   - Use to prioritize investigation

5. **Prediction Drift**
   - Changes in model output distribution
   - May indicate concept drift
   - Compare with feature drift
   - Critical for model performance

### Common Drift Patterns

1. **Gradual Drift**
   - Slow, consistent changes over time
   - May indicate changing business conditions
   - Consider scheduled model updates

2. **Sudden Drift**
   - Abrupt changes in distributions
   - May indicate data pipeline issues
   - Investigate immediately

3. **Seasonal Drift**
   - Regular patterns in drift scores
   - May require seasonal model variants
   - Document and account for in monitoring

4. **Feature-Specific Drift**
   - Drift in specific features only
   - May indicate data quality issues
   - Check data collection processes

## Using the Drift Detection API

The project includes a FastAPI endpoint for drift detection:

### Starting the API Server

```bash
# Start the API server
python drift_api.py

# With custom port
PORT=8080 python drift_api.py
```

### API Endpoints

The API is documented at http://localhost:8000/docs and includes:

- `GET /health`: Health check endpoint
- `POST /detect-drift`: Detect drift from JSON data
  - Parameters:
    - `data`: List of feature records
    - `threshold`: Drift detection threshold (default: 0.05)
    - `confidence`: Statistical test confidence level (default: 0.95)
    - `features`: Optional list of features to monitor
- `GET /`: API information and documentation

### Example API Usage

```python
import requests
import pandas as pd

# Load data
df = pd.read_csv("path/to/data.csv")
data = df.to_dict(orient="records")

# Detect drift
response = requests.post(
    "http://localhost:8000/detect-drift",
    json={
        "data": data,
        "threshold": 0.05,
        "confidence": 0.95,
        "features": ["age", "salary", "satisfaction_score"]
    }
)

# Check results
result = response.json()
print(f"Drift detected: {result['drift_detected']}")
print(f"Drift score: {result['drift_score']}")
print(f"Drifted features: {result['drifted_features']}")
```

## Automated Drift Detection

### Production Monitoring

The system includes two main drift detection scripts:

1. `check_production_drift.py`:
   - Runs scheduled drift checks
   - Compares against reference data
   - Logs results to MLflow
   - Generates drift reports

2. `check_drift_via_api.py`:
   - Uses the drift detection API
   - Suitable for integration with external systems
   - Supports custom thresholds and features

### GitHub Actions Integration

The project includes GitHub Actions workflows for automated drift detection:

#### Scheduled Monitoring
- Runs automatically every Monday
- Performs drift detection on latest data
- Creates GitHub issues if drift is detected
- Generates HTML reports
- Fixes MLflow metadata issues

#### Manual Triggering
1. Go to the "Actions" tab in your GitHub repository
2. Select the "Drift Detection" workflow
3. Click "Run workflow"

## MLflow Integration

### Drift Metrics in MLflow

The system logs the following metrics to MLflow:

1. **Feature Drift Metrics**
   - `drift_detected`: Binary indicator
   - `drift_score`: Overall drift score
   - `n_drifted_features`: Number of drifted features
   - `drifted_features`: List of drifted feature names

2. **Prediction Drift Metrics**
   - `prediction_drift_detected`: Binary indicator
   - `prediction_drift_score`: Prediction drift score
   - `prediction_distribution_drift`: Distribution drift score

### Viewing Drift Metrics

1. Start the MLflow UI:
   ```bash
   mlflow ui --port 5001
   ```

2. Navigate to the drift detection experiment
3. View drift metrics and reports
4. Compare drift scores over time

## Reference Data Management

### Setting Up Reference Data

1. Save new reference data:
   ```bash
   python save_reference_data.py
   ```

2. Update reference predictions:
   ```bash
   python save_reference_predictions.py
   ```

### Reference Data Structure

- `reference_data/`: Contains baseline feature data
- `reference_predictions/`: Contains baseline predictions
- `reports/`: Stores drift detection reports

## Troubleshooting

### Common Issues

1. **MLflow Connection Issues**
   ```bash
   # Check MLflow server
   curl http://localhost:5001/health
   ```

2. **Reference Data Issues**
   ```bash
   # Verify reference data
   python scripts/verify_reference_data.py
   ```

3. **API Connection Issues**
   ```bash
   # Test API connection
   curl http://localhost:8000/health
   ```

### Logging

- Check `production_automation.log` for production drift detection logs
- Check `test_production_automation.log` for test drift detection logs
- MLflow logs contain detailed drift detection metrics 