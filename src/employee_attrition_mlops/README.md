# Employee Attrition ML Pipeline

This directory contains the core machine learning pipeline for the Employee Attrition prediction system. The pipeline implements a complete workflow from data processing to model serving.

## Components

### Data Processing (`data_processing.py`)
- Raw data ingestion and validation
- Feature engineering
- Data cleaning and preprocessing
- Train/test split
- Feature scaling and encoding

### Model Pipeline (`pipelines.py`)
- Model training pipeline
- Hyperparameter optimization
- Model evaluation
- Cross-validation
- Model selection

### API (`api.py`)
- FastAPI endpoints for model serving
- Prediction endpoints
- Model metadata endpoints
- Health check endpoints

### Configuration (`config.py`)
- Pipeline configuration
- Model parameters
- Feature settings
- API settings

### Drift Detection (`drift_detection.py`)
- Feature drift monitoring
- Prediction drift detection
- Statistical tests implementation
- Alert generation

### Utilities (`utils.py`)
- Helper functions
- Data validation
- Logging utilities
- Error handling

## Pipeline Flow

1. **Data Ingestion**
   - Load raw employee data
   - Validate data schema
   - Handle missing values

2. **Feature Engineering**
   - Create derived features
   - Encode categorical variables
   - Scale numerical features
   - Handle imbalanced classes

3. **Model Training**
   - Split data into train/validation/test
   - Train multiple models
   - Optimize hyperparameters
   - Select best model

4. **Model Evaluation**
   - Calculate performance metrics
   - Generate feature importance
   - Create SHAP explanations
   - Validate model fairness

5. **Model Serving**
   - Deploy model to production
   - Serve predictions via API
   - Monitor model performance
   - Handle drift detection

## Usage

### Training Pipeline
```python
from employee_attrition_mlops.pipelines import train_pipeline

# Train model
model, metrics = train_pipeline(
    data_path="path/to/data.csv",
    target_column="attrition",
    config_path="config.yaml"
)
```

### Prediction Pipeline
```python
from employee_attrition_mlops.pipelines import predict_pipeline

# Make predictions
predictions = predict_pipeline(
    data=test_data,
    model_path="path/to/model.pkl"
)
```

### API Usage
```python
import requests

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": feature_values}
)
predictions = response.json()
```

## Configuration

The pipeline can be configured through:
- Environment variables
- Configuration files
- Command-line arguments

Key configuration options:
- Model parameters
- Feature engineering settings
- Training parameters
- API settings

## Testing

Run the test suite:
```bash
pytest tests/test_ml_pipeline.py
```

## Monitoring

The pipeline includes:
- Performance metrics tracking
- Feature drift detection
- Prediction drift monitoring
- Automated alerts

## Best Practices

1. **Data Quality**
   - Validate input data
   - Handle missing values
   - Check for data drift

2. **Model Development**
   - Use cross-validation
   - Optimize hyperparameters
   - Evaluate multiple models
   - Check for bias

3. **Production Readiness**
   - Log predictions
   - Monitor performance
   - Handle errors gracefully
   - Version models

4. **Documentation**
   - Document all functions
   - Include examples
   - Update README
   - Maintain changelog 