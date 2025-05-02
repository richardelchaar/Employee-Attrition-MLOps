#!/usr/bin/env python3
"""
Script to check for prediction drift between reference and current predictions.
This can be run independently or as part of the MLOps pipeline.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import mlflow
import requests
from mlflow.tracking import MlflowClient

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from employee_attrition_mlops.data_processing import load_data
from employee_attrition_mlops.model_training import train_model
from employee_attrition_mlops.model_evaluation import evaluate_model
from employee_attrition_mlops.model_selection import select_best_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_reference_predictions():
    """Load reference predictions from MLflow."""
    try:
        client = MlflowClient()
        # Get the latest production model
        model_details = client.get_latest_versions("EmployeeAttrition", stages=["Production"])[0]
        
        # Load reference predictions from MLflow
        reference_predictions = pd.read_csv(
            f"mlruns/{model_details.run_id}/artifacts/reference_predictions.csv"
        )
        return reference_predictions
    except Exception as e:
        logger.error(f"Error loading reference predictions: {str(e)}")
        return None

def save_drift_report(drift_results):
    """Save drift detection results to a JSON file."""
    os.makedirs('reports', exist_ok=True)
    with open('reports/prediction_drift_report.json', 'w') as f:
        json.dump(drift_results, f, indent=2)

def main():
    try:
        # Load reference predictions
        reference_predictions = load_reference_predictions()
        if reference_predictions is None:
            logger.error("Could not load reference predictions")
            sys.exit(1)
            
        # Load and prepare data
        logger.info("Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        logger.info("Training model...")
        model = train_model(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        current_predictions = pd.DataFrame({
            'prediction': model.predict(X_test),
            'probability': model.predict_proba(X_test)[:, 1]
        })
        
        # Call drift API
        logger.info("Calling drift API...")
        api_url = "http://localhost:8000/drift/prediction"  # Update with your API URL
        
        response = requests.post(
            api_url,
            json={
                'reference_predictions': reference_predictions.to_dict(orient='records'),
                'current_predictions': current_predictions.to_dict(orient='records')
            }
        )
        
        if response.status_code != 200:
            logger.error(f"API call failed: {response.text}")
            sys.exit(1)
            
        drift_results = response.json()
        
        # Save drift report
        save_drift_report(drift_results)
        
        # Log to MLflow
        with mlflow.start_run(run_name="prediction_drift_check") as run:
            mlflow.log_metrics({
                'drift_score': drift_results['drift_score'],
                'drift_detected': int(drift_results['drift_detected'])
            })
            
            # Log the drift report
            mlflow.log_artifact('reports/prediction_drift_report.json')
            
            # Set output for GitHub Actions
            print(f"::set-output name=run_id::{run.info.run_id}")
            print(f"::set-output name=drift_detected::{str(drift_results['drift_detected']).lower()}")
            
        # Exit with appropriate status code
        if drift_results['drift_detected']:
            logger.info("Drift detected in predictions")
            sys.exit(1)
        else:
            logger.info("No drift detected in predictions")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error during prediction drift check: {str(e)}")
        # Save error information
        error_summary = {
            'error_type': type(e).__name__,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        os.makedirs('reports', exist_ok=True)
        with open('reports/automation_error.json', 'w') as f:
            json.dump(error_summary, f, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    main() 