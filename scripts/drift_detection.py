#!/usr/bin/env python
# scripts/drift_detection.py

import argparse
import os
import sys
import logging
import mlflow
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from employee_attrition_mlops.config import (
    MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME, DRIFT_REPORT_FILENAME,
    REPORTS_PATH, RETRAIN_TRIGGER_FEATURE_COUNT
)
from employee_attrition_mlops.utils import (
    save_json, load_json, get_production_model_run_id, download_mlflow_artifact
)
from employee_attrition_mlops.data_processing import load_and_clean_data

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")
logging.getLogger('mlflow').setLevel(logging.DEBUG)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_baseline_artifacts(run_id):
    """Download baseline artifacts from MLflow."""
    artifact_dir = "drift_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Download baseline profile
    baseline_profile_path = download_mlflow_artifact(
        run_id, "drift_reference/training_data_profile.json", artifact_dir
    )
    
    # Download reference data 
    reference_data_path = download_mlflow_artifact(
        run_id, "drift_reference/reference_train_data.parquet", artifact_dir
    )
    
    # Download feature names list
    feature_names_path = download_mlflow_artifact(
        run_id, "drift_reference/reference_feature_names.json", artifact_dir
    )
    
    return baseline_profile_path, reference_data_path, feature_names_path

def check_drift(current_data, reference_data):
    """Check for drift between current and reference data."""
    # Create Evidently drift report with specific metrics
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Extract drift metrics
    report_dict = data_drift_report.as_dict()
    
    try:
        # Get results from DatasetDriftMetric
        drift_metric = report_dict['metrics'][0]['result']
        drift_table = report_dict['metrics'][1]['result']
        
        # Extract drifted features from the drift table
        drifted_features = []
        for feature, stats in drift_table['drift_by_columns'].items():
            if stats['drift_detected']:
                drifted_features.append(feature)
        
        dataset_drift = drift_metric['dataset_drift']
        drift_share = len(drifted_features) / len(drift_table['drift_by_columns'])
        
        # Add detailed logging
        logger.info(f"Drift detection details:")
        for feature, stats in drift_table['drift_by_columns'].items():
            logger.info(f"Feature: {feature}")
            logger.info(f"  - Drift detected: {stats['drift_detected']}")
            logger.info(f"  - P-value: {stats.get('p_value', 'N/A')}")
            logger.info(f"  - Test statistic: {stats.get('test_statistic', 'N/A')}")
        
        return {
            'dataset_drift': dataset_drift,
            'drift_share': drift_share,
            'drifted_features': drifted_features,
            'n_drifted_features': len(drifted_features),
            'report': report_dict
        }
    except KeyError as e:
        logger.error(f"Error extracting drift metrics: {e}")
        logger.debug(f"Available keys in report: {report_dict.keys()}")
        logger.debug(f"Metrics structure: {report_dict.get('metrics', [])}")
        return {
            'dataset_drift': False,
            'drift_share': 0,
            'drifted_features': [],
            'n_drifted_features': 0,
            'error': str(e)
        }

def should_trigger_retraining(drift_results):
    """Determine if retraining should be triggered based on drift results."""
    # Check if dataset drift is detected
    if drift_results['dataset_drift']:
        logger.info("Dataset drift detected. Suggesting retraining.")
        return True
    
    # Check if number of drifted features exceeds threshold
    if drift_results['n_drifted_features'] >= RETRAIN_TRIGGER_FEATURE_COUNT:
        logger.info(f"Number of drifted features ({drift_results['n_drifted_features']}) exceeds threshold ({RETRAIN_TRIGGER_FEATURE_COUNT}). Suggesting retraining.")
        return True
    
    logger.info("Drift does not exceed retraining thresholds.")
    return False

def trigger_github_workflow(trigger_retraining):
    """Trigger GitHub workflow for retraining if needed."""
    if not trigger_retraining:
        logger.info("No retraining needed, skipping GitHub workflow trigger.")
        return
    
    # Import GitHub workflow trigger functionality
    try:
        from employee_attrition_mlops.github_actions import trigger_workflow
        
        logger.info("Triggering GitHub workflow for model retraining.")
        success = trigger_workflow()
        if success:
            logger.info("Successfully triggered retraining workflow in GitHub Actions.")
        else:
            logger.error("Failed to trigger retraining workflow.")
    except ImportError:
        logger.error("GitHub Actions integration not available. Implement github_actions.py first.")
        print("RETRAINING REQUIRED - Manual intervention needed!")

def main(args):
    """Main function for drift detection."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get run_id of current production model
        run_id = args.run_id or get_production_model_run_id(PRODUCTION_MODEL_NAME)
        if not run_id:
            logger.error("Could not find production model run ID.")
            return 2  # Error exit code
        
        # Get baseline artifacts
        baseline_profile_path, reference_data_path, feature_names_path = get_baseline_artifacts(run_id)
        
        # Load reference data
        reference_data = pd.read_parquet(reference_data_path)
        logger.info(f"Loaded reference data with shape: {reference_data.shape}")
        
        # Create current data by sampling from reference data
        current_data = reference_data.copy()
        
        # If simulating drift, modify the numeric columns
        if args.simulate_drift:
            logger.info("Simulating drift in data")
            numeric_cols = current_data.select_dtypes(include=['number']).columns
            logger.info(f"Numeric columns available for drift: {numeric_cols}")
            
            for col in numeric_cols[:3]:
                # Log original stats
                logger.info(f"\nColumn {col} before drift:")
                logger.info(f"Mean: {current_data[col].mean():.2f}")
                logger.info(f"Std: {current_data[col].std():.2f}")
                
                # Apply stronger drift
                multiplier = np.random.uniform(2.0, 3.0, len(current_data))
                current_data[col] = current_data[col] * multiplier
                
                # Log modified stats
                logger.info(f"Column {col} after drift:")
                logger.info(f"Mean: {current_data[col].mean():.2f}")
                logger.info(f"Std: {current_data[col].std():.2f}")
        
        # Check drift between reference and modified current data
        drift_results = check_drift(current_data, reference_data)
        
        # Save drift report
        os.makedirs(REPORTS_PATH, exist_ok=True)
        drift_report_path = os.path.join(REPORTS_PATH, DRIFT_REPORT_FILENAME)
        save_json(drift_results, drift_report_path)
        logger.info(f"Drift report saved to {drift_report_path}")
        
        # Log drift metrics to MLflow
        with mlflow.start_run(run_name="drift_monitoring") as run:
            mlflow.log_param("production_model_run_id", run_id)
            mlflow.log_param("data_path", args.data_path or "simulated_data")
            mlflow.log_param("simulate_drift", args.simulate_drift)
            
            mlflow.log_metric("dataset_drift", int(drift_results['dataset_drift']))
            mlflow.log_metric("drift_share", drift_results['drift_share'])
            mlflow.log_metric("n_drifted_features", drift_results['n_drifted_features'])
            
            mlflow.log_artifact(drift_report_path)
        
        # Determine if retraining is needed
        trigger_retraining = should_trigger_retraining(drift_results)
        
        # If retraining is needed, trigger GitHub workflow
        if trigger_retraining and args.trigger_workflow:
            trigger_github_workflow(trigger_retraining)
        
        # Return appropriate exit code based on drift detection
        return 1 if trigger_retraining else 0
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Attrition Drift Detection")
    parser.add_argument("--data-path", type=str, help="Path to new data for drift detection")
    parser.add_argument("--run-id", type=str, help="MLflow run ID of the baseline model")
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate drift for testing")
    parser.add_argument("--trigger-workflow", action="store_true", help="Trigger retraining workflow if drift detected")
    
    args = parser.parse_args()
    exit_code = main(args)
    sys.exit(exit_code)