import mlflow
import logging
import pandas as pd
import json
import os
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")
logging.getLogger('mlflow').setLevel(logging.DEBUG)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_and_log_drift_reference(run_id, data_path=None):
    """Create and log drift reference artifacts for a model."""
    
    # If no data provided, create sample data (for testing)
    if data_path is None:
        # Create sample data with consistent length
        n_samples = 100
        data = pd.DataFrame({
            'Age': np.random.randint(20, 60, n_samples),
            'MonthlyIncome': np.random.randint(30000, 150000, n_samples),
            'YearsAtCompany': np.random.randint(0, 40, n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
            'WorkLifeBalance': np.random.randint(1, 5, n_samples),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
            'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Manager'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
        })
    else:
        # Load actual data
        data = pd.read_csv(data_path)
    
    # Create temporary directory for artifacts
    os.makedirs("drift_reference", exist_ok=True)
    
    # 1. Save reference data
    data.to_parquet("drift_reference/reference_train_data.parquet")
    
    # 2. Create and save feature names
    feature_names = list(data.columns)
    with open("drift_reference/reference_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    
    # 3. Create and save data profile
    profile = Report(metrics=[
        DataDriftPreset(),
    ])
    profile.run(reference_data=data, current_data=data)  # Using same data as reference and current for profile
    profile_dict = profile.as_dict()
    
    with open("drift_reference/training_data_profile.json", "w") as f:
        json.dump(profile_dict, f)
    
    print("Created drift reference artifacts locally")
    
    # Log artifacts to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts("drift_reference", "drift_reference")
        print(f"Successfully logged drift reference artifacts to MLflow run {run_id}!")

def check_drift(current_data, reference_data):
    """Check for drift between current and reference data."""
    # Create Evidently drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),  # Remove stat_test_threshold parameter
    ])
    
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Extract drift metrics
    report_dict = data_drift_report.as_dict()
    
    # Count drifted features
    drifted_features = []
    try:
        metrics = report_dict['metrics'][0]['result']['data_drift']  # Updated path
        feature_results = report_dict['metrics'][0]['result']['data_drift']['drift_by_columns']  # Updated path
        
        for feature, result in feature_results.items():
            if result.get('drift_detected', False):
                drifted_features.append(feature)
                
        dataset_drift = metrics.get('dataset_drift', False)
        drift_share = metrics.get('share_of_drifted_columns', 0)  # Updated key name
        
        logger.info(f"Dataset drift detected: {dataset_drift}")
        logger.info(f"Share of drifted features: {drift_share:.2f}")
        logger.info(f"Drifted features ({len(drifted_features)}): {drifted_features}")
        
        return {
            'dataset_drift': dataset_drift,
            'drift_share': drift_share,
            'drifted_features': drifted_features,
            'n_drifted_features': len(drifted_features),
            'report': report_dict
        }
    except KeyError as e:
        logger.error(f"Error extracting drift metrics: {e}")
        return {
            'dataset_drift': False,
            'drift_share': 0,
            'drifted_features': [],
            'n_drifted_features': 0,
            'error': str(e)
        }

if __name__ == "__main__":
    # Get the run_id of your production model
    client = mlflow.tracking.MlflowClient()
    model_name = "AttritionProductionModel"
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if versions:
        run_id = versions[0].run_id
        print(f"Found production model run_id: {run_id}")
        create_and_log_drift_reference(run_id)
    else:
        print("No production model found!")