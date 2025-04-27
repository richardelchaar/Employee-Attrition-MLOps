#!/usr/bin/env python
# scripts/run_production_automation.py
import os
import sys
import logging
import argparse
from datetime import datetime
import mlflow
from dotenv import load_dotenv
import pandas as pd

# Add src to Python path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from employee_attrition_mlops.data_processing import load_and_clean_data
from employee_attrition_mlops.pipelines import create_preprocessing_pipeline
from employee_attrition_mlops.config import (
    TARGET_COLUMN, DB_HISTORY_TABLE, DATABASE_URL_PYMSSQL,
    SNAPSHOT_DATE_COL, SKEWNESS_THRESHOLD, MLFLOW_TRACKING_URI,
    PRODUCTION_MODEL_NAME, REPORTS_PATH, DRIFT_REPORT_FILENAME
)
from employee_attrition_mlops.utils import save_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("production_automation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("employee_attrition_production")

def run_production_automation(args):
    """
    Run the complete production automation:
    1. Load and clean data
    2. Run drift detection
    3. Retrain if needed
    4. Make predictions
    5. Log results
    """
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"production_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. Load and clean data
            logger.info("Loading and cleaning data...")
            if args.csv_path:
                data = load_and_clean_data(path=args.csv_path)
            else:
                data = load_and_clean_data()  # Load from DB
            
            if data is None or data.empty:
                raise ValueError("No data available for processing")
            
            # 2. Run drift detection
            logger.info("Running drift detection...")
            from employee_attrition_mlops.drift_detection import (
                get_baseline_artifacts, check_drift, should_trigger_retraining,
                get_production_model_run_id
            )
            
            # Get production model run ID
            run_id = get_production_model_run_id(PRODUCTION_MODEL_NAME)
            if not run_id:
                raise ValueError("Could not find production model run ID")
            
            # Get baseline artifacts
            baseline_profile_path, reference_data_path, feature_names_path = get_baseline_artifacts(run_id)
            
            # Load reference data
            reference_data = pd.read_parquet(reference_data_path)
            logger.info(f"Loaded reference data with shape: {reference_data.shape}")
            
            # Check drift
            drift_results = check_drift(data, reference_data)
            
            # Save drift report
            os.makedirs(REPORTS_PATH, exist_ok=True)
            drift_report_path = os.path.join(REPORTS_PATH, DRIFT_REPORT_FILENAME)
            save_json(drift_results, drift_report_path)
            logger.info(f"Drift report saved to {drift_report_path}")
            
            # Determine if retraining is needed
            needs_retraining = should_trigger_retraining(drift_results)
            
            # 3. Check if retraining is needed
            if needs_retraining or args.force_retrain:
                logger.info("Starting retraining process...")
                from scripts.optimize_train_select import optimize_select_and_train
                optimize_select_and_train()  # This will handle model retraining
                
                # Promote the new model if it's better
                from scripts.promote_model import promote_model_to_production
                promote_model_to_production()
            else:
                logger.info("No retraining needed. Using existing model.")
            
            # 4. Make predictions
            logger.info("Making predictions...")
            from scripts.batch_predict import main as run_batch_prediction
            prediction_results = run_batch_prediction()  # No arguments needed
            
            # 5. Log results
            logger.info("Logging results...")
            if isinstance(prediction_results, dict) and 'predictions' in prediction_results:
                mlflow.log_metric("num_predictions", len(prediction_results['predictions']))
                mlflow.log_metric("attrition_rate", prediction_results.get('attrition_rate', 0.0))
            else:
                logger.warning("Unexpected prediction results format. Skipping prediction metrics.")
                mlflow.log_metric("num_predictions", 0)
                mlflow.log_metric("attrition_rate", 0.0)
            
            mlflow.log_metric("was_retrained", int(needs_retraining or args.force_retrain))
            
            # Log drift metrics
            mlflow.log_metric("dataset_drift", int(drift_results['dataset_drift']))
            mlflow.log_metric("drift_share", drift_results['drift_share'])
            mlflow.log_metric("n_drifted_features", drift_results['n_drifted_features'])
            
            # Log drift report
            mlflow.log_artifact(drift_report_path)
            
            logger.info("Production automation completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in production automation: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Run the employee attrition production automation")
    parser.add_argument("--csv-path", type=str, help="Path to CSV file (optional, defaults to database)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining regardless of drift")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    if not DATABASE_URL_PYMSSQL:
        logger.error("DATABASE_URL_PYMSSQL not found in environment variables")
        sys.exit(1)
    
    # Run automation
    run_production_automation(args)

if __name__ == "__main__":
    main() 