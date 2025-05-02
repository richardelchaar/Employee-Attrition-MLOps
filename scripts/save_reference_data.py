#!/usr/bin/env python3
"""
Script to save reference data after model training.
This data will be used for drift detection in future runs.
"""
import os
import sys
import logging
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src path for imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from employee_attrition_mlops.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_MODEL_NAME,
    DB_HISTORY_TABLE,
    SNAPSHOT_DATE_COL
)
from employee_attrition_mlops.data_processing import load_data

def save_reference_data():
    """Save reference data for drift detection."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Get the latest production model
        logger.info(f"Getting latest version of {PRODUCTION_MODEL_NAME}")
        model_details = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["Production"])[0]
        
        # Load the data used for training
        logger.info("Loading training data...")
        X_train, X_test, y_train, y_test = load_data()
        
        # Combine features and target
        reference_data = pd.concat([
            pd.DataFrame(X_train, columns=X_train.columns),
            pd.Series(y_train, name='Attrition')
        ], axis=1)
        
        # Save reference data to MLflow
        with mlflow.start_run(run_name="save_reference_data") as run:
            # Save as CSV
            reference_data.to_csv("reference_data.csv", index=False)
            mlflow.log_artifact("reference_data.csv", "reference_data")
            
            # Save predictions for prediction drift detection
            model = mlflow.sklearn.load_model(f"runs:/{model_details.run_id}/model")
            reference_predictions = pd.DataFrame({
                'prediction': model.predict(X_test),
                'probability': model.predict_proba(X_test)[:, 1]
            })
            reference_predictions.to_csv("reference_predictions.csv", index=False)
            mlflow.log_artifact("reference_predictions.csv", "reference_predictions")
            
            # Log metadata
            mlflow.log_params({
                "num_samples": len(reference_data),
                "num_features": len(reference_data.columns) - 1,  # Exclude target
                "model_version": model_details.version,
                "model_run_id": model_details.run_id
            })
            
            logger.info(f"Reference data saved successfully. Run ID: {run.info.run_id}")
            
        # Clean up temporary files
        os.remove("reference_data.csv")
        os.remove("reference_predictions.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving reference data: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = save_reference_data()
    sys.exit(0 if success else 1) 