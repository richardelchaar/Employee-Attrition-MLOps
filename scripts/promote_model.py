#!/usr/bin/env python
# scripts/promote_model.py

import argparse
import logging
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from employee_attrition_mlops.config import (
    MLFLOW_TRACKING_URI,
    PRODUCTION_MODEL_NAME
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def promote_model_to_production(model_version: int = None, run_id: str = None) -> bool:
    """
    Promotes a model to production. Can specify either model_version or run_id.
    Returns True if promotion was successful.
    """
    client = MlflowClient()
    
    try:
        # If run_id is provided, find its model version
        if run_id and not model_version:
            versions = client.search_model_versions(f"name='{PRODUCTION_MODEL_NAME}'")
            for version in versions:
                if version.run_id == run_id:
                    model_version = version.version
                    break
            if not model_version:
                logger.error(f"Could not find model version for run_id {run_id}")
                return False
        
        # If neither is provided, use latest staging version
        if not model_version and not run_id:
            latest_staging = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["Staging"])
            if not latest_staging:
                logger.error("No model version found in staging")
                return False
            model_version = latest_staging[0].version
        
        # Transition the model to production
        client.transition_model_version_stage(
            name=PRODUCTION_MODEL_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True  # Archive current production model
        )
        
        logger.info(f"Successfully promoted model version {model_version} to production")
        return True
        
    except Exception as e:
        logger.error(f"Error promoting model to production: {e}")
        return False

def main(args):
    """Main function for model promotion."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        success = promote_model_to_production(
            model_version=args.model_version,
            run_id=args.run_id
        )
        
        if not success:
            logger.error("Model promotion failed")
            return 1
        
        logger.info("Model promotion completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in model promotion: {e}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Attrition Model Promotion")
    parser.add_argument("--model-version", type=int, help="MLflow model version to promote")
    parser.add_argument("--run-id", type=str, help="MLflow run ID of the model to promote")
    
    args = parser.parse_args()
    sys.exit(main(args))# Test comment for workflow verification
