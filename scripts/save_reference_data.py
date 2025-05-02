#!/usr/bin/env python3
"""
Script to save reference data after model training.
This data will be used for drift detection comparing new data against these references.

WHY TWO SEPARATE REFERENCE FILES:
1. reference_data.parquet: Contains the FEATURE values that new input data will be compared against
   - Used for FEATURE DRIFT detection (comparing new employee feature values against past values)
   - Contains both features and target from training data to maintain the full distribution

2. reference_predictions.csv: Contains the MODEL OUTPUTS (predictions & probabilities) 
   - Used for PREDICTION DRIFT detection (comparing new model outputs against past outputs)
   - Contains prediction values and probabilities to detect shifts in model behavior
   - Even if features haven't drifted, the model's prediction patterns could drift
"""
import os
import sys
import logging
import pandas as pd
import mlflow
import time
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

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
    SNAPSHOT_DATE_COL,
    TARGET_COLUMN
)
from employee_attrition_mlops.data_processing import (
    load_and_clean_data, 
    AddNewFeaturesTransformer, 
    AgeGroupTransformer
)

# Define MLflow artifact paths
REFERENCE_FEATURES_DIR = "reference_data"
REFERENCE_FEATURES_FILENAME = "reference_data.parquet"
REFERENCE_PREDICTIONS_DIR = "reference_predictions"
REFERENCE_PREDICTIONS_FILENAME = "reference_predictions.csv"

# Retry configuration - same as API
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds

def load_production_model(model_name: str):
    """Load the latest model version from MLflow - same as API."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting to load model '{model_name}' from registry.")
            client = mlflow.tracking.MlflowClient()
            registered_model = client.get_registered_model(model_name)
            if not registered_model.latest_versions:
                raise Exception(f"No versions found for model '{model_name}'")

            # Get the latest version (assuming highest version number is latest)
            latest_version_info = max(registered_model.latest_versions, key=lambda v: int(v.version))
            logger.info(f"Found latest version: {latest_version_info.version}")

            loaded_model = mlflow.sklearn.load_model(latest_version_info.source)
            logger.info(f"Model '{model_name}' version {latest_version_info.version} loaded successfully.")
            return loaded_model, {
                "name": latest_version_info.name,
                "version": latest_version_info.version,
                "source": latest_version_info.source,
                "run_id": latest_version_info.run_id,
                "status": latest_version_info.status,
                "current_stage": latest_version_info.current_stage
            }
        except Exception as e:
            logger.error(f"Error loading model (attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Failed to load model.")
                return None, {}
    return None, {}

def save_reference_data():
    """
    Save reference data for drift detection.
    
    This saves two separate reference files to MLflow:
    1. Feature reference data: Used to detect drift in input features 
    2. Prediction reference data: Used to detect drift in model outputs
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get the latest production model using the same method as API
        logger.info(f"Getting latest production model: {PRODUCTION_MODEL_NAME}")
        model, model_info = load_production_model(PRODUCTION_MODEL_NAME)
        
        if model is None:
            logger.error("Failed to load the production model. Cannot generate reference data.")
            return False
            
        run_id_for_artifacts = model_info["run_id"]
        logger.info(f"Will log reference artifacts to run_id: {run_id_for_artifacts}")
        
        # Load the data used for training
        logger.info("Loading training data...")
        df = load_and_clean_data()
        
        # Apply feature transformers to ensure we have all required features
        logger.info("Applying feature transformers...")
        
        # Add new features
        feature_adder = AddNewFeaturesTransformer()
        df_with_features = feature_adder.fit_transform(df)
        
        # Add age group
        age_grouper = AgeGroupTransformer()
        df_transformed = age_grouper.fit_transform(df_with_features)
        logger.info(f"Transformers applied successfully. Final shape: {df_transformed.shape}")
        
        # Split the data into features and target
        X = df_transformed.drop(columns=[TARGET_COLUMN])
        y = df_transformed[TARGET_COLUMN]
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data split complete: Train size={len(X_train)}, Test size={len(X_test)}")
        
        # Combine features and target for reference data - use transformed data
        reference_data = pd.concat([
            pd.DataFrame(X_train, columns=X_train.columns),
            pd.Series(y_train, name='Attrition') # Include target in reference data if needed
        ], axis=1)
        
        # Generate reference predictions using the production model on test set
        logger.info(f"Generating reference predictions using model version {model_info['version']}")
        reference_predictions = pd.DataFrame({
            'prediction': model.predict(X_test),
            'probability': model.predict_proba(X_test)[:, 1]
        })
        logger.info(f"Generated {len(reference_predictions)} reference predictions.")

        # Log artifacts to the original model training run
        with mlflow.start_run(run_id=run_id_for_artifacts):
            # Save directly to MLflow - temporary files are created in memory only
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet') as temp_feature_file:
                reference_data.to_parquet(temp_feature_file.name, index=False)
                mlflow.log_artifact(temp_feature_file.name, REFERENCE_FEATURES_DIR)
                logger.info(f"Logged reference data to MLflow: {REFERENCE_FEATURES_DIR}/{REFERENCE_FEATURES_FILENAME}")
            
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_pred_file:
                reference_predictions.to_csv(temp_pred_file.name, index=False)
                mlflow.log_artifact(temp_pred_file.name, REFERENCE_PREDICTIONS_DIR)
                logger.info(f"Logged reference predictions to MLflow: {REFERENCE_PREDICTIONS_DIR}/{REFERENCE_PREDICTIONS_FILENAME}")

            # Log metadata
            try:
                mlflow.log_params({
                    "reference_data_samples": len(reference_data),
                    "reference_data_features": len(reference_data.columns) - 1,
                    "reference_predictions_samples": len(reference_predictions)
                })
            except mlflow.exceptions.MlflowException as e:
                logger.warning(f"Could not log reference data params (may already exist): {e}")
            
            logger.info(f"Reference data and predictions saved successfully to MLflow run ID: {run_id_for_artifacts}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving reference data: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = save_reference_data()
    sys.exit(0 if success else 1) 