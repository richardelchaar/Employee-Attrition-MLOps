#!/usr/bin/env python3
"""
Batch prediction script:
- Loads the Production model from MLflow Registry
- Fetches the latest snapshot of employees from the database
- Applies necessary transformations
- Generates predictions and writes to the batch_prediction_results table
- Saves features and predictions to JSON files for drift checks
"""
import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from sqlalchemy import create_engine, text

# Add src to Python path to allow imports from employee_attrition_mlops
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ''))
from src.employee_attrition_mlops.data_processing import (
    AddNewFeaturesTransformer, AgeGroupTransformer, load_and_clean_data_from_db
)
from src.employee_attrition_mlops.config import (
    MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME, TARGET_COLUMN,
    DB_HISTORY_TABLE, DB_BATCH_PREDICTION_TABLE, SNAPSHOT_DATE_COL,
    EMPLOYEE_ID_COL, REPORTS_PATH, DATABASE_URL_PYMSSQL, DATABASE_URL_PYODBC
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set MLflow tracking URI for model loading
    if MLFLOW_TRACKING_URI:
        logger.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    else:
        logger.warning("MLFLOW_TRACKING_URI is not configured. Using default.")

    # Load the production model from MLflow
    logger.info(f"Loading the production model from MLflow registry: {PRODUCTION_MODEL_NAME}")
    client = mlflow.tracking.MlflowClient()
    try:
        registered_model = client.get_registered_model(PRODUCTION_MODEL_NAME)
        if not registered_model.latest_versions:
            raise Exception(f"No versions found for model '{PRODUCTION_MODEL_NAME}'")
        latest_version_info = max(registered_model.latest_versions, key=lambda v: int(v.version))
        pipeline_uri = latest_version_info.source
        logger.info(f"Loading pipeline version {latest_version_info.version} from source: {pipeline_uri}")
        pipeline = mlflow.sklearn.load_model(pipeline_uri)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Load the data using our helper function which handles Docker vs. local environment
    logger.info("Loading latest data from database...")
    df_all = load_and_clean_data_from_db(table_name=DB_HISTORY_TABLE)
    
    if df_all is None:
        logger.error("Failed to load data from database. Exiting.")
        sys.exit(1)
    
    # Get the latest snapshot date
    if SNAPSHOT_DATE_COL in df_all.columns:
        max_date = df_all[SNAPSHOT_DATE_COL].max()
        logger.info(f"Latest snapshot date found: {max_date}")
        
        # Filter for only the latest snapshot data and current employees (Attrition = 0)
        df = df_all[(df_all[SNAPSHOT_DATE_COL] == max_date) & (df_all[TARGET_COLUMN] == 0)]
        logger.info(f"Filtered to {len(df)} current employees from '{DB_HISTORY_TABLE}' for snapshot '{max_date}'")
    else:
        logger.error(f"Snapshot date column '{SNAPSHOT_DATE_COL}' not found in data. Exiting.")
        sys.exit(1)

    if len(df) == 0:
        logger.warning("No current employees found in the latest snapshot. Exiting.")
        # Save empty files to avoid workflow errors
        os.makedirs(REPORTS_PATH, exist_ok=True)
        with open(os.path.join(REPORTS_PATH, 'batch_features.json'), 'w') as f: json.dump([], f)
        with open(os.path.join(REPORTS_PATH, 'batch_predictions.json'), 'w') as f: json.dump([], f)
        with open(os.path.join(REPORTS_PATH, 'batch_prediction_summary.json'), 'w') as f: 
            json.dump({'num_predictions': 0, 'attrition_rate': 0.0}, f)
        sys.exit(0)

    # --- Apply Initial Custom Transformations MANUALLY ---
    # These steps are assumed to be missing from the saved pipeline artifact
    try:
        logger.info("Applying initial custom transformations (AddNewFeatures, AgeGroup)...")
        feature_adder = AddNewFeaturesTransformer()
        df_with_new_features = feature_adder.fit_transform(df)
        logger.info(f"Shape after AddNewFeatures: {df_with_new_features.shape}")

        age_grouper = AgeGroupTransformer()
        df_transformed = age_grouper.fit_transform(df_with_new_features) # Apply to output of previous step
        logger.info(f"Shape after AgeGroup: {df_transformed.shape}")
    except Exception as e:
        logger.error(f"Error applying initial custom transformations: {e}", exc_info=True)
        sys.exit(1)
    # ------------------------------------------------------

    # --- Generate predictions using the main pipeline ---
    # Pass the DataFrame with BOTH sets of new features to the main pipeline.
    try:
        logger.info(f"Generating predictions using loaded pipeline on DataFrame with shape {df_transformed.shape}...")
        preds = pipeline.predict(df_transformed) # Pass the fully transformed data
        # Also get probabilities for prediction drift checks
        probs = pipeline.predict_proba(df_transformed)[:, 1] 
        logger.info(f"Predictions generated successfully.")
    except Exception as e:
        logger.error(f"Prediction failed using main pipeline: {e}", exc_info=True)
        sys.exit(1)
    # ------------------------------------------------------

    # Build results DataFrame for DB (use original df for IDs)
    db_results = pd.DataFrame({
        EMPLOYEE_ID_COL: df[EMPLOYEE_ID_COL],
        SNAPSHOT_DATE_COL: df[SNAPSHOT_DATE_COL],
        'Prediction': preds
    })
    
    # Build prediction DataFrame for drift check file
    drift_predictions = pd.DataFrame({
        'prediction': preds,
        'probability': probs
    })
    
    # --- Save Features and Predictions to Files --- 
    os.makedirs(REPORTS_PATH, exist_ok=True)
    features_path = os.path.join(REPORTS_PATH, 'batch_features.json')
    predictions_path = os.path.join(REPORTS_PATH, 'batch_predictions.json')
    summary_path = os.path.join(REPORTS_PATH, 'batch_prediction_summary.json')

    try:
        # Convert NaNs or other non-serializable types if necessary before saving
        df_transformed_serializable = df_transformed.replace({pd.NA: None, np.nan: None})
        df_transformed_serializable.to_json(features_path, orient='records', indent=2)
        logger.info(f"Saved batch features to {features_path}")
        
        drift_predictions.to_json(predictions_path, orient='records', indent=2)
        logger.info(f"Saved batch predictions to {predictions_path}")

        # Calculate summary stats
        num_predictions = len(db_results)
        attrition_rate = (db_results['Prediction'] == 1).mean() # Assuming 1 means Attrition=Yes
        summary_data = {
            'num_predictions': num_predictions,
            'attrition_rate': attrition_rate
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Saved batch summary to {summary_path}")

    except Exception as e:
        logger.error(f"Failed to save features/predictions/summary to JSON files: {e}", exc_info=True)
        sys.exit(1) # Exit if we can't save files needed by workflow
    # --------------------------------------------------

    # --- Database writing logic --- 
    # Get a database connection using the same approach as our load function
    # This ensures we use the right driver based on environment (Docker vs local)
    in_docker = os.path.exists("/.dockerenv")
    connection_string = None
    
    if in_docker:
        if not DATABASE_URL_PYMSSQL:
            logger.error("DATABASE_URL_PYMSSQL not configured for Docker environment.")
            sys.exit(1)
        connection_string = DATABASE_URL_PYMSSQL
        driver_name = "pymssql"
        logger.info("Using pymssql connection for database writes (Docker environment)")
    else:
        if not DATABASE_URL_PYODBC:
            logger.error("DATABASE_URL_PYODBC not configured for local environment.")
            sys.exit(1)
        connection_string = DATABASE_URL_PYODBC
        driver_name = "pyodbc"
        logger.info("Using pyodbc connection for database writes (local environment)")
    
    # Create database engine
    try:
        engine = create_engine(connection_string)
        logger.info(f"Successfully created SQLAlchemy engine using {driver_name} for database writes.")
    except Exception as e:
        logger.error(f"Failed to create database engine for writing results: {e}", exc_info=True)
        logger.warning("Prediction results won't be written to the database, but files were saved.")
        sys.exit(1)
    
    try:
        with engine.begin() as conn:
            # Check if batch prediction table exists using INFORMATION_SCHEMA
            table_check_sql = text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = :table_name")
            result = conn.execute(table_check_sql, {"table_name": DB_BATCH_PREDICTION_TABLE}).fetchone()

            if not result:
                # Table does not exist, so create it (without IF NOT EXISTS)
                logger.info(f"Table '{DB_BATCH_PREDICTION_TABLE}' does not exist. Creating it...")
                create_table_sql = text(f"""
                CREATE TABLE {DB_BATCH_PREDICTION_TABLE} (
                    {EMPLOYEE_ID_COL} INT,
                    {SNAPSHOT_DATE_COL} VARCHAR(50),
                    Prediction INT, -- Changed to INT to store 0 or 1
                    PRIMARY KEY ({EMPLOYEE_ID_COL}, {SNAPSHOT_DATE_COL})
                )
                """)
                conn.execute(create_table_sql)
                logger.info(f"Successfully created table '{DB_BATCH_PREDICTION_TABLE}'.")
            else:
                logger.info(f"Batch prediction table '{DB_BATCH_PREDICTION_TABLE}' already exists.")

            # Remove existing predictions for this snapshot
            delete_sql = text(f"DELETE FROM {DB_BATCH_PREDICTION_TABLE} WHERE {SNAPSHOT_DATE_COL} = :snap")
            deleted = conn.execute(delete_sql, {'snap': max_date}).rowcount
            logger.info(f"Deleted {deleted} existing records for snapshot '{max_date}' in '{DB_BATCH_PREDICTION_TABLE}'")

            # Insert new batch predictions
            insert_sql = text(f"INSERT INTO {DB_BATCH_PREDICTION_TABLE} ({EMPLOYEE_ID_COL}, {SNAPSHOT_DATE_COL}, Prediction) VALUES (:emp, :snap, :pred)")
            # Ensure prediction is integer for DB
            records = [{'emp': int(emp), 'snap': str(snap), 'pred': int(pred)} for emp, snap, pred in zip(db_results[EMPLOYEE_ID_COL], db_results[SNAPSHOT_DATE_COL], db_results['Prediction'])]
            conn.execute(insert_sql, records)
            logger.info(f"Inserted {len(records)} batch predictions into '{DB_BATCH_PREDICTION_TABLE}'.")
            
        logger.info("Database operations completed successfully.")
    except Exception as e:
        logger.error(f"Database operations failed: {e}", exc_info=True)
        logger.warning("Prediction results were not written to the database, but files were saved.")
    finally:
        if engine:
            engine.dispose()
            logger.info(f"Database engine ({driver_name}) disposed.")

    logger.info("Batch prediction process completed successfully.")

if __name__ == '__main__':
    main() 