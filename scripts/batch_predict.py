#!/usr/bin/env python3
"""
Batch prediction script:
- Loads the Production model from MLflow Registry
- Fetches the latest snapshot of employees from the database
- Generates predictions and writes to the batch_prediction_results table
"""
import os
import sys
import logging
import pandas as pd
import mlflow.sklearn
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src path for imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from employee_attrition_mlops.config import (
    DATABASE_URL_PYODBC as DATABASE_URL,
    MLFLOW_TRACKING_URI,
    PRODUCTION_MODEL_NAME,
    DB_HISTORY_TABLE,
    DB_BATCH_PREDICTION_TABLE,
    EMPLOYEE_ID_COL,
    SNAPSHOT_DATE_COL
)
# Import necessary transformers
from employee_attrition_mlops.data_processing import AgeGroupTransformer, AddNewFeaturesTransformer

def main():
    # Set MLflow tracking URI for model loading
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Load the latest version of the main preprocessing+model pipeline
    logger.info(f"Loading latest version of pipeline '{PRODUCTION_MODEL_NAME}'")
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

    # Initialize DB engine using the imported PYODBC URL
    if not DATABASE_URL:
         logger.error("DATABASE_URL_PYODBC is not configured. Cannot run batch predictions.")
         sys.exit(1)
    logger.info("Connecting to database using pyodbc driver (URL from config).")
    engine = create_engine(DATABASE_URL)

    # Fetch the latest snapshot date
    with engine.connect() as conn:
        max_date_res = conn.execute(text(f"SELECT MAX({SNAPSHOT_DATE_COL}) as max_date FROM {DB_HISTORY_TABLE}"))
        max_date = max_date_res.scalar()
        logger.info(f"Latest snapshot date found: {max_date}")

    # Fetch employee records for the latest snapshot (raw data)
    query = text(f"SELECT * FROM {DB_HISTORY_TABLE} WHERE {SNAPSHOT_DATE_COL} = :snap")
    df = pd.read_sql(query, engine, params={'snap': max_date})
    logger.info(f"Fetched {len(df)} rows from '{DB_HISTORY_TABLE}' for snapshot '{max_date}'")

    # --- Apply Initial Custom Transformations MANUALLY ---
    # These steps are assumed to be missing from the saved pipeline artifact
    try:
        logger.info("Applying initial custom transformations (AddNewFeatures, AgeGroup)...")
        # Apply AddNewFeaturesTransformer FIRST
        feature_adder = AddNewFeaturesTransformer()
        df_with_new_features = feature_adder.fit_transform(df)
        logger.info(f"Shape after AddNewFeatures: {df_with_new_features.shape}")

        # Apply AgeGroupTransformer NEXT
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
        logger.info(f"Predictions generated successfully.")
    except Exception as e:
        logger.error(f"Prediction failed using main pipeline: {e}", exc_info=True)
        sys.exit(1)
    # ------------------------------------------------------

    # Build results DataFrame (use original df for IDs)
    results = pd.DataFrame({
        EMPLOYEE_ID_COL: df[EMPLOYEE_ID_COL],
        SNAPSHOT_DATE_COL: df[SNAPSHOT_DATE_COL],
        'Prediction': preds
    })

    # --- Database writing logic --- 
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
                Prediction VARCHAR(50),
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
        records = [{'emp': int(emp), 'snap': str(snap), 'pred': str(pred)} for emp, snap, pred in zip(results[EMPLOYEE_ID_COL], results[SNAPSHOT_DATE_COL], results['Prediction'])]
        conn.execute(insert_sql, records)
        logger.info(f"Inserted {len(records)} batch predictions into '{DB_BATCH_PREDICTION_TABLE}'.")


if __name__ == '__main__':
    main() 