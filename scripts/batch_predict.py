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
import mlflow.pyfunc
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src path for imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from employee_attrition_mlops.config import (
    DATABASE_URL,
    MLFLOW_TRACKING_URI,
    PRODUCTION_MODEL_NAME,
    DB_HISTORY_TABLE,
    DB_BATCH_PREDICTION_TABLE,
    EMPLOYEE_ID_COL,
    SNAPSHOT_DATE_COL
)


def main():
    # Set MLflow tracking URI for model loading
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI

    # Load the production model
    model_uri = f"models:/{PRODUCTION_MODEL_NAME}/Production"
    logger.info(f"Loading model from '{model_uri}'")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Initialize DB engine
    engine = create_engine(DATABASE_URL)

    # Fetch the latest snapshot date
    with engine.connect() as conn:
        max_date_res = conn.execute(text(
            f"SELECT MAX({SNAPSHOT_DATE_COL}) as max_date FROM {DB_HISTORY_TABLE}"
        ))
        max_date = max_date_res.scalar()
        logger.info(f"Latest snapshot date found: {max_date}")

    # Fetch employee records for the latest snapshot
    query = text(
        f"SELECT * FROM {DB_HISTORY_TABLE} WHERE {SNAPSHOT_DATE_COL} = :snap"
    )
    df = pd.read_sql(query, engine, params={'snap': max_date})
    logger.info(f"Fetched {len(df)} rows from '{DB_HISTORY_TABLE}' for snapshot '{max_date}'")

    # Prepare feature DataFrame (drop ID and snapshot columns)
    features = df.drop(columns=[EMPLOYEE_ID_COL, SNAPSHOT_DATE_COL], errors='ignore')

    # Generate predictions
    try:
        preds = model.predict(features)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

    # Build results DataFrame
    results = pd.DataFrame({
        EMPLOYEE_ID_COL: df[EMPLOYEE_ID_COL],
        SNAPSHOT_DATE_COL: df[SNAPSHOT_DATE_COL],
        'Prediction': preds
    })

    # Ensure batch prediction table exists
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {DB_BATCH_PREDICTION_TABLE} (
        {EMPLOYEE_ID_COL} INT,
        {SNAPSHOT_DATE_COL} VARCHAR(50),
        Prediction VARCHAR(50),
        PRIMARY KEY ({EMPLOYEE_ID_COL}, {SNAPSHOT_DATE_COL})
    )
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        logger.info(f"Ensured batch prediction table '{DB_BATCH_PREDICTION_TABLE}' exists.")

        # Remove existing predictions for this snapshot
        delete_sql = text(
            f"DELETE FROM {DB_BATCH_PREDICTION_TABLE} WHERE {SNAPSHOT_DATE_COL} = :snap"
        )
        deleted = conn.execute(delete_sql, {'snap': max_date}).rowcount
        logger.info(f"Deleted {deleted} existing records for snapshot '{max_date}' in '{DB_BATCH_PREDICTION_TABLE}'")

        # Insert new batch predictions
        insert_sql = text(
            f"INSERT INTO {DB_BATCH_PREDICTION_TABLE} ({EMPLOYEE_ID_COL}, {SNAPSHOT_DATE_COL}, Prediction)"
            " VALUES (:emp, :snap, :pred)"
        )
        records = [
            {'emp': int(emp), 'snap': str(snap), 'pred': str(pred)}
            for emp, snap, pred in zip(
                results[EMPLOYEE_ID_COL], results[SNAPSHOT_DATE_COL], results['Prediction']
            )
        ]
        conn.execute(insert_sql, records)
        logger.info(f"Inserted {len(records)} batch predictions into '{DB_BATCH_PREDICTION_TABLE}'.")


if __name__ == '__main__':
    main() 