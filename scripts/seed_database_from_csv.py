# scripts/seed_database_from_csv.py
import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text, types as sqltypes # Import sqltypes for mapping
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import re

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Add src directory to Python path if needed for config ---
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

try:
    from employee_attrition_mlops.config import RAW_DATA_PATH
except ImportError as e:
    logger.warning(
        f"Could not import RAW_DATA_PATH from config: {e}. Using default relative path."
    )
    PROJECT_ROOT_FALLBACK = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    RAW_DATA_PATH = os.path.join(
        PROJECT_ROOT_FALLBACK,
        "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    )
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"FATAL: CSV file not found at fallback path: {RAW_DATA_PATH}")
        sys.exit(1)


def clean_column_name(col_name):
    """Cleans column names to be SQL-friendly."""
    cleaned = col_name.replace(" ", "_")
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "", cleaned)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    if not cleaned:
        raise ValueError(f"Column name '{col_name}' resulted in empty string after cleaning.")
    return cleaned


def enforce_data_types(df, sql_schema_map):
    """
    Enforces data types on DataFrame columns based on a target SQL schema map.

    Args:
        df (pd.DataFrame): The input DataFrame.
        sql_schema_map (dict): A dictionary mapping column names (lowercase)
                               to target SQLAlchemy types (e.g., sqltypes.Integer).

    Returns:
        pd.DataFrame: DataFrame with types enforced.
    """
    logger.info("Enforcing data types based on SQL schema map...")
    df_cleaned = df.copy()
    expected_cols_lower = sql_schema_map.keys()

    for col in df_cleaned.columns:
        col_lower = col.lower() # Compare using lowercase
        if col_lower not in expected_cols_lower:
            logger.warning(f"Column '{col}' found in DataFrame but not in target SQL schema map. Skipping type enforcement.")
            continue

        target_sql_type = sql_schema_map[col_lower]
        current_dtype = df_cleaned[col].dtype

        logger.debug(f"Column: '{col}', Current dtype: {current_dtype}, Target SQL Type: {target_sql_type}")

        try:
            if isinstance(target_sql_type, (sqltypes.Integer, sqltypes.BigInteger)):
                # Use pandas nullable Integer type to handle NaNs correctly
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                logger.debug(f"  Converted '{col}' to Int64 (Nullable Integer).")
            elif isinstance(target_sql_type, (sqltypes.Float, sqltypes.Numeric)):
                 # Use pandas nullable Float type
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Float64')
                logger.debug(f"  Converted '{col}' to Float64 (Nullable Float).")
            elif isinstance(target_sql_type, (sqltypes.String, sqltypes.VARCHAR, sqltypes.NVARCHAR)):
                 # Convert to string, filling NaNs with empty string or None if needed
                 # Use object type which handles None better for SQL insertion
                df_cleaned[col] = df_cleaned[col].astype(object).where(pd.notnull(df_cleaned[col]), None)
                logger.debug(f"  Converted '{col}' to object (String/None).")
            elif isinstance(target_sql_type, sqltypes.Date):
                # Convert to datetime objects then extract date part
                if col_lower == 'snapshotdate': # Special handling for the added date column
                     # Ensure it's treated as a date object for SQL
                     df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce').dt.date
                     logger.debug(f"  Ensured '{col}' is Date object.")
            # Add more type conversions if needed (Boolean, DateTime, etc.)

        except Exception as e:
            logger.error(f"Error converting column '{col}' to target type {target_sql_type}: {e}", exc_info=True)
            # Depending on severity, you might want to exit or just warn
            # sys.exit(1)

    logger.info("Data type enforcement finished.")
    return df_cleaned


def seed_database(csv_path: str, db_url: str, table_name: str, snapshot_date_str: str):
    """
    Loads data from a CSV file into a specified database table for a specific
    snapshot date. It first deletes any existing data for that specific date
    in the table and then appends the new data. Assumes the table exists.

    Args:
        csv_path (str): Path to the input CSV file.
        db_url (str): SQLAlchemy connection string for the target database.
        table_name (str): Name of the target table in the database.
        snapshot_date_str (str): The date string (YYYY-MM-DD) for the snapshot.
    """
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()
        logger.info(f"Using snapshot date: {snapshot_date}")
    except ValueError:
        logger.error("Invalid date format provided. Please use<y_bin_46>-MM-DD.")
        sys.exit(1)

    if not db_url:
        logger.error("FATAL: DATABASE_URL environment variable not set or empty.")
        sys.exit(1)

    engine = None # Initialize engine to None
    try:
        logger.info(f"Reading CSV data from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows and {len(df.columns)} columns from CSV.")

        # --- Data Preparation ---
        original_columns = df.columns.tolist()
        df.columns = [clean_column_name(col) for col in original_columns]
        cleaned_columns = df.columns.tolist()
        renamed_map = {orig: clean for orig, clean in zip(original_columns, cleaned_columns) if orig != clean}
        if renamed_map: logger.warning(f"Renamed columns for SQL compatibility: {renamed_map}")
        logger.info(f"Using cleaned DataFrame columns: {cleaned_columns}")

        df["SnapshotDate"] = snapshot_date
        logger.info(f"Added 'SnapshotDate' column with value {snapshot_date}.")

        if "EmployeeNumber" not in df.columns:
             logger.error("FATAL: Critical column 'EmployeeNumber' not found after cleaning.")
             sys.exit(1)

        target_schema = { # Define target schema map (lowercase keys)
            'employeenumber': sqltypes.Integer, 'snapshotdate': sqltypes.Date,
            'age': sqltypes.Integer, 'attrition': sqltypes.VARCHAR(10),
            'gender': sqltypes.VARCHAR(10), 'maritalstatus': sqltypes.VARCHAR(50),
            'over18': sqltypes.VARCHAR(5), 'department': sqltypes.VARCHAR(100),
            'educationfield': sqltypes.VARCHAR(100), 'joblevel': sqltypes.Integer,
            'jobrole': sqltypes.VARCHAR(100), 'businesstravel': sqltypes.VARCHAR(50),
            'distancefromhome': sqltypes.Integer, 'education': sqltypes.Integer,
            'dailyrate': sqltypes.Integer, 'hourlyrate': sqltypes.Integer,
            'monthlyincome': sqltypes.Integer, 'monthlyrate': sqltypes.Integer,
            'percentsalaryhike': sqltypes.Integer, 'stockoptionlevel': sqltypes.Integer,
            'overtime': sqltypes.VARCHAR(5), 'standardhours': sqltypes.Integer,
            'employeecount': sqltypes.Integer, 'numcompaniesworked': sqltypes.Integer,
            'totalworkingyears': sqltypes.Integer, 'trainingtimeslastyear': sqltypes.Integer,
            'yearsatcompany': sqltypes.Integer, 'yearsincurrentrole': sqltypes.Integer,
            'yearssincelastpromotion': sqltypes.Integer, 'yearswithcurrmanager': sqltypes.Integer,
            'environmentsatisfaction': sqltypes.Integer, 'jobinvolvement': sqltypes.Integer,
            'jobsatisfaction': sqltypes.Integer, 'performancerating': sqltypes.Integer,
            'relationshipsatisfaction': sqltypes.Integer, 'worklifebalance': sqltypes.Integer,
        }
        df = enforce_data_types(df, target_schema)

        # --- Database Connection & Transaction ---
        logger.info(f"Connecting to database...")
        engine = create_engine(db_url)

        # Use a transaction to ensure atomicity (delete and insert together)
        with engine.connect() as connection:
            with connection.begin(): # Start transaction
                # 1. Delete existing data for this specific snapshot date
                delete_stmt = text(f"DELETE FROM {table_name} WHERE SnapshotDate = :snapshot_date")
                logger.info(f"Executing: DELETE FROM {table_name} WHERE SnapshotDate = '{snapshot_date}'")
                result = connection.execute(delete_stmt, {"snapshot_date": snapshot_date})
                logger.info(f"Deleted {result.rowcount} existing rows for snapshot date {snapshot_date}.")

                # 2. Append new data using pandas.to_sql within the transaction
                logger.info(f"Appending {len(df)} new rows into table: '{table_name}'...")
                df.to_sql(
                    name=table_name,
                    con=connection, # Use the connection within the transaction
                    if_exists="append", # Append data, don't try to drop/create
                    index=False,
                    chunksize=1000,
                )
            # Transaction commits automatically here if no exceptions occurred
            logger.info(
                f"Successfully loaded {len(df)} rows into '{table_name}' for snapshot {snapshot_date}."
            )

    except FileNotFoundError:
        logger.error(f"FATAL: CSV file not found at path: {csv_path}")
        sys.exit(1)
    except SQLAlchemyError as db_err:
        # Catch potential database errors during delete or insert.
        logger.error(f"Database error during operation: {db_err}", exc_info=True)
        if "Invalid object name" in str(db_err) and table_name in str(db_err):
             logger.error(f"Hint: Table '{table_name}' likely does not exist. Please create it first using the SQL script.")
        elif "Violation of PRIMARY KEY constraint" in str(db_err):
             logger.error("Hint: Primary key violation (EmployeeNumber, SnapshotDate). This shouldn't happen with the DELETE first approach unless there are duplicates *within* the CSV for the same EmployeeNumber.")
        elif "String data, right truncation" in str(db_err):
             logger.error("Hint: A string value in the DataFrame is longer than the VARCHAR size defined in the DB table.")
        elif "Invalid character value for cast specification" in str(db_err) or "Error converting data type" in str(db_err):
             logger.error("Hint: A data type mismatch likely still exists. Double-check CSV values and SQL table definitions.")
             if 'df' in locals(): logger.error(f"DataFrame dtypes before sending:\n{df.dtypes}")
        else:
             logger.error("Hint: Check column names, types, constraints (NULLs?), and data values against the DB table definition.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Dispose the engine connection pool if it was created.
        if engine:
            engine.dispose()
            logger.info("Database connection pool disposed.")


if __name__ == "__main__":
    # --- Load Environment Variables ---
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Attempted to load environment variables from: {dotenv_path}")
    db_connection_url = os.getenv("DATABASE_URL")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Load Employee Attrition CSV data into Azure SQL Database for a specific snapshot date."
    )
    parser.add_argument(
        "--csv", type=str, default=RAW_DATA_PATH, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--table", type=str, default="employees_history", help="Target database table name."
    )
    parser.add_argument(
        "--date", type=str, required=True, help="Snapshot date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--db_url", type=str, default=db_connection_url, help="Database connection URL (overrides .env)."
    )
    args = parser.parse_args()
    final_db_url = args.db_url if args.db_url else db_connection_url

    seed_database(
        csv_path=args.csv,
        db_url=final_db_url,
        table_name=args.table,
        snapshot_date_str=args.date,
    )
