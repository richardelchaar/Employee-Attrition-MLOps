# src/employee_attrition_mlops/config.py
import os
from dotenv import load_dotenv
import logging

# --- Basic Logging Setup (Optional, but good practice) ---
# Configure logging early, though individual modules might refine it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Load environment variables from .env file located at the project root
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables set externally.")

# --- File Paths ---
# Directory to store generated reports (like confusion matrix JSON, drift reports)
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")
# Define standard filenames for artifacts logged to MLflow or saved locally
BASELINE_PROFILE_FILENAME = "training_data_profile.html" # Example, adjust if using JSON
CONFUSION_MATRIX_PLOT_FILENAME = "confusion_matrix.png" # Example plot name
ROC_CURVE_PLOT_FILENAME = "roc_curve.png" # Example plot name
FEATURE_IMPORTANCE_PLOT_FILENAME = "feature_importance.png" # Example plot name
DRIFT_REPORT_FILENAME = "drift_report.html" # Example drift report name (Evidently often uses HTML)
TEST_METRICS_FILENAME = "test_metrics_summary.json" # For saving test metrics

# --- Database Configuration ---
# Load database connection strings from environment variables
DATABASE_URL_PYODBC = os.getenv("DATABASE_URL_PYODBC")
DATABASE_URL_PYMSSQL = os.getenv("DATABASE_URL_PYMSSQL")

if not DATABASE_URL_PYODBC:
    logger.warning("DATABASE_URL_PYODBC environment variable is not set. Training/preprocessing requiring DB access might fail.")
if not DATABASE_URL_PYMSSQL:
    logger.warning("DATABASE_URL_PYMSSQL environment variable is not set. API requiring DB access might fail.")

# Keep original DATABASE_URL for potential backward compatibility or other uses? Or remove?
# Let's comment it out for now to avoid confusion.
# DATABASE_URL = os.getenv("DATABASE_URL")

# Define table names used in the project
DB_HISTORY_TABLE = "employees_history" # Table containing the historical employee data
# Add other table names if used (e.g., for prediction logs, monitoring logs)
DB_PREDICTION_LOG_TABLE = os.getenv("DB_PREDICTION_LOG_TABLE", "prediction_logs") # Table to store individual prediction logs via API
DB_BATCH_PREDICTION_TABLE = os.getenv("DB_BATCH_PREDICTION_TABLE", "batch_prediction_results") # Table for batch predictions
# DB_MONITORING_LOG_TABLE = "monitoring_logs"

# --- Data Columns ---
TARGET_COLUMN = "Attrition" # Name of the target variable in the database/dataframe
EMPLOYEE_ID_COL = "EmployeeNumber" # Unique identifier for an employee
SNAPSHOT_DATE_COL = "SnapshotDate" # Column indicating the date of the data snapshot in the DB

# Columns to drop *after* loading from DB (if they exist and are deemed unnecessary for modeling)
# These are typically constant value columns identified during EDA or ID columns not used as features.
# 'EmployeeNumber' is kept as it might be useful for joining or tracking predictions.
COLS_TO_DROP_POST_LOAD = ['EmployeeCount', 'StandardHours', 'Over18']

# Define sensitive features for fairness analysis - *** ADJUST AS NEEDED ***
# 'AgeGroup' must be created during data processing (e.g., by AgeGroupTransformer) if used here.
SENSITIVE_FEATURES = ['Gender', 'AgeGroup']

# --- Feature Engineering ---
# Mapping for ordinal encoding of BusinessTravel (used by CustomOrdinalEncoder if applied)
BUSINESS_TRAVEL_MAPPING = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
# Skewness threshold to identify columns for Log or Box-Cox transformation
SKEWNESS_THRESHOLD = 0.75

# --- Model Training ---
TEST_SIZE = 0.2 # Proportion of data to use for the test set
RANDOM_STATE = 42 # Seed for reproducibility across random operations (splits, models, etc.)
# Primary evaluation metric for HPO and model selection (must match metric name logged by training script)
# Ensure the scorer in optimize_train_select.py uses the correct positive label (e.g., 1 for 'Yes')
PRIMARY_METRIC = "f2" # Corresponds to 'test_f2' or 'f2_cv_mean' logged by MLflow

# --- Hyperparameter Optimization (Optuna) ---
HPO_N_TRIALS = 50 # Number of trials per model in Optuna study
HPO_CV_FOLDS = 5 # Folds for cross-validation within HPO objective function
HPO_EXPERIMENT_NAME = "Attrition HPO Pipeline Search (DB)" # Experiment name for HPO runs
# List of model aliases (keys in CLASSIFIER_MAP/PARAM_FUNC_MAP in optimize_train_select.py)
#MODELS_TO_OPTIMIZE = ["logistic_regression", "random_forest", "gradient_boosting", "mlp"] # Example list
MODELS_TO_OPTIMIZE = ["logistic_regression"] # Example list

# --- Monitoring & Retraining ---
# Example Drift detection thresholds (tune these based on validation/business needs)
DRIFT_FEATURE_THRESHOLD = int(os.getenv("DRIFT_FEATURE_THRESHOLD", 5)) # Number of drifted features to trigger alert/retrain
DRIFT_PREDICTION_THRESHOLD = float(os.getenv("DRIFT_PREDICTION_THRESHOLD", 0.05)) # Threshold for prediction drift metric
MONITORING_LOOKBACK_DAYS = int(os.getenv("MONITORING_LOOKBACK_DAYS", 90)) # How far back to look for reference data

# GitHub Actions Triggering (requires PAT stored securely, e.g., GitHub Secrets)
GITHUB_ACTIONS_PAT_ENV_VAR = "GH_PAT_DISPATCH" # Name of env var holding the PAT
# *** REPLACE PLACEHOLDER with your actual repo ***
GITHUB_OWNER_REPO = os.getenv("GITHUB_OWNER_REPO", "<YOUR_GITHUB_USERNAME_OR_ORG>/employee-attrition-mlops")
RETRAIN_WORKFLOW_FILENAME = os.getenv("RETRAIN_WORKFLOW_FILENAME", "train.yml") # Filename of the training workflow

# --- API / Deployment ---
API_PORT = int(os.getenv("API_PORT", 8000))
API_HOST = os.getenv("API_HOST", "0.0.0.0") # Listen on all interfaces for Docker/deployment
PRODUCTION_MODEL_NAME = "AttritionProductionModel" # Registered model name for API/deployment lookup

# --- MLflow ---
# Load MLflow tracking URI from environment or use a default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
# Note: If running MLflow server in Docker, client URI might need adjustment
# (e.g., "http://host.docker.internal:5000" from another container, or service name in docker-compose)
DEFAULT_EXPERIMENT_NAME = "Employee Attrition Default (DB)" # Default experiment if none is set explicitly

# --- Drift Detection Constants ---
DRIFT_CONFIDENCE = float(os.getenv("DRIFT_CONFIDENCE", 0.95))  # Confidence level for drift detection
DRIFT_STATTEST_THRESHOLD = float(os.getenv("DRIFT_STATTEST_THRESHOLD", 0.05))  # Statistical test threshold
RETRAIN_TRIGGER_FEATURE_COUNT = int(os.getenv("RETRAIN_TRIGGER_FEATURE_COUNT", 3))  # Number of drifted features to trigger retraining
RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE = float(os.getenv("RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE", 0.05))  # P-value threshold for dataset drift

logger.info("Configuration loaded.")
# Log key configurations (optional, avoid logging sensitive info like full DB URL)
logger.debug(f"Project Root: {PROJECT_ROOT}")
logger.debug(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
logger.debug(f"Database Table: {DB_HISTORY_TABLE}")
logger.debug(f"Target Column: {TARGET_COLUMN}")

