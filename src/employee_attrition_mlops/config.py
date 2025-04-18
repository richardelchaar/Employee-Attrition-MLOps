# src/employee_attrition_mlops/config.py
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# --- Environment Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Consider using python-dotenv for sensitive info or env-specific paths
# from dotenv import load_dotenv
# load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# --- File Paths ---
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/01_raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "data/08_reporting/")
BASELINE_PROFILE_FILENAME = "training_data_profile.json"
DRIFT_REPORT_FILENAME = "drift_report.json"

# --- Data Columns ---
TARGET_COLUMN = "Attrition"
# Columns to drop immediately after loading
INITIAL_COLS_TO_DROP = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
# Define sensitive features for fairness analysis - *** ADJUST AS NEEDED ***
SENSITIVE_FEATURES = ['Gender', 'AgeGroup'] # 'AgeGroup' must be created in data_processing

# --- Feature Engineering ---
BUSINESS_TRAVEL_MAPPING = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
# Skewness threshold for transformations
SKEWNESS_THRESHOLD = 0.75

# --- Model Training ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Primary evaluation metric for HPO and model selection
PRIMARY_METRIC = "f2" # Corresponds to 'test_f2' logged by MLflow in train script

# --- Hyperparameter Optimization (Optuna) ---
# Define search spaces for Optuna within hpo.py, not statically here.
HPO_N_TRIALS = 50 # Number of trials for Optuna study
HPO_CV_FOLDS = 5 # Folds for cross-validation within HPO objective function
HPO_EXPERIMENT_NAME = "Attrition HPO Pipeline Search"
# Aliases for models to optimize
MODELS_TO_OPTIMIZE = ["logistic_regression", "random_forest"] # Add others like "gradient_boosting"

# --- Monitoring & Retraining ---
# Drift detection thresholds (using Evidently defaults or define custom)
# E.g., for DataDriftPreset
DRIFT_CONFIDENCE = 0.95
DRIFT_STATTEST_THRESHOLD = 0.05 # Default p-value for statistical tests

# Threshold for number of drifting features to trigger retraining
RETRAIN_TRIGGER_FEATURE_COUNT = 5 # Example: Retrain if 5 or more features drift
# Trigger based on overall dataset drift p-value from Evidently
RETRAIN_TRIGGER_DATASET_DRIFT_P_VALUE = 0.05

# GitHub Actions Triggering (requires PAT)
GITHUB_ACTIONS_PAT_ENV_VAR = "GH_PAT_DISPATCH" # Name of env var holding the PAT
GITHUB_OWNER_REPO = "<YOUR_GITHUB_USERNAME>/employee-attrition-mlops" # *** REPLACE PLACEHOLDER ***
RETRAIN_WORKFLOW_ID = "retrain.yml" # Filename of the retraining workflow

# --- API / Deployment ---
API_PORT = 8000
API_HOST = "0.0.0.0"
PRODUCTION_MODEL_NAME = "AttritionProductionModel" # Registered model name for API

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# Docker needs adjusted URI: "http://host.docker.internal:5000" or service name
DEFAULT_EXPERIMENT_NAME = "Employee Attrition Default"
