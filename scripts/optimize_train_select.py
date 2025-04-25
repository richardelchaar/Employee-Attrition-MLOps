# scripts/optimize_train_select.py
import argparse
import os
import sys
import time
import logging
import json
import warnings
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (make_scorer, fbeta_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)
# --- Import Model Classes ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, SelectFromModel # Import feature selectors
from sklearn.base import BaseEstimator # Import BaseEstimator
# --- Import Imblearn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE # Import SMOTE explicitly if used

# --- Add Fairness & Explainability Imports ---
import matplotlib.pyplot as plt
import shap
from fairlearn.metrics import MetricFrame, count, selection_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# --- Data Profiling ---
from sklearn.metrics import RocCurveDisplay
try:
    # Import from the NEW library name
    from ydata_profiling import ProfileReport
except ImportError:
    # Update warning message
    logger.warning("ydata-profiling not found. Skipping data profile generation.")
    ProfileReport = None # Define as None if not available

# --- Add src directory to Python path ---
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
logger = logging.getLogger(__name__) # Define logger early
logger.info(f"Adding '{SRC_PATH}' to sys.path")

# --- Import from local modules ---
try:
    # Import configurations and utility functions
    from employee_attrition_mlops.config import (
        TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
        MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME,
        HPO_N_TRIALS, HPO_CV_FOLDS, PRIMARY_METRIC,
        MODELS_TO_OPTIMIZE,
        REPORTS_PATH, SKEWNESS_THRESHOLD,
        DATABASE_URL_PYODBC,
        DB_HISTORY_TABLE,
        SENSITIVE_FEATURES, # Import sensitive features list
        FEATURE_IMPORTANCE_PLOT_FILENAME, # Import filenames for artifacts
        # Add other necessary filenames from config if needed
    )
    from employee_attrition_mlops.data_processing import (
        load_and_clean_data_from_db,
        identify_column_types, find_skewed_columns,
        AddNewFeaturesTransformer, AgeGroupTransformer # Ensure AgeGroupTransformer is imported
    )
    from employee_attrition_mlops.pipelines import create_preprocessing_pipeline, create_full_pipeline
    from employee_attrition_mlops.utils import save_json, load_json
    logger.info("Successfully imported local modules.")
except ImportError as e:
    logger.error(f"Error importing project modules: {e}", exc_info=True)
    logger.error(f"Ensure PYTHONPATH includes '{SRC_PATH}' or run from the project root.")
    sys.exit(1)
except NameError as e:
     logger.error(f"NameError during import: {e}. Check if all required variables are defined in config.py", exc_info=True)
     sys.exit(1)


# --- Setup Logging ---
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# REMOVED the problematic line: warnings.filterwarnings('ignore', category=shap.errors.ExplainerError)

# --- Define Scorer ---
beta_value = 2
if TARGET_COLUMN == 'Attrition':
    pos_label = 1
else:
     pos_label = 1
     logger.warning(f"TARGET_COLUMN ('{TARGET_COLUMN}') not explicitly handled for pos_label. Assuming positive class is 1.")

primary_scorer = make_scorer(fbeta_score, beta=beta_value, pos_label=pos_label, zero_division=0)
# Define other metrics for fairness analysis
metrics_dict = {
    'accuracy': accuracy_score,
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    f'{PRIMARY_METRIC}': lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=beta_value, pos_label=pos_label, zero_division=0),
    'selection_rate': selection_rate, # From fairlearn
    'count': count # From fairlearn
    }
logger.info(f"Using primary metric: {PRIMARY_METRIC} (F-beta with beta={beta_value}, pos_label={pos_label})")
logger.info(f"Fairness metrics to compute: {list(metrics_dict.keys())}")


# === Define Classifier Map Locally ===
CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
}
logger.info(f"Defined CLASSIFIER_MAP locally for models: {list(CLASSIFIER_MAP.keys())}")


# === HPO Parameter Function Definitions ===
# (Keep existing get_logreg_params, get_rf_params, get_gb_params, get_mlp_params functions as they are)
def get_logreg_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for Logistic Regression."""
    solver = trial.suggest_categorical("model_solver", ["liblinear", "saga"])
    if solver == 'liblinear':
        penalty = trial.suggest_categorical("model_penalty_liblinear", ["l1", "l2"])
    else: # saga solver
        penalty = trial.suggest_categorical("model_penalty_saga", ["l1", "l2", "elasticnet", None])

    params = {
        "C": trial.suggest_float("model_C", 1e-4, 1e2, log=True),
        "solver": solver,
        "penalty": penalty,
        "max_iter": trial.suggest_int("model_max_iter", 300, 1500),
        "class_weight": trial.suggest_categorical("model_class_weight", [None, "balanced"]),
        "random_state": RANDOM_STATE,
    }
    if solver == 'saga' and penalty == 'elasticnet':
         params['l1_ratio'] = trial.suggest_float("model_l1_ratio", 0, 1)
    return params

def get_rf_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for Random Forest."""
    return {
        "n_estimators": trial.suggest_int("model_n_estimators", 50, 500),
        "max_depth": trial.suggest_int("model_max_depth", 3, 20, step=1, log=False),
        "min_samples_split": trial.suggest_int("model_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("model_min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("model_max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("model_class_weight", [None, "balanced", "balanced_subsample"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

def get_gb_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for Gradient Boosting."""
    return {
        "n_estimators": trial.suggest_int("model_n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("model_learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("model_max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("model_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("model_min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("model_subsample", 0.6, 1.0),
        "max_features": trial.suggest_categorical("model_max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE,
    }

def get_mlp_params(trial: optuna.Trial) -> dict:
     """Suggest hyperparameters for MLP Classifier."""
     n_layers = trial.suggest_int("model_n_layers", 1, 3)
     layers = []
     for i in range(n_layers):
          layers.append(trial.suggest_int(f"model_n_units_l{i}", 32, 256))
     hidden_layer_sizes = tuple(layers)

     return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("model_activation", ["relu", "tanh", "logistic"]),
        "solver": trial.suggest_categorical("model_solver", ["adam", "sgd"]),
        "alpha": trial.suggest_float("model_alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("model_learning_rate_init", 1e-5, 1e-2, log=True),
        "max_iter": trial.suggest_int("model_max_iter", 200, 1000),
        "early_stopping": trial.suggest_categorical("model_early_stopping", [True, False]),
        "random_state": RANDOM_STATE,
    }

PARAM_FUNC_MAP = {
    "logistic_regression": get_logreg_params,
    "random_forest": get_rf_params,
    "gradient_boosting": get_gb_params,
    "mlp": get_mlp_params,
}
logger.info(f"Defined PARAM_FUNC_MAP locally for models: {list(PARAM_FUNC_MAP.keys())}")


# === Optuna Objective Function ===
# (Keep existing optuna_objective function as it is - it focuses on HPO)
def optuna_objective(trial: optuna.Trial, model_type: str, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    Builds and evaluates a pipeline using cross-validation.
    Logs results to a nested MLflow run for each trial.
    """
    # Start a nested MLflow run for this specific trial
    with mlflow.start_run(run_name=f"Trial_{trial.number}_{model_type}", nested=True) as trial_run:
        mlflow.set_tag("optuna_trial_number", str(trial.number))
        if trial.study: mlflow.set_tag("optuna_study_name", trial.study.study_name)
        mlflow.set_tag("model_type", model_type)
        logger.debug(f"Starting Optuna Trial {trial.number} for {model_type}")

        try:
            # --- Suggest Parameters (Pipeline & Model) ---
            numeric_transformer_type = trial.suggest_categorical("pipe_num_transform", ['log', 'boxcox', 'passthrough'])
            numeric_scaler_type = trial.suggest_categorical("pipe_num_scaler", ['standard', 'minmax', 'passthrough'])
            business_encoder_type = trial.suggest_categorical("pipe_bt_encoder", ['ordinal', 'onehot'])
            feature_selector_type = trial.suggest_categorical("pipe_selector", ['rfe', 'lasso', 'tree', 'passthrough'])
            smote_active = trial.suggest_categorical("pipe_smote", [True, False])

            feature_selector_params = {}
            if feature_selector_type == 'rfe':
                # Ensure max_rfe_features is at least 1 if X_train has only 1 column (edge case)
                max_rfe_features = max(1, X_train.shape[1] - 1)
                # Ensure lower bound is not higher than upper bound
                lower_bound_rfe = 5
                upper_bound_rfe = max(lower_bound_rfe, max_rfe_features)
                feature_selector_params['n_features_to_select'] = trial.suggest_int("selector_rfe_n", lower_bound_rfe, upper_bound_rfe)
            elif feature_selector_type == 'lasso':
                feature_selector_params['C'] = trial.suggest_float("selector_lasso_C", 1e-3, 1e1, log=True)
            elif feature_selector_type == 'tree':
                feature_selector_params['threshold'] = trial.suggest_categorical("selector_tree_thresh", ['median', 'mean', '1.25*mean'])

            if model_type not in PARAM_FUNC_MAP: raise ValueError(f"Unsupported model_type: {model_type}")
            if model_type not in CLASSIFIER_MAP: raise ValueError(f"Classifier class not defined: {model_type}")

            model_params = PARAM_FUNC_MAP[model_type](trial)
            classifier_class = CLASSIFIER_MAP[model_type]

            params_to_log = {k:v for k,v in trial.params.items() if isinstance(v, (str, int, float, bool))}
            mlflow.log_params(params_to_log)
            mlflow.log_param("smote_active", smote_active)

            # --- Prepare Data for Pipeline (Identify column types and skewness) ---
            try:
                if not isinstance(X_train, pd.DataFrame): raise TypeError("X_train must be a pandas DataFrame.")
                # Need AgeGroup for column identification if it's a sensitive feature
                # Apply necessary transformers *before* identifying types for the pipeline build
                temp_age_grouper = AgeGroupTransformer()
                X_train_temp_age = temp_age_grouper.fit_transform(X_train.copy())

                feature_adder = AddNewFeaturesTransformer()
                X_train_eng_temp = feature_adder.fit_transform(X_train_temp_age) # Apply feature adding

                col_types = identify_column_types(X_train_eng_temp, None) # Identify types on engineered data
                skewed_cols = find_skewed_columns(X_train_eng_temp, col_types['numerical'], threshold=SKEWNESS_THRESHOLD)
                del X_train_eng_temp, X_train_temp_age # Clean up intermediate dataframes
            except Exception as data_proc_err:
                 logger.error(f"Trial {trial.number}: Feature engineering or column ID failed: {data_proc_err}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED_DATA_PROC")
                 mlflow.log_param("error", str(data_proc_err)[:250])
                 return -1.0

            # --- Create Preprocessing and Full Pipeline ---
            try:
                # Pass the identified column types to the preprocessor builder
                preprocessor = create_preprocessing_pipeline(
                    numerical_cols=col_types['numerical'],
                    categorical_cols=col_types['categorical'],
                    ordinal_cols=col_types['ordinal'],
                    business_travel_col=col_types['business_travel'],
                    skewed_cols=skewed_cols,
                    numeric_transformer_type=numeric_transformer_type,
                    numeric_scaler_type=numeric_scaler_type,
                    business_encoder_type=business_encoder_type,
                )

                # Build the full pipeline including the preprocessor
                full_pipeline = create_full_pipeline(
                    classifier_class=classifier_class,
                    model_params=model_params,
                    preprocessor=preprocessor, # Pass the created preprocessor
                    feature_selector_type=feature_selector_type,
                    feature_selector_params=feature_selector_params,
                    smote_active=smote_active
                )
                logger.debug(f"Trial {trial.number}: Pipeline created successfully.")
            except Exception as pipe_err:
                 logger.error(f"Trial {trial.number}: Pipeline creation failed: {pipe_err}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED_PIPE_CREATION")
                 mlflow.log_param("error", str(pipe_err)[:250])
                 return -1.0

            # --- Cross-validation ---
            try:
                cv = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                logger.debug(f"Trial {trial.number}: Starting {HPO_CV_FOLDS}-fold cross-validation...")
                # Pass the original X_train here, as the full_pipeline handles internal transformations
                scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring=primary_scorer, n_jobs=1, error_score='raise')
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                logger.debug(f"Trial {trial.number}: CV Scores: {scores}, Mean: {mean_score:.4f}, Std: {std_score:.4f}")

                mlflow.log_metric(f"{PRIMARY_METRIC}_cv_mean", mean_score)
                mlflow.log_metric(f"{PRIMARY_METRIC}_cv_std", std_score)
                mlflow.set_tag("optuna_trial_state", "COMPLETE")

                return mean_score

            except optuna.exceptions.TrialPruned as e:
                 logger.info(f"Trial {trial.number} pruned: {e}")
                 mlflow.set_tag("optuna_trial_state", "PRUNED")
                 raise e
            except Exception as cv_err:
                 logger.error(f"Trial {trial.number} failed during CV: {cv_err}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED_CV")
                 mlflow.log_param("error", str(cv_err)[:250])
                 if "Solver " in str(cv_err) and "supports " in str(cv_err):
                      logger.warning(f"Potential incompatible solver/penalty combination suggested in trial {trial.number}.")
                 return -1.0

        except Exception as outer_err:
             logger.error(f"Unexpected error in Trial {trial.number} objective: {outer_err}", exc_info=True)
             try:
                 mlflow.set_tag("optuna_trial_state", "FAILED_UNEXPECTED")
                 mlflow.log_param("error", str(outer_err)[:250])
             except Exception as log_err:
                  logger.error(f"Failed to log unexpected trial error to MLflow: {log_err}")
             return -1.0


# === Main Execution Function ===
def optimize_select_and_train(models_to_opt: list):
    """
    Orchestrates HPO, model selection, final training, evaluation, fairness/XAI analysis, and registration.
    Loads data from the database.
    """
    logger.info("Starting optimization and training process (loading from Database)...")

    # --- MLflow Setup ---
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Set MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not found in config. Using default MLflow tracking.")
    mlflow.set_experiment("Attrition Optimization and Evaluation (DB)") # Updated experiment name
    logger.info("Set MLflow experiment to 'Attrition Optimization and Evaluation (DB)'")

    best_overall_score = -1.0
    best_optuna_config = None
    best_model_type_name = None

    # Start the main parent MLflow run
    with mlflow.start_run(run_name="DB_MultiModel_Optimize_Train_Evaluate") as parent_run: # Updated run name
        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent MLflow Run ID: {parent_run_id}")
        mlflow.log_param("data_source", "database")
        mlflow.log_param("db_table", DB_HISTORY_TABLE)
        mlflow.log_param("optimization_method", "Optuna")
        mlflow.log_param("models_considered", ", ".join(models_to_opt))
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("optuna_trials_per_model", HPO_N_TRIALS)
        mlflow.log_param("cv_folds", HPO_CV_FOLDS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("sensitive_features", ", ".join(SENSITIVE_FEATURES))

        # --- Data Loading and Preparation ---
        X_train, X_test, y_train, y_test = None, None, None, None # Initialize
        try:
            logger.info("Loading data from database...")
            if not DATABASE_URL_PYODBC:
                 logger.error("FATAL: DATABASE_URL_PYODBC environment variable not set.")
                 mlflow.set_tag("status", "FAILED_DB_CONFIG")
                 mlflow.end_run("FAILED")
                 sys.exit(1)

            df_raw = load_and_clean_data_from_db(table_name=DB_HISTORY_TABLE)

            # --- Create AgeGroup Feature ---
            # This needs to be done *before* splitting if AgeGroup is a sensitive feature
            # and used for fairness, as fairlearn needs the group identifier.
            if 'AgeGroup' in SENSITIVE_FEATURES:
                 logger.info("Creating 'AgeGroup' feature before splitting...")
                 age_grouper = AgeGroupTransformer()
                 df_raw = age_grouper.fit_transform(df_raw)
                 if 'AgeGroup' not in df_raw.columns:
                      logger.error("FATAL: AgeGroupTransformer failed to add 'AgeGroup' column.")
                      mlflow.set_tag("status", "FAILED_AGEGROUP_CREATION")
                      mlflow.end_run("FAILED")
                      sys.exit(1)
                 logger.info(f"Unique AgeGroups created: {df_raw['AgeGroup'].unique()}")


            # --- Process Target Column ---
            if TARGET_COLUMN not in df_raw.columns:
                 logger.error(f"FATAL: Target column '{TARGET_COLUMN}' not found.")
                 mlflow.set_tag("status", "FAILED_TARGET_MISSING")
                 mlflow.end_run("FAILED")
                 sys.exit(1)

            if df_raw[TARGET_COLUMN].dtype == 'object':
                logger.info(f"Encoding target column '{TARGET_COLUMN}' from Yes/No to 1/0.")
                unique_targets = df_raw[TARGET_COLUMN].unique()
                if not all(val in ['Yes', 'No'] for val in unique_targets if pd.notna(val)):
                     logger.warning(f"Target column contains values other than 'Yes'/'No': {unique_targets}.")
                df_raw[TARGET_COLUMN] = df_raw[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
                if df_raw[TARGET_COLUMN].isnull().any():
                     nan_count = df_raw[TARGET_COLUMN].isnull().sum()
                     logger.warning(f"{nan_count} NaNs found in target after mapping. Dropping rows.")
                     initial_count = len(df_raw)
                     df_raw.dropna(subset=[TARGET_COLUMN], inplace=True)
                     logger.warning(f"Dropped {initial_count - len(df_raw)} rows.")
                     if len(df_raw) == 0: raise ValueError("All rows dropped after target handling.")
            elif pd.api.types.is_numeric_dtype(df_raw[TARGET_COLUMN]):
                 logger.info(f"Target column '{TARGET_COLUMN}' is already numeric.")
                 unique_numeric_targets = df_raw[TARGET_COLUMN].unique()
                 if not all(val in [0, 1] for val in unique_numeric_targets if pd.notna(val)):
                      logger.warning(f"Numeric target contains values other than 0/1: {unique_numeric_targets}.")
            else:
                 raise TypeError(f"Target column '{TARGET_COLUMN}' has unexpected type: {df_raw[TARGET_COLUMN].dtype}.")

            df_raw[TARGET_COLUMN] = df_raw[TARGET_COLUMN].astype(int)

            # --- Split Data ---
            logger.info("Splitting data into training and test sets...")
            X = df_raw.drop(TARGET_COLUMN, axis=1)
            y = df_raw[TARGET_COLUMN]

            if not isinstance(X, pd.DataFrame):
                logger.warning("X is not DataFrame. Attempting conversion.")
                X = pd.DataFrame(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            logger.info(f"Data split complete: Train shape={X_train.shape}, Test shape={X_test.shape}")
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_metric("target_positive_ratio_train", y_train.mean())
            mlflow.log_metric("target_positive_ratio_test", y_test.mean())

        except Exception as data_err:
             logger.error(f"Failed during data loading or preparation: {data_err}", exc_info=True)
             mlflow.set_tag("status", "FAILED_DATA_LOAD")
             mlflow.end_run("FAILED")
             sys.exit(1)

        # Generate baseline data profile
        logger.info("Generating baseline data profile...")
        try:
            # Create preprocessing pipeline with default settings
            preprocessor = create_preprocessing_pipeline(
                numerical_cols=[col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])],
                categorical_cols=[col for col in X_train.columns if pd.api.types.is_categorical_dtype(X_train[col])],
                ordinal_cols=[],  # Add your ordinal columns if any
                business_travel_col=[],  # Add if you have business travel columns
                skewed_cols=find_skewed_columns(X_train, threshold=SKEWNESS_THRESHOLD)
            )
            
            # Fit and transform the training data
            X_train_processed = preprocessor.fit_transform(X_train)
            if isinstance(X_train_processed, np.ndarray):
                X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
            else:
                X_train_processed_df = X_train_processed

            # Generate and save baseline profile
            profile = generate_evidently_profile(X_train_processed_df)
            os.makedirs(REPORTS_PATH, exist_ok=True)
            profile_path = os.path.join(REPORTS_PATH, "baseline_profile.json")
            profile.save_html(profile_path.replace('.json', '.html'))
            profile_json = profile.json()
            save_json(profile_json, profile_path)

            # Log artifacts to MLflow
            mlflow.log_artifact(profile_path, artifact_path="drift_reference")
            mlflow.log_artifact(profile_path.replace('.json', '.html'), artifact_path="drift_reference")

            # Save reference data sample
            ref_data_path = os.path.join(REPORTS_PATH, "reference_train_data.parquet")
            X_train_processed_df.sample(min(1000, len(X_train_processed_df))).to_parquet(
                ref_data_path, index=False
            )
            mlflow.log_artifact(ref_data_path, artifact_path="drift_reference")
            logger.info("Successfully generated and saved baseline data profile")

        except Exception as profile_err:
            logger.error(f"Failed to generate baseline data profile: {profile_err}", exc_info=True)
            mlflow.set_tag("profile_generation_status", "FAILED")

        # --- Optuna HPO Loop ---
        optuna_results = {}
        for model_type in models_to_opt:
            if model_type not in CLASSIFIER_MAP or model_type not in PARAM_FUNC_MAP:
                logger.warning(f"Skipping Optuna HPO for '{model_type}': Class or Param function not defined.")
                continue
            logger.info(f"--- Starting Optuna HPO for: {model_type} ---")
            with mlflow.start_run(run_name=f"Optuna_{model_type}_Study", nested=True) as study_run:
                 study_run_id = study_run.info.run_id
                 mlflow.log_param("model_type", model_type)
                 study_name = f"optuna_{model_type}_{study_run_id[:8]}"
                 study = optuna.create_study(direction="maximize", study_name=study_name)
                 logger.info(f"Created Optuna study: {study_name}")
                 try:
                    study.optimize(
                        lambda trial: optuna_objective(trial, model_type, X_train, y_train),
                        n_trials=HPO_N_TRIALS,
                        gc_after_trial=True,
                    )
                    if study.best_trial:
                        best_trial = study.best_trial
                        current_best_score = best_trial.value
                        optuna_results[model_type] = {
                            "score": current_best_score, "params": best_trial.params,
                            "study_run_id": study_run_id, "best_trial_number": best_trial.number
                        }
                        mlflow.log_metric("best_cv_score", current_best_score)
                        params_to_log = {k:v for k,v in best_trial.params.items() if isinstance(v, (str, int, float, bool))}
                        mlflow.log_params({f"best_{k}":v for k,v in params_to_log.items()})
                        mlflow.set_tag("status", "COMPLETED")
                        logger.info(f"Optuna HPO study for {model_type} complete. Best Score ({PRIMARY_METRIC}): {current_best_score:.4f} in Trial {best_trial.number}")
                        if current_best_score > best_overall_score:
                            best_overall_score = current_best_score
                            best_optuna_config = best_trial.params.copy()
                            best_optuna_config['model_type'] = model_type
                            best_model_type_name = model_type
                            logger.info(f"*** New best overall model found: Optuna {model_type} (Score: {best_overall_score:.4f}) ***")
                    else:
                         logger.error(f"Optuna study for {model_type} finished without a best trial.")
                         mlflow.set_tag("status", "FAILED_NO_TRIAL")
                         optuna_results[model_type] = {"score": -1.0, "params": {}, "study_run_id": study_run_id}
                 except Exception as study_err:
                     if isinstance(study_err, mlflow.exceptions.MlflowException): logger.error(f"MLflow logging failed during Optuna study for {model_type}: {study_err}", exc_info=False)
                     else: logger.error(f"Optuna study for {model_type} failed: {study_err}", exc_info=True)
                     try:
                         mlflow.set_tag("status", "FAILED")
                         mlflow.log_param("study_error", str(study_err)[:250])
                     except Exception as log_err: logger.error(f"Failed to log study error to MLflow: {log_err}")
                     optuna_results[model_type] = {"score": -1.0, "params": {}, "study_run_id": study_run_id}


        # --- Final Model Selection and Training ---
        if best_optuna_config is None:
            logger.error("No successful Optuna optimization runs. Cannot train final model.")
            mlflow.set_tag("status", "FAILED_NO_BEST_MODEL")
            mlflow.end_run("FAILED")
            sys.exit(1)

        logger.info(f"--- Training Final Best Model (Optuna Winner: {best_model_type_name}) ---")
        with mlflow.start_run(run_name=f"Final_Training_{best_model_type_name}", nested=True) as final_run:
            final_run_id = final_run.info.run_id
            mlflow.log_param("winning_hpo_method", "Optuna")
            mlflow.log_param("winning_model_type", best_model_type_name)
            mlflow.log_metric("best_cv_score_from_hpo", best_overall_score)
            if best_model_type_name in optuna_results:
                 mlflow.log_param("source_study_run_id", optuna_results[best_model_type_name].get("study_run_id"))
                 mlflow.log_param("source_best_trial_number", optuna_results[best_model_type_name].get("best_trial_number"))

            # Prepare Final Pipeline Configuration
            final_model_params = {}
            pipeline_params = {}
            feature_selector_params_final = {}
            model_type = best_optuna_config['model_type']
            for key, value in best_optuna_config.items():
                 if key.startswith('model_') and key != 'model_type':
                      clean_key = key.replace('model_', '', 1)
                      if model_type == 'logistic_regression' and ('penalty_' in key): final_model_params['penalty'] = value
                      else: final_model_params[clean_key] = value
                 elif key.startswith('pipe_'): pipeline_params[key.replace('pipe_', '', 1)] = value
                 elif key.startswith('selector_'): feature_selector_params_final[key.replace('selector_', '', 1)] = value

            mlflow.log_params({f"final_pipe_{k}": v for k,v in pipeline_params.items()})
            mlflow.log_params({f"final_selector_{k}": v for k,v in feature_selector_params_final.items()})
            mlflow.log_params({f"final_model_{k}": v for k,v in final_model_params.items()})
            mlflow.log_dict(best_optuna_config, "best_optuna_trial_params.json")

            final_pipeline = None
            preprocessor_final = None # Define preprocessor variable

            try:
                logger.info("Rebuilding pipeline using best Optuna parameters...")
                classifier_class = CLASSIFIER_MAP[model_type]

                # Re-run Data Prep Steps for Final Pipeline Build
                # Apply necessary transformers *before* identifying types for the final pipeline build
                temp_age_grouper_final = AgeGroupTransformer()
                X_train_temp_age_final = temp_age_grouper_final.fit_transform(X_train.copy()) # Use X_train copy
                temp_feature_adder_final = AddNewFeaturesTransformer()
                X_train_eng_temp_final = temp_feature_adder_final.fit_transform(X_train_temp_age_final)
                col_types_final = identify_column_types(X_train_eng_temp_final, None) # Identify types on engineered train data
                skewed_cols_final = find_skewed_columns(X_train_eng_temp_final, col_types_final['numerical'], threshold=SKEWNESS_THRESHOLD)
                del X_train_eng_temp_final, X_train_temp_age_final # Clean up

                # Create the preprocessor part of the pipeline
                preprocessor_final = create_preprocessing_pipeline(
                    numerical_cols=col_types_final['numerical'],
                    categorical_cols=col_types_final['categorical'],
                    ordinal_cols=col_types_final['ordinal'],
                    business_travel_col=col_types_final['business_travel'],
                    skewed_cols=skewed_cols_final,
                    numeric_transformer_type=pipeline_params.get('num_transform', 'passthrough'),
                    numeric_scaler_type=pipeline_params.get('num_scaler', 'passthrough'),
                    business_encoder_type=pipeline_params.get('bt_encoder', 'onehot'),
                )
                # Create the full final pipeline
                final_pipeline = create_full_pipeline(
                    classifier_class=classifier_class,
                    model_params=final_model_params,
                    preprocessor=preprocessor_final, # Use the created preprocessor
                    feature_selector_type=pipeline_params.get('selector', 'passthrough'),
                    feature_selector_params=feature_selector_params_final,
                    smote_active=pipeline_params.get('smote', False)
                )
                logger.info("Final pipeline built successfully.")

                # --- Actual Training on Full Training Data ---
                logger.info("Fitting final pipeline on full training data (X_train, y_train)...")
                start_time = time.time()
                # Fit the *entire* pipeline on the original X_train
                final_pipeline.fit(X_train, y_train)
                end_time = time.time()
                training_duration = end_time - start_time
                logger.info(f"Final pipeline fitting complete. Time taken: {training_duration:.2f} seconds")
                mlflow.log_metric("final_training_time_seconds", training_duration)

                # --- Evaluation on Test Set ---
                logger.info("Evaluating final model on test set (X_test, y_test)...")
                y_pred_test = final_pipeline.predict(X_test)
                test_metrics = {
                    "test_accuracy": accuracy_score(y_test, y_pred_test),
                    "test_f1": f1_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0),
                    f"test_{PRIMARY_METRIC}": fbeta_score(y_test, y_pred_test, beta=beta_value, pos_label=pos_label, zero_division=0),
                    "test_precision": precision_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0),
                    "test_recall": recall_score(y_test, y_pred_test, pos_label=pos_label, zero_division=0),
                }
                if hasattr(final_pipeline, "predict_proba"):
                    try:
                        y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
                        test_metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
                        # --- Log ROC Curve Plot ---
                        try:
                            logger.info("Generating and logging ROC Curve plot...")
                            roc_plot_path = os.path.join(REPORTS_PATH, f"roc_curve_{final_run_id}.png")
                            fig, ax = plt.subplots()
                            RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=ax, name=f"{best_model_type_name} Test Set")
                            plt.title("ROC Curve")
                            plt.grid(True)
                            plt.savefig(roc_plot_path)
                            plt.close(fig) # Close the figure to free memory
                            mlflow.log_artifact(roc_plot_path, artifact_path="evaluation_reports")
                            logger.info("ROC Curve plot saved and logged.")
                        except Exception as roc_err:
                            logger.error(f"Could not generate or log ROC Curve plot: {roc_err}", exc_info=True)
                    except Exception as auc_err:
                        logger.warning(f"Could not calculate AUC score: {auc_err}")
                        test_metrics["test_auc"] = -1.0
                else:
                     logger.warning(f"Classifier {model_type} does not support predict_proba. AUC not calculated.")
                     test_metrics["test_auc"] = -1.0

                mlflow.log_metrics(test_metrics)
                logger.info(f"Final Test Metrics: {test_metrics}")

                # --- Log Confusion Matrix ---
                try:
                    cm = confusion_matrix(y_test, y_pred_test)
                    os.makedirs(REPORTS_PATH, exist_ok=True)
                    cm_filename = f"confusion_matrix_{final_run_id}.json"
                    cm_path = os.path.join(REPORTS_PATH, cm_filename)
                    save_json({"labels": [0, 1], "matrix": cm.tolist()}, cm_path)
                    mlflow.log_artifact(cm_path, artifact_path="evaluation_reports")
                    logger.info(f"Confusion matrix saved and logged.")
                except Exception as cm_err:
                    logger.error(f"Could not log confusion matrix: {cm_err}", exc_info=True)
                
                # --- Log Data Profile Baseline (X_train) ---
                if ProfileReport: # Check if ydata-profiling was imported successfully
                    try:
                        logger.info("Generating Data Profile for X_train...")
                        profile_path = os.path.join(REPORTS_PATH, f"training_data_profile_{final_run_id}.html")
                        # Use minimal=True for faster report generation if needed
                        profile = ProfileReport(X_train, title="Training Data Profile", minimal=True)
                        profile.to_file(profile_path)
                        mlflow.log_artifact(profile_path, artifact_path="data_baselines")
                        logger.info("Training data profile saved and logged.")
                    except Exception as profile_err:
                        logger.error(f"Could not generate or log data profile: {profile_err}", exc_info=True)
                else:
                    # Update warning message
                    logger.warning("Skipping data profile generation as ydata-profiling is not installed or import failed.")
                
                # --- Log Prediction Baseline (Test Set Probabilities) ---
                if 'y_test_proba' in locals(): # Check if probabilities were calculated
                    try:
                        logger.info("Logging prediction baseline statistics...")
                        pred_stats = pd.Series(y_test_proba).describe().to_dict()
                        mlflow.log_metrics({f"test_pred_proba_{k}": v for k,v in pred_stats.items()})

                        # Optional: Log histogram artifact
                        hist_path = os.path.join(REPORTS_PATH, f"prediction_histogram_{final_run_id}.png")
                        fig, ax = plt.subplots()
                        pd.Series(y_test_proba).hist(ax=ax, bins=50)
                        ax.set_title("Test Set Prediction Probability Distribution")
                        ax.set_xlabel("Predicted Probability (Class 1)")
                        ax.set_ylabel("Frequency")
                        plt.grid(True)
                        plt.savefig(hist_path)
                        plt.close(fig)
                        mlflow.log_artifact(hist_path, artifact_path="data_baselines")
                        logger.info("Prediction baseline stats and histogram logged.")

                    except Exception as pred_base_err:
                        logger.error(f"Could not log prediction baseline: {pred_base_err}", exc_info=True)
                else:
                    logger.warning("Skipping prediction baseline logging as y_test_proba is not available.")


                # ===========================================================
                # --- START: Fairness Analysis ---
                # ===========================================================
                logger.info("--- Starting Fairness Analysis ---")
                fairness_metrics_log = {}
                fairness_reports = {}

                # Ensure sensitive features are present in the original X_test
                missing_sensitive = [f for f in SENSITIVE_FEATURES if f not in X_test.columns]
                if missing_sensitive:
                    logger.error(f"FATAL: Sensitive features {missing_sensitive} not found in X_test. Cannot perform fairness analysis.")
                    mlflow.set_tag("fairness_status", "FAILED_MISSING_FEATURES")
                else:
                    try:
                        for sensitive_col in SENSITIVE_FEATURES:
                            logger.info(f"Calculating fairness metrics for sensitive feature: {sensitive_col}")
                            # Use MetricFrame to calculate metrics grouped by the sensitive feature
                            grouped_on_col = MetricFrame(metrics=metrics_dict,
                                                         y_true=y_test,
                                                         y_pred=y_pred_test,
                                                         sensitive_features=X_test[sensitive_col])

                            # --- CORRECTED Fairness Difference Calculation ---
                            # Calculate difference for each metric manually from by_group results
                            by_group_metrics = grouped_on_col.by_group
                            if isinstance(by_group_metrics, pd.Series): # Handle single metric case if metrics_dict had only one
                                by_group_metrics = by_group_metrics.unstack()

                            for metric_name in metrics_dict.keys():
                                if metric_name in by_group_metrics.columns:
                                    metric_values = by_group_metrics[metric_name]
                                    metric_diff = metric_values.max() - metric_values.min()
                                    fairness_metrics_log[f'{sensitive_col}_{metric_name}_diff'] = metric_diff
                                    logger.debug(f"  Metric '{metric_name}' difference for {sensitive_col}: {metric_diff:.4f}")
                                else:
                                    logger.warning(f"Metric '{metric_name}' not found in MetricFrame results for {sensitive_col}.")


                            # Calculate and log specific fairness metrics (DPD, EOD)
                            try:
                                dpd = demographic_parity_difference(y_test, y_pred_test, sensitive_features=X_test[sensitive_col])
                                eod = equalized_odds_difference(y_test, y_pred_test, sensitive_features=X_test[sensitive_col])
                                fairness_metrics_log[f'{sensitive_col}_demographic_parity_difference'] = dpd
                                fairness_metrics_log[f'{sensitive_col}_equalized_odds_difference'] = eod
                                logger.info(f"  Demographic Parity Difference: {dpd:.4f}")
                                logger.info(f"  Equalized Odds Difference: {eod:.4f}")
                            except Exception as fair_metric_err:
                                logger.warning(f"Could not calculate specific fairness metrics (DPD, EOD) for {sensitive_col}: {fair_metric_err}")
                                fairness_metrics_log[f'{sensitive_col}_demographic_parity_difference'] = np.nan
                                fairness_metrics_log[f'{sensitive_col}_equalized_odds_difference'] = np.nan


                            # Store the detailed MetricFrame results for artifact logging
                            # Convert potential Series/DataFrames in the dict to basic types for JSON
                            report_dict = {
                                "overall": grouped_on_col.overall.to_dict() if hasattr(grouped_on_col.overall, 'to_dict') else grouped_on_col.overall,
                                "by_group": grouped_on_col.by_group.to_dict() if hasattr(grouped_on_col.by_group, 'to_dict') else grouped_on_col.by_group,
                                # Recalculate difference/ratio for the report artifact if needed
                                "difference_overall": grouped_on_col.difference(method='between_groups').to_dict() if hasattr(grouped_on_col.difference(), 'to_dict') else grouped_on_col.difference(method='between_groups'),
                                "ratio_overall": grouped_on_col.ratio(method='between_groups').to_dict() if hasattr(grouped_on_col.ratio(), 'to_dict') else grouped_on_col.ratio(method='between_groups')
                            }
                            fairness_reports[sensitive_col] = report_dict
                            logger.info(f"MetricFrame calculated for {sensitive_col}.")
                            logger.debug(f"Report for {sensitive_col}: {json.dumps(report_dict, indent=2)}")


                        # Log the calculated fairness metrics to MLflow
                        mlflow.log_metrics(fairness_metrics_log)
                        logger.info("Logged fairness difference metrics to MLflow.")

                        # Save the detailed fairness reports as a JSON artifact
                        fairness_report_path = os.path.join(REPORTS_PATH, f"fairness_report_{final_run_id}.json")
                        save_json(fairness_reports, fairness_report_path)
                        mlflow.log_artifact(fairness_report_path, artifact_path="evaluation_reports")
                        logger.info(f"Detailed fairness report saved and logged as artifact.")
                        mlflow.set_tag("fairness_status", "COMPLETED")

                    except Exception as fair_err:
                        logger.error(f"Fairness analysis failed: {fair_err}", exc_info=True)
                        mlflow.set_tag("fairness_status", "FAILED")
                        mlflow.log_param("fairness_error", str(fair_err)[:250])

                # ===========================================================
                # --- END: Fairness Analysis ---
                # ===========================================================


                # ===========================================================
                # --- START: Explainability Analysis (SHAP) ---
                # ===========================================================
                logger.info("--- Starting Explainability Analysis (SHAP) ---")
                X_test_processed_df = None # Initialize
                try:
                    # --- CORRECTED SHAP Data Preparation v2 ---
                    logger.info("Applying pipeline steps (excluding classifier) to X_test for SHAP...")

                    # Get the final fitted classifier instance
                    model_instance = final_pipeline.named_steps['classifier']
                    # Get data transformed by all steps *before* the classifier
                    # We need to iterate through the steps and transform the data
                    X_test_transformed = X_test.copy() # Start with original X_test
                    feature_names_current = list(X_test.columns) # Keep track of names

                    for step_name, step_obj in final_pipeline.steps:
                        if step_name == 'classifier':
                            break # Stop before the classifier

                        logger.debug(f"Applying step '{step_name}' for SHAP data prep...")
                        if step_name == 'smote': # Skip SMOTE for SHAP on test data
                             logger.debug("Skipping SMOTE step for SHAP data preparation.")
                             continue

                        # Apply the transform method of the current step
                        # Note: This assumes each step has a 'transform' method
                        if hasattr(step_obj, 'transform'):
                             X_test_transformed = step_obj.transform(X_test_transformed)
                             logger.debug(f"Shape after step '{step_name}': {X_test_transformed.shape}")

                             # Attempt to get feature names if possible
                             if hasattr(step_obj, 'get_feature_names_out'):
                                 try:
                                     # Check if input_features argument is needed
                                     import inspect
                                     sig = inspect.signature(step_obj.get_feature_names_out)
                                     if 'input_features' in sig.parameters:
                                         # Pass the names from the previous step
                                         feature_names_current = step_obj.get_feature_names_out(input_features=feature_names_current)
                                     else: # No input_features needed
                                         feature_names_current = step_obj.get_feature_names_out()
                                     logger.debug(f"Feature names after step '{step_name}': {len(feature_names_current)}")
                                 except Exception as name_err:
                                     logger.warning(f"Could not get feature names from step '{step_name}': {name_err}. Feature names might become inaccurate.")
                                     # Fallback: if shape changed, generate generic names
                                     if X_test_transformed.shape[1] != len(feature_names_current):
                                          feature_names_current = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
                             elif isinstance(X_test_transformed, np.ndarray) and X_test_transformed.shape[1] != len(feature_names_current):
                                 # If output is numpy array and shape changed, generate generic names
                                 logger.warning(f"Output of step '{step_name}' is numpy array and shape changed. Using generic feature names.")
                                 feature_names_current = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
                        else:
                             logger.warning(f"Step '{step_name}' does not have a 'transform' method. Skipping transformation for this step in SHAP prep.")

                    # Ensure the final transformed data is a DataFrame for SHAP
                    if isinstance(X_test_transformed, np.ndarray):
                        # Validate feature name length
                        if len(feature_names_current) != X_test_transformed.shape[1]:
                            logger.warning(f"Final feature name count ({len(feature_names_current)}) doesn't match data columns ({X_test_transformed.shape[1]}). Using generic names.")
                            feature_names_current = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
                        X_test_processed_df = pd.DataFrame(X_test_transformed, columns=feature_names_current)
                    elif isinstance(X_test_transformed, pd.DataFrame):
                         X_test_processed_df = X_test_transformed # Already a DataFrame
                         # Update feature names if necessary (e.g., if ColumnTransformer returned DF with names)
                         feature_names_current = list(X_test_processed_df.columns)
                    else:
                         raise TypeError(f"Unexpected data type after transformations for SHAP: {type(X_test_transformed)}")

                    logger.info(f"X_test processed for SHAP. Final Shape: {X_test_processed_df.shape}")
                    logger.debug(f"Final features for SHAP: {list(X_test_processed_df.columns)}")


                    # --- Select SHAP explainer ---
                    logger.info(f"Creating SHAP explainer for model type: {model_type}")
                    # Use the processed data as the background/masker
                    masker = X_test_processed_df

                    # Choose explainer based on the final model instance
                    if model_type in ["random_forest", "gradient_boosting"]:
                        explainer = shap.TreeExplainer(model_instance, data=masker, feature_perturbation="interventional")
                    elif model_type == "logistic_regression":
                        explainer = shap.LinearExplainer(model_instance, masker)
                    elif model_type == "mlp":
                        logger.info("Using PartitionExplainer for MLP.")
                        predict_fn = model_instance.predict_proba if hasattr(model_instance, "predict_proba") else model_instance.predict
                        explainer = shap.PartitionExplainer(predict_fn, masker)
                    else:
                         logger.warning(f"SHAP explainer not explicitly defined for {model_type}. Attempting PermutationExplainer.")
                         predict_fn = model_instance.predict_proba if hasattr(model_instance, "predict_proba") else model_instance.predict
                         explainer = shap.PermutationExplainer(predict_fn, masker)

                    # --- Calculate SHAP values ---
                    logger.info("Calculating SHAP values...")
                    start_shap_time = time.time()

                    # Calculation depends on explainer type and model output
                    if isinstance(explainer, shap.LinearExplainer):
                        shap_values_obj = explainer(X_test_processed_df)
                        shap_values = shap_values_obj.values
                    elif isinstance(explainer, shap.TreeExplainer):
                         shap_values_output = explainer.shap_values(X_test_processed_df) # Pass DataFrame
                         # Handle list output for binary classification
                         if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                              shap_values = shap_values_output[1] # Positive class
                         else: # Assume single output (e.g., regression or already handled)
                              shap_values = shap_values_output
                    elif isinstance(explainer, (shap.PartitionExplainer, shap.KernelExplainer, shap.PermutationExplainer)):
                         shap_values_output = explainer(X_test_processed_df) # Pass DataFrame
                         # Handle Explanation object or list output
                         if isinstance(shap_values_output, shap.Explanation):
                              if len(shap_values_output.shape) > 1 and shap_values_output.shape[-1] == 2:
                                   shap_values = shap_values_output.values[..., 1] # Positive class for proba
                              else:
                                   shap_values = shap_values_output.values
                         elif isinstance(shap_values_output, list) and len(shap_values_output) == 2:
                              shap_values = shap_values_output[1] # Positive class for proba list output
                         else:
                              shap_values = shap_values_output # Fallback
                    else: # Fallback if explainer type is unknown
                        shap_values = explainer.shap_values(X_test_processed_df)


                    end_shap_time = time.time()
                    if isinstance(shap_values, np.ndarray):
                         logger.info(f"SHAP values calculated. Time: {end_shap_time - start_shap_time:.2f} sec. Shape: {shap_values.shape}")
                    else:
                         logger.warning(f"SHAP values calculated, but type is {type(shap_values)}. Cannot log shape.")


                    # --- Generate and Log SHAP Summary Plot ---
                    try:
                        logger.info("Generating SHAP summary plot...")
                        shap_summary_plot_path = os.path.join(REPORTS_PATH, f"shap_summary_{final_run_id}.png")
                        plt.figure()
                        # Pass the DataFrame for correct feature names
                        shap.summary_plot(shap_values, X_test_processed_df, plot_type="dot", show=False)
                        plt.tight_layout()
                        plt.savefig(shap_summary_plot_path)
                        plt.close()
                        mlflow.log_artifact(shap_summary_plot_path, artifact_path="explainability_reports")
                        logger.info(f"SHAP summary plot saved and logged.")
                    except Exception as shap_plot_err:
                        logger.error(f"Failed to generate or log SHAP summary plot: {shap_plot_err}", exc_info=True)

                    # --- Generate and Log Feature Importance Plot (from SHAP) ---
                    try:
                         logger.info("Generating Feature Importance plot from SHAP values...")
                         feat_importance_plot_path = os.path.join(REPORTS_PATH, f"{FEATURE_IMPORTANCE_PLOT_FILENAME.replace('.png', '')}_{final_run_id}.png")
                         plt.figure()
                         # Pass the DataFrame for correct feature names
                         shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
                         plt.tight_layout()
                         plt.savefig(feat_importance_plot_path)
                         plt.close()
                         mlflow.log_artifact(feat_importance_plot_path, artifact_path="explainability_reports")
                         logger.info(f"Feature importance plot (SHAP bar) saved and logged.")
                    except Exception as fi_plot_err:
                         logger.error(f"Failed to generate or log Feature Importance plot: {fi_plot_err}", exc_info=True)

                    mlflow.set_tag("explainability_status", "COMPLETED")

                except Exception as shap_err:
                    logger.error(f"Explainability analysis (SHAP) failed: {shap_err}", exc_info=True)
                    mlflow.set_tag("explainability_status", "FAILED")
                    mlflow.log_param("shap_error", str(shap_err)[:250])

                # ===========================================================
                # --- END: Explainability Analysis (SHAP) ---
                # ===========================================================


                # --- Log and Register Final Model ---
                logger.info(f"Logging and registering the final model pipeline as: {PRODUCTION_MODEL_NAME}")
                # Define input example and signature for better model serving/inference
                try:
                    input_example = X_train.iloc[:5] # Use a few rows from original training data
                    # Use predict() for signature inference unless predict_proba is required
                    signature = mlflow.models.infer_signature(X_train, final_pipeline.predict(X_train))
                    mlflow.sklearn.log_model(
                        sk_model=final_pipeline,
                        artifact_path="final_model_pipeline",
                        registered_model_name=PRODUCTION_MODEL_NAME,
                        signature=signature,
                        input_example=input_example
                    )
                    logger.info(f"Model logged with signature and input example, registered as '{PRODUCTION_MODEL_NAME}'.")
                except Exception as log_model_err:
                     logger.warning(f"Failed to log model with signature/example: {log_model_err}. Logging without.")
                     mlflow.sklearn.log_model(
                        sk_model=final_pipeline,
                        artifact_path="final_model_pipeline",
                        registered_model_name=PRODUCTION_MODEL_NAME
                     )
                     logger.info(f"Model logged (without signature) and registered as '{PRODUCTION_MODEL_NAME}'.")


                # --- Transition Model to Staging Stage ---
                client = mlflow.tracking.MlflowClient()
                try:
                    time.sleep(5) # Allow time for registration
                    latest_versions = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["None"])
                    if latest_versions:
                        model_version = latest_versions[0].version
                        logger.info(f"Transitioning model version {model_version} of '{PRODUCTION_MODEL_NAME}' to Staging stage.")
                        client.transition_model_version_stage(
                            name=PRODUCTION_MODEL_NAME, version=model_version,
                            stage="Staging", archive_existing_versions=False # Typically don't archive when moving to Staging
                        )
                        mlflow.set_tag("model_registered_stage", "Staging")
                        logger.info(f"Model version {model_version} successfully transitioned to Staging.")
                    else:
                        logger.error("Could not find newly registered model version in 'None' stage.")
                        mlflow.set_tag("model_registration_status", "Transition_Failed_NotFound")
                except Exception as transition_err:
                    logger.error(f"Failed to transition model to Staging stage: {transition_err}", exc_info=True)
                    mlflow.set_tag("model_registration_status", f"Transition_Failed_{type(transition_err).__name__}")


                mlflow.set_tag("status", "COMPLETED")
                logger.info("Final model training, evaluation, fairness/XAI, and registration complete.")

            except Exception as final_train_err:
                logger.error(f"Final model training/evaluation/analysis failed: {final_train_err}", exc_info=True)
                mlflow.set_tag("status", "FAILED_FINAL_TRAINING")
                try: mlflow.log_param("final_training_error", str(final_train_err)[:250])
                except Exception: pass

        # Log overall best results to the parent run
        mlflow.log_metric("best_overall_cv_score", best_overall_score)
        mlflow.log_param("winning_model_type", best_model_type_name if best_model_type_name else "None")
        mlflow.set_tag("status", "COMPLETED_PROCESS")
        logger.info(f"Overall process finished. Best model type: {best_model_type_name}, Best CV Score ({PRIMARY_METRIC}): {best_overall_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna HPO, select best model, train, evaluate (incl. fairness/XAI), and register using DB data.")
    args = parser.parse_args()

    if 'MODELS_TO_OPTIMIZE' not in globals() or not MODELS_TO_OPTIMIZE:
        logger.error("MODELS_TO_OPTIMIZE not defined or empty in config.py. Cannot proceed.")
        sys.exit(1)
    models_list = MODELS_TO_OPTIMIZE

    logger.info(f"Starting script execution for models: {models_list}")
    optimize_select_and_train(models_list)
    logger.info("Script execution finished.")
