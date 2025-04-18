
import os, sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add it to sys.path if it's not already there
if src_path not in sys.path:
    sys.path.append(src_path)

# scripts/hpo.py
import argparse
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score # Ensure fbeta_score is imported
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier # Import MLP if used
# Add other model classes as needed from sklearn or elsewhere
import sys
import os
import logging
import time
import json # Added for saving params

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your source code
# Ensure correct package name based on your project structure
from employee_attrition_mlops.config import (
    RAW_DATA_PATH, TARGET_COLUMN, RANDOM_STATE, HPO_N_TRIALS, HPO_CV_FOLDS,
    MLFLOW_TRACKING_URI, HPO_EXPERIMENT_NAME, MODELS_TO_OPTIMIZE, PRIMARY_METRIC,
    REPORTS_PATH
)
from employee_attrition_mlops.data_processing import (
    load_and_clean_data, identify_column_types, find_skewed_columns,
    AddNewFeaturesTransformer
)
from employee_attrition_mlops.pipelines import create_preprocessing_pipeline, create_full_pipeline
from employee_attrition_mlops.utils import save_json

# --- Define Model Classes ---
CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier, # Added MLP
    # Add others as needed
}

# --- Define Search Spaces ---
# (Keep the get_logreg_params, get_rf_params, get_gb_params, get_mlp_params functions)
def get_logreg_params(trial: optuna.Trial) -> dict:
    # Compatibility check for L1 penalty
    solver = trial.suggest_categorical("model_solver", ["liblinear", "saga"])
    if solver == 'liblinear':
        penalty = trial.suggest_categorical("model_penalty", ["l1", "l2"])
    else: # saga supports both
        penalty = trial.suggest_categorical("model_penalty_saga", ["l1", "l2", "elasticnet"]) # saga also supports elasticnet

    # Ensure penalty matches solver if needed (e.g., liblinear doesn't support elasticnet)
    # This logic might need refinement based on specific solver/penalty combos
    params = {
        "C": trial.suggest_float("model_C", 1e-4, 1e2, log=True),
        "solver": solver,
        "penalty": penalty,
        "max_iter": trial.suggest_int("model_max_iter", 300, 1500), # Increased max_iter for saga
        "class_weight": trial.suggest_categorical("model_class_weight", [None, "balanced"]),
        "random_state": RANDOM_STATE,
    }
    # Add l1_ratio for elasticnet with saga
    if solver == 'saga' and penalty == 'elasticnet':
         params['l1_ratio'] = trial.suggest_float("model_l1_ratio", 0, 1)

    return params

def get_rf_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("model_n_estimators", 50, 500),
        "max_depth": trial.suggest_int("model_max_depth", 3, 20, step=1, log=False), # Use step for integer range
        "min_samples_split": trial.suggest_int("model_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("model_min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("model_max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("model_class_weight", [None, "balanced", "balanced_subsample"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1 # Use all available cores
    }

def get_gb_params(trial: optuna.Trial) -> dict:
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
     # Suggest number of layers
     n_layers = trial.suggest_int("model_n_layers", 1, 3)
     layers = []
     for i in range(n_layers):
          layers.append(trial.suggest_int(f"model_n_units_l{i}", 32, 256))
     hidden_layer_sizes = tuple(layers)

     return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("model_activation", ["relu", "tanh", "logistic"]),
        "solver": trial.suggest_categorical("model_solver", ["adam", "sgd"]),
        "alpha": trial.suggest_float("model_alpha", 1e-6, 1e-2, log=True), # Regularization
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
    # Add mappings for other models
}

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function to optimize pipeline and model hyperparameters."""
    mlflow_run = None # Initialize to handle potential errors before start_run
    # --- Suggest Pipeline Parameters ---
    numeric_transformer_type = trial.suggest_categorical("pipe_num_transform", ['log', 'boxcox', 'passthrough'])
    numeric_scaler_type = trial.suggest_categorical("pipe_num_scaler", ['standard', 'minmax'])
    business_encoder_type = trial.suggest_categorical("pipe_bt_encoder", ['ordinal', 'onehot'])
    feature_selector_type = trial.suggest_categorical("pipe_selector", ['rfe', 'lasso', 'tree', 'passthrough'])
    smote_active = trial.suggest_categorical("pipe_smote", [True, False])

    feature_selector_params = {}
    if feature_selector_type == 'rfe':
        feature_selector_params['n_features_to_select'] = trial.suggest_int("selector_rfe_n", 5, X.shape[1] // 2) # Select between 5 and half features
    elif feature_selector_type == 'lasso':
        feature_selector_params['C'] = trial.suggest_float("selector_lasso_C", 1e-3, 1e1, log=True)
        # Ensure solver compatible with L1
        feature_selector_params['solver'] = 'liblinear' # Or saga
        feature_selector_params['penalty'] = 'l1'
        # Add max_iter for solver convergence
        feature_selector_params['max_iter'] = 1000 # Increase if needed
    elif feature_selector_type == 'tree':
         feature_selector_params['threshold'] = trial.suggest_categorical("selector_tree_thresh", ['median', 'mean', 0.01, 0.005])
         # Optionally tune tree params for feature selection too
         # feature_selector_params['estimator__max_depth'] = trial.suggest_int("selector_tree_depth", 3, 10)

    # --- Suggest Model Hyperparameters ---
    if model_type not in PARAM_FUNC_MAP:
        raise ValueError(f"Unsupported model_type for HPO: {model_type}")
    model_params = PARAM_FUNC_MAP[model_type](trial)
    classifier_class = CLASSIFIER_MAP[model_type]

    # --- Prepare Data (Identify Columns) ---
    # Apply initial feature engineering that affects column types/availability
    feature_adder = AddNewFeaturesTransformer()
    # Important: Use fit_transform on a copy for each trial to avoid state issues if X is reused
    X_eng = feature_adder.fit_transform(X.copy())
    col_types = identify_column_types(X_eng, None)
    skewed_cols = find_skewed_columns(X_eng, col_types['numerical'], threshold=0.75)

    # --- Create Pipeline ---
    try:
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

        # Handle incompatible parameters (e.g., L1 penalty with non-compatible solver)
        # This logic might need refinement based on specific sklearn versions/warnings
        if classifier_class == LogisticRegression:
            if model_params.get('penalty') == 'l1' and model_params.get('solver') not in ['liblinear', 'saga']:
                model_params['solver'] = 'liblinear' # Force compatible solver
                logger.warning(f"Trial {trial.number}: Forced solver to 'liblinear' for L1 penalty.")
            if model_params.get('penalty') == 'elasticnet' and model_params.get('solver') != 'saga':
                 model_params['solver'] = 'saga' # Force compatible solver
                 logger.warning(f"Trial {trial.number}: Forced solver to 'saga' for elasticnet penalty.")
                 # Ensure l1_ratio is present if elasticnet
                 if 'l1_ratio' not in model_params: model_params['l1_ratio'] = 0.5 # Default

        full_pipeline = create_full_pipeline(
            classifier_class=classifier_class,
            model_params=model_params,
            preprocessor=preprocessor,
            feature_selector_type=feature_selector_type,
            feature_selector_params=feature_selector_params,
            smote_active=smote_active,
        )
    except Exception as pipe_err:
         logger.error(f"Trial {trial.number}: Pipeline creation failed: {pipe_err}", exc_info=True)
         # Log failure to MLflow before returning
         try: # Try logging failure within the objective's MLflow context if possible
             if mlflow.active_run():
                 mlflow.set_tag("optuna_trial_state", "FAILED_PIPE_CREATION")
                 mlflow.log_param("error", str(pipe_err)[:250]) # Log truncated error
         except Exception as log_err:
              logger.error(f"Failed to log pipeline creation error to MLflow: {log_err}")
         return -1.0 # Return poor score if pipeline fails

    # --- Cross-validation ---
    try:
        cv = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # Define scorer - use the primary metric from config
        scorer = make_scorer(fbeta_score, beta=2, pos_label=1) # *** Explicitly set pos_label=1 ***

        logger.debug(f"Trial {trial.number}: Starting cross-validation...")
        # Pass the numerically encoded y
        scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring=scorer, n_jobs=1, error_score='raise')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        logger.debug(f"Trial {trial.number}: CV Scores: {scores}, Mean: {mean_score:.4f}")

        # --- MLflow Logging ---
        # Check if we are inside a parent run
        parent_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        # Use try-with-resources for nested run
        with mlflow.start_run(nested=True) as trial_run:
            # Log all suggested params (pipeline and model)
            mlflow.log_params(trial.params)
            mlflow.log_metric(f"{PRIMARY_METRIC}_cv_mean", mean_score)
            mlflow.log_metric(f"{PRIMARY_METRIC}_cv_std", std_score)
            mlflow.set_tag("optuna_trial_number", str(trial.number))
            if trial.study: mlflow.set_tag("optuna_study_name", trial.study.study_name)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("optuna_trial_state", "COMPLETE")

        return mean_score

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial.number} pruned: {e}")
        # Log pruned state to MLflow if possible
        try:
             if mlflow.active_run(): # Check if parent run context still exists
                 with mlflow.start_run(nested=True) as trial_run: # Log as separate nested run
                      mlflow.log_params(trial.params) # Log params even if pruned
                      mlflow.set_tag("optuna_trial_state", "PRUNED")
        except Exception as log_err:
             logger.error(f"Failed to log pruned trial state to MLflow: {log_err}")
        raise e # Re-raise prune exception for Optuna
    except Exception as e:
        logger.error(f"Trial {trial.number} failed during CV or logging: {e}", exc_info=True)
        # Log failed state to MLflow if possible
        try:
             if mlflow.active_run():
                 with mlflow.start_run(nested=True) as trial_run:
                      mlflow.log_params(trial.params) # Log params even on failure
                      mlflow.set_tag("optuna_trial_state", "FAILED")
                      mlflow.log_param("error", str(e)[:250]) # Log truncated error
        except Exception as log_err:
             logger.error(f"Failed to log failed trial state to MLflow: {log_err}")
        return -1.0 # Indicate failure to Optuna


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization using Optuna")
    parser.add_argument(
        "--model-type", type=str, required=True, choices=list(CLASSIFIER_MAP.keys()), # Use keys from map
        help="Type of model to optimize."
    )
    parser.add_argument(
        "--n-trials", type=int, default=HPO_N_TRIALS, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--output-params-file", type=str, default="best_hpo_params.json",
        help="File name to save the best parameters found (relative to reports dir)."
    )
    args = parser.parse_args()

    # Ensure the model type requested is supported
    if args.model_type not in PARAM_FUNC_MAP:
         logger.error(f"Model type '{args.model_type}' does not have a parameter function defined in PARAM_FUNC_MAP.")
         sys.exit(1)

    logger.info(f"Starting HPO for model type: {args.model_type} with {args.n_trials} trials.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)

    # Start a parent MLflow run for the HPO study
    with mlflow.start_run(run_name=f"{args.model_type}_HPO_Study") as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent MLflow Run ID for HPO Study: {parent_run_id}")
        mlflow.log_param("hpo_model_type", args.model_type)
        mlflow.log_param("hpo_n_trials", args.n_trials)

        # Load data once
        try:
            df_raw = load_and_clean_data(RAW_DATA_PATH)
            X = df_raw.drop(TARGET_COLUMN, axis=1)
            # *** FIX: Encode y here ***
            y = df_raw[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
            logger.info("Target variable 'y' encoded to 0/1.")
        except Exception as data_err:
             logger.error(f"Failed to load or preprocess data: {data_err}", exc_info=True)
             mlflow.set_tag("status", "FAILED_DATA_LOAD")
             sys.exit(1)


        # --- Run Optuna Study ---
        study_name = f"{args.model_type}_hpo_study_{parent_run_id}" # Unique study name per run
        # Define pruner if desired (e.g., MedianPruner)
        # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", study_name=study_name) #, pruner=pruner)

        try:
            study.optimize(
                lambda trial: objective(trial, args.model_type, X, y), # Pass encoded y
                n_trials=args.n_trials,
                # No MLflow callback needed if logging manually inside objective
                gc_after_trial=True
            )
        except Exception as study_err:
             logger.error(f"Optuna study optimization failed: {study_err}", exc_info=True)
             mlflow.set_tag("status", "FAILED_STUDY_OPTIMIZE")
             # Still try to log best trial if any completed
             pass # Continue to logging best trial if possible

        # --- Output and Save Best Results ---
        logger.info(f"HPO study '{study_name}' completed.")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Number of complete trials: {len(completed_trials)}")

        if completed_trials: # Check if there are completed trials
            best_trial = study.best_trial # best_trial is only based on COMPLETED trials
            logger.info(f"Best trial number: {best_trial.number}")
            logger.info(f"Best {PRIMARY_METRIC} Score (CV Mean): {best_trial.value:.4f}")
            logger.info("Best Parameters Found:")

            best_params_combined = best_trial.params.copy()
            best_params_combined['model_type'] = args.model_type
            best_params_combined[f'best_{PRIMARY_METRIC}_cv_mean'] = best_trial.value

            for key, value in best_params_combined.items():
                logger.info(f"  {key}: {value}")
                # Log best params to parent MLflow run
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_metric(f"best_{PRIMARY_METRIC}_cv_mean", best_trial.value)

            # --- Save best parameters to JSON file ---
            os.makedirs(REPORTS_PATH, exist_ok=True)
            output_file_path = os.path.join(REPORTS_PATH, args.output_params_file)
            save_json(best_params_combined, output_file_path)
            logger.info(f"Best parameters saved to {output_file_path}")
            # Log the best params file as an artifact to the parent run
            mlflow.log_artifact(output_file_path)
            mlflow.set_tag("status", "COMPLETED")
        else:
            logger.warning("No successful trials completed in the HPO study. Cannot determine best parameters.")
            mlflow.set_tag("status", "FAILED_NO_TRIALS")

