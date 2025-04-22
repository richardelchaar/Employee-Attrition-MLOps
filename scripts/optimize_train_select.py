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
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# --- Import Model Classes ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, SelectFromModel # Import feature selectors
from sklearn.base import BaseEstimator # Import BaseEstimator
# --- Import Imblearn ---
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Add src directory to Python path ---
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- Import from local modules ---
try:
    # Import configurations and utility functions
    from employee_attrition_mlops.config import (
        RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
        MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME,
        HPO_N_TRIALS, HPO_CV_FOLDS, PRIMARY_METRIC,
        MODELS_TO_OPTIMIZE, # List of model aliases IS in config
        REPORTS_PATH, BASELINE_PROFILE_FILENAME, SKEWNESS_THRESHOLD
    )
    from employee_attrition_mlops.data_processing import (
        load_and_clean_data, identify_column_types, find_skewed_columns,
        AddNewFeaturesTransformer
    )
    from employee_attrition_mlops.pipelines import create_preprocessing_pipeline, create_full_pipeline
    # Removed generate_profile_dict import
    from employee_attrition_mlops.utils import save_json, load_json
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure PYTHONPATH includes '{SRC_PATH}' or run from the project root.")
    sys.exit(1)


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Define Scorer ---
beta_value = 2 # Assuming F2-score based on previous context
primary_scorer = make_scorer(fbeta_score, beta=beta_value, pos_label=1, zero_division=0)
logger.info(f"Using primary metric: {PRIMARY_METRIC} (F-beta with beta={beta_value})")

# === Define Classifier Map Locally ===
CLASSIFIER_MAP = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
}
logger.info(f"Defined CLASSIFIER_MAP locally for models: {list(CLASSIFIER_MAP.keys())}")


# === HPO Parameter Function Definitions ===
def get_logreg_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for Logistic Regression."""
    solver = trial.suggest_categorical("model_solver", ["liblinear", "saga"])
    penalty_key = "penalty" # Default key name
    if solver == 'liblinear':
        # Use 'penalty' key for suggestion as it's unique to liblinear options here
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    else: # saga solver
        # Use a temporary suggestion key 'penalty_saga' to avoid Optuna name collision
        penalty = trial.suggest_categorical("penalty_saga", ["l1", "l2", "elasticnet"])

    params = {
        "C": trial.suggest_float("model_C", 1e-4, 1e2, log=True),
        "solver": solver,
        # *** MODIFIED: Always store under the standard 'penalty' key ***
        "penalty": penalty,
        "max_iter": trial.suggest_int("model_max_iter", 300, 1500),
        "class_weight": trial.suggest_categorical("model_class_weight", [None, "balanced"]),
        "random_state": RANDOM_STATE,
    }
    if solver == 'saga' and penalty == 'elasticnet':
         params['l1_ratio'] = trial.suggest_float("model_l1_ratio", 0, 1)
         # Ensure penalty is correctly set for elasticnet
         params['penalty'] = 'elasticnet'

    # No cleanup needed as we always use the 'penalty' key now
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


# === Optuna Objective Function (Reverted Logic AND Logging to Match hpo.py) ===
def optuna_objective(trial: optuna.Trial, model_type: str, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Optuna objective function - Logic matches original hpo.py. Logs to trial-specific nested run."""
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True) as trial_run:
        mlflow.set_tag("optuna_trial_number", str(trial.number))
        if trial.study: mlflow.set_tag("optuna_study_name", trial.study.study_name)
        mlflow.set_tag("model_type", model_type)

        try:
            # --- Suggest Parameters (Pipeline & Model) ---
            numeric_transformer_type = trial.suggest_categorical("pipe_num_transform", ['log', 'boxcox', 'passthrough'])
            numeric_scaler_type = trial.suggest_categorical("pipe_num_scaler", ['standard', 'minmax'])
            business_encoder_type = trial.suggest_categorical("pipe_bt_encoder", ['ordinal', 'onehot'])
            feature_selector_type = trial.suggest_categorical("pipe_selector", ['rfe', 'lasso', 'tree', 'passthrough'])
            smote_active = trial.suggest_categorical("pipe_smote", [True, False])

            feature_selector_params = {}
            if feature_selector_type == 'rfe':
                feature_selector_params['n_features_to_select'] = trial.suggest_int("selector_rfe_n", 5, max(5, X_train.shape[1] // 2))
            elif feature_selector_type == 'lasso':
                feature_selector_params['C'] = trial.suggest_float("selector_lasso_C", 1e-3, 1e1, log=True)
            elif feature_selector_type == 'tree':
                feature_selector_params['threshold'] = trial.suggest_categorical("selector_tree_thresh", ['median', 'mean', 0.01, 0.005])

            if model_type not in PARAM_FUNC_MAP: raise ValueError(f"Unsupported model_type: {model_type}")
            if model_type not in CLASSIFIER_MAP: raise ValueError(f"Classifier class not defined: {model_type}")

            model_params = PARAM_FUNC_MAP[model_type](trial) # This now correctly returns 'penalty' key
            classifier_class = CLASSIFIER_MAP[model_type]

            # Log all suggested params to this trial's run
            # Use trial.params which includes the temporary Optuna keys like 'penalty_saga'
            params_to_log = {k:v for k,v in trial.params.items() if isinstance(v, (str, int, float, bool))}
            mlflow.log_params(params_to_log)

            # --- Prepare Data (hpo.py logic: engineer THEN identify) ---
            try:
                if not isinstance(X_train, pd.DataFrame): raise TypeError("X_train must be DataFrame.")
                feature_adder = AddNewFeaturesTransformer()
                X_train_eng = feature_adder.fit_transform(X_train.copy())
                col_types = identify_column_types(X_train_eng, None)
                skewed_cols = find_skewed_columns(X_train_eng, col_types['numerical'], threshold=SKEWNESS_THRESHOLD)
                del X_train_eng
            except Exception as data_proc_err:
                 logger.error(f"Trial {trial.number}: Feature engineering or column ID failed: {data_proc_err}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED_DATA_PROC")
                 mlflow.log_param("error", str(data_proc_err)[:250])
                 return -1.0

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

                # Pass the model_params dict which now correctly contains 'penalty'
                full_pipeline = create_full_pipeline(
                    classifier_class=classifier_class,
                    model_params=model_params,
                    preprocessor=preprocessor,
                    feature_selector_type=feature_selector_type,
                    feature_selector_params=feature_selector_params,
                    smote_active=smote_active
                )
            except Exception as pipe_err:
                 logger.error(f"Trial {trial.number}: Pipeline creation failed: {pipe_err}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED_PIPE_CREATION")
                 mlflow.log_param("error", str(pipe_err)[:250])
                 return -1.0

            # --- Cross-validation ---
            try:
                cv = StratifiedKFold(n_splits=HPO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                logger.debug(f"Trial {trial.number}: Starting cross-validation...")
                scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring=primary_scorer, n_jobs=1, error_score='raise')
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                logger.debug(f"Trial {trial.number}: CV Scores: {scores}, Mean: {mean_score:.4f}")

                mlflow.log_metric(f"{PRIMARY_METRIC}_cv_mean", mean_score)
                mlflow.log_metric(f"{PRIMARY_METRIC}_cv_std", std_score)
                mlflow.set_tag("optuna_trial_state", "COMPLETE")

                return mean_score

            except optuna.exceptions.TrialPruned as e:
                 logger.info(f"Trial {trial.number} pruned: {e}")
                 mlflow.set_tag("optuna_trial_state", "PRUNED")
                 raise e
            except Exception as e:
                 logger.error(f"Trial {trial.number} failed during CV: {e}", exc_info=True)
                 mlflow.set_tag("optuna_trial_state", "FAILED")
                 mlflow.log_param("error", str(e)[:250])
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
    Orchestrates HPO (Optuna ONLY), model selection, final training,
    evaluation, and registration. Uses hpo.py logic and logging structure.
    """
    logger.info("Starting Optuna-only optimization (hpo.py logic & logging) and training process...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Attrition Optuna Optimization (hpo.py logic)")

    best_overall_score = -1.0
    best_optuna_config = None # This will store the winning trial.params dict
    best_model_type_name = None

    with mlflow.start_run(run_name="Optuna_MultiModel_Optimize_Train_Select") as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Parent MLflow Run ID: {parent_run_id}")
        mlflow.log_param("optimization_method", "Optuna")
        mlflow.log_param("objective_logic", "hpo.py")
        mlflow.log_param("logging_structure", "run_per_trial")
        mlflow.log_param("models_considered", ", ".join(models_to_opt))
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("optuna_trials_per_model", HPO_N_TRIALS)
        mlflow.log_param("cv_folds", HPO_CV_FOLDS)

        try:
            df_raw = load_and_clean_data(RAW_DATA_PATH)
            df_raw[TARGET_COLUMN] = df_raw[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
            logger.info("Target variable encoded to 0/1.")

            X = df_raw.drop(TARGET_COLUMN, axis=1)
            y = df_raw[TARGET_COLUMN]
            if not isinstance(X, pd.DataFrame):
                logger.warning("Initial data X is not DataFrame. Attempting conversion.")
                try: X = pd.DataFrame(X)
                except Exception: logger.error("Failed to convert initial X to DataFrame."); raise

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            logger.info(f"Data split: Train shape={X_train.shape}, Test shape={X_test.shape}")
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
        except Exception as data_err:
             logger.error(f"Failed to load or split data: {data_err}", exc_info=True)
             mlflow.set_tag("status", "FAILED_DATA_LOAD")
             sys.exit(1)

        optuna_results = {}
        for model_type in models_to_opt:
            if model_type not in CLASSIFIER_MAP or model_type not in PARAM_FUNC_MAP:
                logger.warning(f"Skipping Optuna HPO for '{model_type}': Class or Param function not defined locally.")
                continue

            logger.info(f"--- Starting Optuna HPO for: {model_type} ---")
            with mlflow.start_run(run_name=f"Optuna_{model_type}_Study", nested=True) as study_run:
                mlflow.log_param("model_type", model_type)
                study_run_id = study_run.info.run_id
                study_name = f"optuna_{model_type}_{study_run_id[:8]}"
                study = optuna.create_study(direction="maximize", study_name=study_name)

                try:
                    study.optimize(
                        lambda trial: optuna_objective(trial, model_type, X_train, y_train),
                        n_trials=HPO_N_TRIALS,
                        gc_after_trial=True
                    )

                    best_trial = study.best_trial
                    current_best_score = best_trial.value if best_trial else -1.0

                    if best_trial:
                        optuna_results[model_type] = {
                            "score": current_best_score,
                            "params": best_trial.params, # Store Optuna's view of params
                            "study_run_id": study_run_id,
                            "best_trial_number": best_trial.number
                        }
                        mlflow.log_metric("best_cv_score", current_best_score)
                        # Log best params (filtered) to the study run
                        params_to_log = {k:v for k,v in best_trial.params.items() if isinstance(v, (str, int, float, bool))}
                        mlflow.log_params({f"best_{k}":v for k,v in params_to_log.items()})
                        mlflow.set_tag("status", "COMPLETED")
                        logger.info(f"Optuna HPO study for {model_type} complete. Best Score: {current_best_score:.4f}")

                        if current_best_score > best_overall_score:
                            best_overall_score = current_best_score
                            # Store the best trial's params directly from Optuna
                            best_optuna_config = best_trial.params.copy()
                            best_optuna_config['model_type'] = model_type # Add model type marker
                            best_model_type_name = model_type
                            logger.info(f"New best overall model found: Optuna {model_type} (Score: {best_overall_score:.4f})")
                    else:
                         logger.error(f"Optuna study for {model_type} finished without a best trial.")
                         mlflow.set_tag("status", "FAILED_NO_TRIAL")
                         optuna_results[model_type] = {"score": -1.0, "params": {}, "study_run_id": study_run_id}

                except Exception as study_err:
                     if isinstance(study_err, mlflow.exceptions.MlflowException):
                          logger.error(f"MLflow logging failed during Optuna study for {model_type}: {study_err}", exc_info=False)
                     else:
                          logger.error(f"Optuna study for {model_type} failed: {study_err}", exc_info=True)
                     try: mlflow.set_tag("status", "FAILED")
                     except Exception: pass
                     optuna_results[model_type] = {"score": -1.0, "params": {}, "study_run_id": study_run_id}


        # --- Final Training and Evaluation ---
        if best_optuna_config is None:
            logger.error("No successful Optuna optimization runs completed. Cannot train final model.")
            mlflow.end_run()
            sys.exit(1)

        logger.info(f"--- Training Final Best Model (Optuna: {best_model_type_name}) ---")
        with mlflow.start_run(run_name=f"Final_Training_Optuna_{best_model_type_name}", nested=True) as final_run:
            final_run_id = final_run.info.run_id
            mlflow.log_param("winning_hpo_method", "Optuna")
            mlflow.log_param("winning_model_type", best_model_type_name)
            mlflow.log_metric("best_cv_score_from_hpo", best_overall_score)

            # Prepare final model parameters, ensuring correct 'penalty' key
            final_model_params = {}
            temp_penalty_key = None
            for key, value in best_optuna_config.items():
                 if key.startswith('model_'):
                      # Handle the penalty key specifically for logreg
                      if best_model_type_name == 'logistic_regression':
                           if key == 'model_penalty' or key == 'model_penalty_saga':
                                final_model_params['penalty'] = value # Always use 'penalty'
                                temp_penalty_key = key # Remember the original Optuna key
                           elif key != 'model_type': # Exclude the marker
                                final_model_params[key.replace('model_', '', 1)] = value
                      elif key != 'model_type': # For other models
                           final_model_params[key.replace('model_', '', 1)] = value

            # Log the final parameters used (excluding temporary Optuna keys)
            mlflow.log_params({f"final_{k}": v for k,v in final_model_params.items()})
            # Also log the original Optuna params for reference (filtered)
            orig_params_log = {k:v for k,v in best_optuna_config.items() if k != 'model_type' and isinstance(v, (str, int, float, bool))}
            mlflow.log_params({f"optuna_{k}": v for k,v in orig_params_log.items()})


            final_pipeline = None
            try:
                logger.info("Rebuilding pipeline from best Optuna parameters...")
                model_type = best_optuna_config['model_type']
                classifier_class = CLASSIFIER_MAP[model_type]
                # Extract pipeline and feature selector params from best_optuna_config
                pipeline_params = {k.replace('pipe_', ''): v for k, v in best_optuna_config.items() if k.startswith('pipe_')}
                feature_selector_params_final = {k.replace('selector_', ''): v for k, v in best_optuna_config.items() if k.startswith('selector_')}

                # Replicate hpo.py logic for final pipeline setup
                temp_feature_adder = AddNewFeaturesTransformer()
                if not isinstance(X_train, pd.DataFrame):
                     raise TypeError("X_train must be DataFrame for final pipeline creation steps.")
                X_train_eng_temp = temp_feature_adder.fit_transform(X_train.copy())
                col_types = identify_column_types(X_train_eng_temp, None)
                skewed_cols = find_skewed_columns(X_train_eng_temp, col_types['numerical'], threshold=SKEWNESS_THRESHOLD)
                del X_train_eng_temp

                preprocessor = create_preprocessing_pipeline(
                    numerical_cols=col_types['numerical'],
                    categorical_cols=col_types['categorical'],
                    ordinal_cols=col_types['ordinal'],
                    business_travel_col=col_types['business_travel'],
                    skewed_cols=skewed_cols,
                    numeric_transformer_type=pipeline_params.get('num_transform'),
                    numeric_scaler_type=pipeline_params.get('num_scaler'),
                    business_encoder_type=pipeline_params.get('bt_encoder'),
                )
                # Create final pipeline using the cleaned final_model_params
                final_pipeline = create_full_pipeline(
                    classifier_class=classifier_class,
                    model_params=final_model_params, # Use the cleaned params dict
                    preprocessor=preprocessor,
                    feature_selector_type=pipeline_params.get('selector'),
                    feature_selector_params=feature_selector_params_final,
                    smote_active=pipeline_params.get('smote')
                )

                # --- Actual Training on Full Training Data ---
                logger.info("Fitting final Optuna pipeline on full training data...")
                start_time = time.time()
                final_pipeline.fit(X_train, y_train)
                end_time = time.time()
                logger.info(f"Pipeline fitting complete. Time taken: {end_time - start_time:.2f} seconds")
                mlflow.log_metric("training_time_seconds", end_time - start_time)

                # --- Evaluation on Test Set ---
                logger.info("Evaluating final model on test set...")
                y_pred_test = final_pipeline.predict(X_test)
                test_metrics = {
                    "test_accuracy": accuracy_score(y_test, y_pred_test),
                    "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
                    f"test_{PRIMARY_METRIC}": fbeta_score(y_test, y_pred_test, beta=beta_value, zero_division=0),
                    "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
                    "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
                }
                if hasattr(final_pipeline, "predict_proba"):
                    try:
                        y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
                        test_metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC: {e}")
                        test_metrics["test_auc"] = -1.0
                else:
                    test_metrics["test_auc"] = -1.0

                mlflow.log_metrics(test_metrics)
                logger.info(f"Final Test Metrics: {test_metrics}")

                # --- Log Confusion Matrix ---
                try:
                    cm = confusion_matrix(y_test, y_pred_test)
                    cm_path = os.path.join(REPORTS_PATH, f"confusion_matrix_{final_run_id}.json")
                    os.makedirs(REPORTS_PATH, exist_ok=True)
                    save_json({"labels": [0, 1], "matrix": cm.tolist()}, cm_path)
                    mlflow.log_artifact(cm_path)
                except Exception as e:
                    logger.error(f"Could not log confusion matrix: {e}")

                # --- Log and Register Final Model ---
                logger.info(f"Logging and registering the final model: {PRODUCTION_MODEL_NAME}")
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path="final_model_pipeline",
                    registered_model_name=PRODUCTION_MODEL_NAME
                )

                # --- Transition Model to Production Stage ---
                client = mlflow.tracking.MlflowClient()
                try:
                    time.sleep(5)
                    latest_versions = client.get_latest_versions(PRODUCTION_MODEL_NAME, stages=["None"])
                    if latest_versions:
                        model_version = latest_versions[0].version
                        logger.info(f"Transitioning model version {model_version} of '{PRODUCTION_MODEL_NAME}' to Production.")
                        client.transition_model_version_stage(
                            name=PRODUCTION_MODEL_NAME,
                            version=model_version,
                            stage="Production",
                            archive_existing_versions=True
                        )
                        mlflow.set_tag("model_registered_stage", "Production")
                    else:
                        logger.error("Could not find the newly registered model version 'None' stage to transition.")
                        mlflow.set_tag("model_registration_status", "Transition_Failed")

                except Exception as e:
                    logger.error(f"Failed to transition model to Production stage: {e}", exc_info=True)
                    mlflow.set_tag("model_registration_status", "Transition_Failed")

                mlflow.set_tag("status", "COMPLETED")
                logger.info("Final model training, evaluation, and registration complete.")

            except Exception as final_train_err:
                logger.error(f"Final model training/evaluation failed: {final_train_err}", exc_info=True)
                mlflow.set_tag("status", "FAILED_FINAL_TRAINING")

        # Log overall results to parent run
        mlflow.log_metric("best_overall_cv_score", best_overall_score)
        mlflow.log_param("winning_model_type", best_model_type_name if best_model_type_name else "None")
        mlflow.set_tag("status", "COMPLETED_PROCESS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna HPO (hpo.py logic), select best model, train, and register.")
    args = parser.parse_args()

    if 'MODELS_TO_OPTIMIZE' not in globals() and 'MODELS_TO_OPTIMIZE' not in locals():
         try: from employee_attrition_mlops.config import MODELS_TO_OPTIMIZE
         except ImportError: logger.error("MODELS_TO_OPTIMIZE not defined. Check config.py import."); sys.exit(1)

    models_list = MODELS_TO_OPTIMIZE
    optimize_select_and_train(models_list)
    logger.info("Script finished.")

