# scripts/train.py
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, fbeta_score,
                                precision_score, recall_score, roc_auc_score,
                                confusion_matrix)
import sys
import os
import logging
import json

def load_json(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from your source code
from employee_attrition_mlops.config import (RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE,
                                            RANDOM_STATE, MODEL_CONFIGS,
                                            MLFLOW_TRACKING_URI, DEFAULT_EXPERIMENT_NAME,
                                            BASELINE_PROFILE_FILENAME)
from employee_attrition_mlops.data_processing import (load_and_clean_data, identify_column_types,
                                                        find_skewed_columns, AddNewFeaturesTransformer)
from employee_attrition_mlops.pipelines import create_preprocessing_pipeline, create_full_pipeline
from employee_attrition_mlops.utils import save_json, generate_profile_dict

def train_model(model_alias: str, register_model: bool = False):
    """
    Loads data, preprocesses, trains, evaluates, and logs a model specified by its alias.
    Optionally registers the model if it's deemed the best.
    """
    logger.info(f"Starting training process for model alias: {model_alias}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # --- Load Configuration ---
    if model_alias not in MODEL_CONFIGS:
        logger.error(f"Model alias '{model_alias}' not found in MODEL_CONFIGS.")
        raise ValueError(f"Invalid model alias: {model_alias}")
    config = MODEL_CONFIGS[model_alias]
    pipeline_params = config.get("pipeline_params", {})
    model_params = config.get("model_params", {})
    experiment_name = config.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
    registered_model_name = config.get("registered_model_name", None) if register_model else None
    artifact_path = config.get("artifact_path", "model_pipeline")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_alias}_final_training"):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        # Log config parameters
        mlflow.log_param("model_alias", model_alias)
        mlflow.log_params({f"pipe_{k}": v for k, v in pipeline_params.items() if k != 'classifier_class'})
        mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # --- Load and Prepare Data ---
        df = load_and_clean_data(RAW_DATA_PATH)
        # Apply initial feature engineering that might affect column types/skewness
        feature_adder = AddNewFeaturesTransformer()
        df = feature_adder.fit_transform(df)
        logger.info("Applied AddNewFeaturesTransformer.")

        # Identify column types AFTER initial feature engineering
        col_types = identify_column_types(df, TARGET_COLUMN)
        numerical_cols = col_types['numerical']
        categorical_cols = col_types['categorical']
        ordinal_cols = col_types['ordinal']
        business_travel_col = col_types['business_travel'] # List: ['BusinessTravel'] or []

        # Find skewed columns for transformation decision
        skewed_cols = find_skewed_columns(df, numerical_cols)
        mlflow.log_param("skewed_features_count", len(skewed_cols))

        # --- Train/Test Split ---
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # --- Create Pipelines ---
        preprocessor = create_preprocessing_pipeline(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            ordinal_cols=ordinal_cols,
            business_travel_col=business_travel_col,
            skewed_cols=skewed_cols, # Pass identified skewed cols
            numeric_transformer_type=pipeline_params.get('numeric_transformer_type'),
            numeric_scaler_type=pipeline_params.get('numeric_scaler_type'),
            business_encoder_type=pipeline_params.get('business_encoder_type'),
            # outlier_remover_active=pipeline_params.get('outlier_remover_active')
        )

        full_pipeline = create_full_pipeline(
            classifier_class=pipeline_params.get('classifier_class'),
            model_params=model_params,
            preprocessor=preprocessor,
            feature_selector_type=pipeline_params.get('feature_selector_type'),
            feature_selector_params=pipeline_params.get('feature_selector_params'),
            smote_active=pipeline_params.get('smote_active'),
            # outlier_remover_active=pipeline_params.get('outlier_remover_active')
        )
        logger.info("Created full pipeline.")

        # --- Training ---
        logger.info("Fitting full pipeline...")
        full_pipeline.fit(X_train, y_train)
        logger.info("Pipeline fitting complete.")

        # --- Generate & Log Training Data Profile ---
        try:
            # Profile data *after* preprocessing step within the final fitted pipeline
            preprocessor_fitted = full_pipeline.named_steps['preprocessor']
            X_train_processed = preprocessor_fitted.transform(X_train)
            profile_dict = generate_profile_dict(X_train_processed) # Use helper from utils
            save_json(profile_dict, BASELINE_PROFILE_FILENAME)
            mlflow.log_artifact(BASELINE_PROFILE_FILENAME)
            logger.info(f"Logged training data profile to {BASELINE_PROFILE_FILENAME}")
            # Save feature names after preprocessing for later use
            processed_feature_names = list(X_train_processed.columns)
            mlflow.log_param("processed_feature_count", len(processed_feature_names))
            # Log as param (might be long) or artifact
            save_json(processed_feature_names, "processed_feature_names.json")
            mlflow.log_artifact("processed_feature_names.json")

        except Exception as e:
            logger.error(f"Error generating/logging training data profile: {e}")
            processed_feature_names = [] # Fallback

        # --- Evaluation ---
        logger.info("Evaluating model...")
        y_train_pred = full_pipeline.predict(X_train)
        y_test_pred = full_pipeline.predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "train_f2": fbeta_score(y_train, y_train_pred, beta=2),
            "test_f2": fbeta_score(y_test, y_test_pred, beta=2), # Key metric
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
        }
        if hasattr(full_pipeline, "predict_proba"):
            try:
                y_test_proba = full_pipeline.predict_proba(X_test)[:, 1]
                metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics["test_auc"] = -1 # Indicate failure
        else: metrics["test_auc"] = -1

        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # --- Log Confusion Matrix ---
        try:
            cm = confusion_matrix(y_test, y_test_pred)
            cm_path = "confusion_matrix.json"
            save_json({"labels": [0, 1], "matrix": cm.tolist()}, cm_path)
            mlflow.log_artifact(cm_path)
            logger.info(f"Logged confusion matrix to {cm_path}")
        except Exception as e:
            logger.error(f"Could not log confusion matrix: {e}")

        # --- Log Model ---
        logger.info(f"Logging model pipeline to artifact path: {artifact_path}")
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path=artifact_path,
            # Register model if flag is set and name provided
            registered_model_name=registered_model_name
        )
        if registered_model_name:
            logger.info(f"Model registered as: {registered_model_name}")
        else:
                logger.info("Model logged, but not registered.")

        # --- Feature Importance ---
        # Accessing feature importance after potential feature selection requires care
        try:
            final_estimator = full_pipeline.named_steps['classifier']
            feature_selector_step = full_pipeline.named_steps.get('feature_selection', None)
            preprocessor_step = full_pipeline.named_steps.get('preprocessor', None)

            if hasattr(final_estimator, 'feature_importances_') or hasattr(final_estimator, 'coef_'):
                # Get feature names *after* preprocessing and selection
                if preprocessor_step:
                    # Get names after preprocessing
                    try:
                            all_processed_names = preprocessor_step.get_feature_names_out()
                    except: # Fallback for older sklearn or complex transformers
                            # Attempt to get from earlier profiling step artifact
                            try:
                                local_names_path = mlflow.download_artifacts("processed_feature_names.json")
                                all_processed_names = load_json(local_names_path)
                            except:
                                logger.warning("Cannot reliably get processed feature names for importance.")
                                all_processed_names = None

                    selected_names = all_processed_names # Assume all pass if no selector or passthrough

                    # If feature selection happened, get the mask/indices
                    if feature_selector_step and feature_selector_step != 'passthrough' and all_processed_names is not None:
                            try:
                                support_mask = feature_selector_step.get_support()
                                selected_names = np.array(all_processed_names)[support_mask].tolist()
                            except Exception as fe_err:
                                logger.warning(f"Could not get selected feature names: {fe_err}")
                                selected_names = None # Cannot map importance reliably

                    if selected_names:
                        if hasattr(final_estimator, 'feature_importances_'):
                            importances = final_estimator.feature_importances_
                        elif hasattr(final_estimator, 'coef_'):
                            importances = final_estimator.coef_[0] # For linear models

                        if len(importances) == len(selected_names):
                            imp_df = pd.DataFrame({'feature': selected_names, 'importance': importances})
                            imp_df = imp_df.sort_values('importance', key=abs, ascending=False)
                            imp_path = "feature_importances.csv"
                            imp_df.to_csv(imp_path, index=False)
                            mlflow.log_artifact(imp_path)
                            logger.info(f"Logged feature importances to {imp_path}")
                        else:
                            logger.warning(f"Mismatch between importance values ({len(importances)}) and selected feature names ({len(selected_names)}). Skipping importance logging.")
                    else:
                            logger.warning("Could not determine final feature names for importance logging.")

        except Exception as e:
            logger.error(f"Could not log feature importances: {e}")

        logger.info(f"Training complete for {model_alias}. Test F2 Score: {metrics['test_f2']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Employee Attrition Model")
    parser.add_argument(
        "--model-alias",
        type=str,
        required=True,
        help=f"Alias of the model configuration to use from config.py (e.g., {list(MODEL_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--register",
        action="store_true", # Flag to register the model
        help="Register the trained model in MLflow Model Registry.",
    )
    args = parser.parse_args()

    train_model(model_alias=args.model_alias, register_model=args.register)
    