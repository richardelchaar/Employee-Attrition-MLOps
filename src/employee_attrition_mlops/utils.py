# src/employee_attrition_mlops/utils.py
import json
import joblib
import pandas as pd
import logging
import os
import mlflow
from mlflow.tracking import MlflowClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd


logger = logging.getLogger(__name__)

def save_json(data, file_path):
    """Saves data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def save_object(obj, file_path):
    """Saves a Python object using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")

def load_object(file_path):
    """Loads a Python object using joblib."""
    try:
        obj = joblib.load(file_path)
        logger.info(f"Successfully loaded object from {file_path}")
        return obj
    except FileNotFoundError:
        logger.error(f"Object file not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        return None

def get_production_model_run_id(model_name: str, stage: str = "Production") -> str | None:
    """Gets the run_id of the latest model version in a specific stage."""
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if latest_versions:
            run_id = latest_versions[0].run_id
            logger.info(f"Found run_id '{run_id}' for model '{model_name}' stage '{stage}'.")
            return run_id
        else:
            logger.warning(f"No model version found for model '{model_name}' stage '{stage}'.")
            return None
    except Exception as e:
        logger.error(f"Error fetching production model run_id for '{model_name}': {e}")
        return None

def download_mlflow_artifact(run_id: str, artifact_path: str, dst_path: str = None) -> str | None:
        """Downloads an artifact from a specific MLflow run."""
        client = MlflowClient()
        try:
            local_path = client.download_artifacts(run_id, artifact_path, dst_path)
            logger.info(f"Downloaded artifact '{artifact_path}' from run '{run_id}' to '{local_path}'.")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download artifact '{artifact_path}' from run '{run_id}': {e}")
            return None

def generate_evidently_profile(current_data: pd.DataFrame, reference_data: pd.DataFrame = None):
    """Generates a comprehensive data profile using Evidently AI."""
    if reference_data is None:
        # Creating a profile without comparison data
        profile = Report(metrics=[DataDriftPreset()])
        profile.run(current_data=current_data)
    else:
        # Creating a profile comparing to reference data
        profile = Report(metrics=[DataDriftPreset()])
        profile.run(reference_data=reference_data, current_data=current_data)
    
    return profile