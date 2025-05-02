#!/usr/bin/env python3
"""
Script to check for drift using the drift API endpoints.
Loads batch prediction data from reports directory and makes API requests.
"""
import json
import requests
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
DRIFT_API_URL = "http://localhost:8001"  # Default port for drift-api in docker-compose
FEATURE_DRIFT_ENDPOINT = f"{DRIFT_API_URL}/drift/feature"
PREDICTION_DRIFT_ENDPOINT = f"{DRIFT_API_URL}/drift/prediction"

# Report paths
REPORTS_DIR = "reports"
BATCH_FEATURES_PATH = os.path.join(REPORTS_DIR, "batch_features.json")
BATCH_PREDICTIONS_PATH = os.path.join(REPORTS_DIR, "batch_predictions.json")
FEATURE_DRIFT_RESULTS_PATH = os.path.join(REPORTS_DIR, "feature_drift_results.json")
PREDICTION_DRIFT_RESULTS_PATH = os.path.join(REPORTS_DIR, "prediction_drift_results.json")

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved results to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to {file_path}: {e}")

def check_feature_drift():
    """Check for feature drift using the API."""
    # Load batch features
    features_data = load_json_file(BATCH_FEATURES_PATH)
    if not features_data:
        logger.error("No feature data available. Run batch_predict.py first.")
        return False
    
    logger.info(f"Sending feature drift check request with {len(features_data)} records")
    
    # Prepare request payload
    payload = {"data": features_data}
    
    try:
        # Make API request
        response = requests.post(FEATURE_DRIFT_ENDPOINT, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Process response
        result = response.json()
        logger.info(f"Feature drift check complete. Drift detected: {result.get('dataset_drift', False)}")
        
        # Add timestamp if not present
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
            
        # Save results
        save_json_file(result, FEATURE_DRIFT_RESULTS_PATH)
        
        return result.get('dataset_drift', False)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return False

def check_prediction_drift():
    """Check for prediction drift using the API."""
    # Load batch predictions
    predictions_data = load_json_file(BATCH_PREDICTIONS_PATH)
    if not predictions_data:
        logger.error("No prediction data available. Run batch_predict.py first.")
        return False
    
    logger.info(f"Sending prediction drift check request with {len(predictions_data)} records")
    
    # Prepare request payload
    payload = {"data": predictions_data}
    
    try:
        # Make API request
        response = requests.post(PREDICTION_DRIFT_ENDPOINT, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Process response
        result = response.json()
        logger.info(f"Prediction drift check complete. Drift detected: {result.get('prediction_drift_detected', False)}")
        
        # Add timestamp if not present
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
            
        # Save results
        save_json_file(result, PREDICTION_DRIFT_RESULTS_PATH)
        
        return result.get('prediction_drift_detected', False)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return False

def main():
    """Run both drift checks."""
    logger.info("Starting drift checks via API...")
    
    # Check feature drift
    feature_drift_detected = check_feature_drift()
    
    # Check prediction drift
    prediction_drift_detected = check_prediction_drift()
    
    # Summarize results
    logger.info("=== Drift Check Summary ===")
    logger.info(f"Feature drift detected: {feature_drift_detected}")
    logger.info(f"Prediction drift detected: {prediction_drift_detected}")
    
    if feature_drift_detected or prediction_drift_detected:
        logger.warning("DRIFT DETECTED! Review drift reports for details.")
    else:
        logger.info("No drift detected in either features or predictions.")
    
    logger.info(f"Results saved to {FEATURE_DRIFT_RESULTS_PATH} and {PREDICTION_DRIFT_RESULTS_PATH}")

if __name__ == "__main__":
    main() 