from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
import json

from .drift_detection import DriftDetector
from .config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Drift Monitoring API")
settings = get_settings()

# Initialize drift detector
drift_detector = DriftDetector(
    reference_data_path=os.path.join("drift_reference", "reference_train_data.parquet"),
    drift_threshold=settings.DRIFT_THRESHOLD
)

class DriftCheckRequest(BaseModel):
    data: List[Dict[str, Any]]
    timestamp: datetime

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    drifted_features: List[str]
    timestamp: datetime
    details: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/drift/check", response_model=DriftResponse)
async def check_drift(request: DriftCheckRequest):
    """
    Check for data drift in the provided data against the reference dataset.
    """
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Check for drift
        drift_results = drift_detector.check_drift(df)
        
        # Save drift artifacts
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        drift_artifacts_dir = os.path.join("drift_artifacts", timestamp_str)
        os.makedirs(drift_artifacts_dir, exist_ok=True)
        
        # Save drift report
        drift_report = {
            "timestamp": timestamp_str,
            "drift_detected": drift_results["drift_detected"],
            "drift_score": drift_results["drift_score"],
            "drifted_features": drift_results["drifted_features"],
            "details": drift_results["details"]
        }
        
        with open(os.path.join(drift_artifacts_dir, "drift_report.json"), "w") as f:
            json.dump(drift_report, f, indent=2)
        
        return DriftResponse(
            drift_detected=drift_results["drift_detected"],
            drift_score=drift_results["drift_score"],
            drifted_features=drift_results["drifted_features"],
            timestamp=datetime.now(),
            details=drift_results["details"]
        )
        
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/reference-info")
async def get_reference_info():
    """
    Get information about the reference dataset used for drift detection.
    """
    try:
        reference_info = drift_detector.get_reference_info()
        return reference_info
    except Exception as e:
        logger.error(f"Error getting reference info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/history")
async def get_drift_history():
    """
    Get the history of drift checks.
    """
    try:
        drift_artifacts_dir = "drift_artifacts"
        if not os.path.exists(drift_artifacts_dir):
            return []
            
        history = []
        for timestamp_dir in sorted(os.listdir(drift_artifacts_dir), reverse=True):
            report_path = os.path.join(drift_artifacts_dir, timestamp_dir, "drift_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    history.append(json.load(f))
                    
        return history
    except Exception as e:
        logger.error(f"Error getting drift history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 