from fastapi import FastAPI, HTTPException, Request
import mlflow
import mlflow.pyfunc
import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
import time
from .config import (
    DATABASE_URL_PYMSSQL, MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME,
    DB_PREDICTION_LOG_TABLE
)
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

model = None
model_info = {}
engine = None

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds

def load_production_model(model_name: str):
    """Load the latest model version from MLflow."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting to load model '{model_name}' from registry.")
            client = mlflow.tracking.MlflowClient()
            registered_model = client.get_registered_model(model_name)
            if not registered_model.latest_versions:
                raise Exception(f"No versions found for model '{model_name}'")

            # Get the latest version (assuming highest version number is latest)
            latest_version_info = max(registered_model.latest_versions, key=lambda v: int(v.version))
            logger.info(f"Found latest version: {latest_version_info.version}")

            loaded_model = mlflow.sklearn.load_model(latest_version_info.source)
            logger.info(f"Model '{model_name}' version {latest_version_info.version} loaded successfully.")
            return loaded_model, {
                "name": latest_version_info.name,
                "version": latest_version_info.version,
                "source": latest_version_info.source,
                "run_id": latest_version_info.run_id,
                "status": latest_version_info.status,
                "current_stage": latest_version_info.current_stage # Still useful info even if not used for loading
            }
        except Exception as e:
            logger.error(f"Error loading model (attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Failed to load model.")
                return None, {}
    return None, {}

@app.on_event("startup")
async def startup_event():
    """Load model and initialize database connection on startup."""
    global model, model_info, engine
    logger.info("API starting up...")

    # Load Model
    model, model_info = load_production_model(PRODUCTION_MODEL_NAME)
    if model is None:
        logger.error("Model could not be loaded on startup. API might not function correctly.")
        # Decide if you want to raise an error or allow startup without model
        # raise RuntimeError("Failed to load production model on startup.")
    else:
        logger.info("Production model loaded successfully.")

    # Initialize Database Connection using PYMSSQL URL
    if DATABASE_URL_PYMSSQL:
        try:
            logger.info("Attempting to connect to database using pymssql driver (URL from config).")
            engine = create_engine(DATABASE_URL_PYMSSQL)
            # Test connection
            with engine.connect() as connection:
                 logger.info("Database connection (pymssql) established successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed on startup (pymssql): {e}", exc_info=True)
            engine = None # Ensure engine is None if connection fails
        except Exception as e:
             logger.error(f"Unexpected error initializing database connection (pymssql): {e}", exc_info=True)
             engine = None
    else:
        logger.warning("DATABASE_URL_PYMSSQL not set. Prediction logging to DB will be disabled.")
        engine = None

@app.on_event("shutdown")
async def shutdown_event():
    """Dispose database engine on shutdown."""
    global engine
    if engine:
        logger.info("Disposing database engine (pymssql)...")
        engine.dispose()
        logger.info("Database engine disposed.")

@app.get("/health")
async def health():
    """Health check endpoint with model information."""
    if model is None:
        return {"status": "error", "model_loaded": False, "error": "Model not loaded yet"}
    try:
        # Get current model version info from the client
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_registered_model(PRODUCTION_MODEL_NAME)
        versions = sorted(model_info.latest_versions, key=lambda v: int(v.version), reverse=True)
        if not versions:
             return {"status": "error", "model_loaded": False, "error": f"No versions found for model '{PRODUCTION_MODEL_NAME}'"}
        latest_version = versions[0]
        
        return {
            "status": "ok",
            "model_loaded": True,
            "registered_model_name": PRODUCTION_MODEL_NAME,
            "loaded_model_version": latest_version.version, # Ideally, we'd store the loaded version at startup
            "loaded_model_run_id": latest_version.run_id  # Ideally, we'd store the loaded run_id at startup
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "model_loaded": True, # Model object exists, but fetching info failed
            "error": f"Failed to retrieve latest model info: {str(e)}"
        }

@app.get("/model-info")
async def model_info_endpoint(): # Renamed to avoid conflict with variable name
    """Get detailed information about the latest registered model version."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        client = mlflow.tracking.MlflowClient()
        registered_model_info = client.get_registered_model(PRODUCTION_MODEL_NAME)
        versions = sorted(registered_model_info.latest_versions, key=lambda v: int(v.version), reverse=True)
        if not versions:
            raise HTTPException(status_code=404, detail=f"No versions found for registered model '{PRODUCTION_MODEL_NAME}'")
            
        latest_version_info = versions[0] # Get the latest version info
        
        # It might be better to return the info of the *actually loaded* model
        # For now, returning the latest registered info
        return {
            "registered_model_name": PRODUCTION_MODEL_NAME,
            "latest_registered_version": latest_version_info.version,
            "latest_registered_run_id": latest_version_info.run_id,
            "latest_registered_status": latest_version_info.status,
            "latest_registered_creation_timestamp": latest_version_info.creation_timestamp
            # Add info about the currently loaded model if stored at startup
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest registered model info: {str(e)}")

@app.post("/predict")
async def predict(request: Request):
    """Make predictions using the loaded model."""
    global engine # Ensure we use the engine initialized at startup
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot make predictions.")

    try:
        data = await request.json()
        logger.info(f"Received prediction request with data: {data}")
        
        emp = data.get("EmployeeNumber")
        snap = data.get("SnapshotDate")
        
        if emp is None or snap is None:
            raise HTTPException(status_code=400, detail="EmployeeNumber and SnapshotDate are required in request.")
        
        # Create a copy of the data for prediction, keeping EmployeeNumber
        payload = data.copy()
        # Only remove SnapshotDate as it's not needed for prediction
        payload.pop("SnapshotDate")
        
        # Define required columns and their default values
        required_columns = {
            'EmployeeNumber': int,
            'Age': int,
            'Gender': str,
            'MaritalStatus': str,
            'Department': str,
            'EducationField': str,
            'JobLevel': int,
            'JobRole': str,
            'BusinessTravel': str,
            'DistanceFromHome': int,
            'Education': int,
            'DailyRate': int,
            'HourlyRate': int,
            'MonthlyIncome': int,
            'MonthlyRate': int,
            'PercentSalaryHike': int,
            'StockOptionLevel': int,
            'OverTime': str,
            'NumCompaniesWorked': int,
            'TotalWorkingYears': int,
            'TrainingTimesLastYear': int,
            'YearsAtCompany': int,
            'YearsInCurrentRole': int,
            'YearsSinceLastPromotion': int,
            'YearsWithCurrManager': int,
            'EnvironmentSatisfaction': int,
            'JobInvolvement': int,
            'JobSatisfaction': int,
            'PerformanceRating': int,
            'RelationshipSatisfaction': int,
            'WorkLifeBalance': int,
            'AgeGroup': str
        }
        
        # Ensure all required columns are present with default values if missing
        for col, dtype in required_columns.items():
            if col not in payload:
                if dtype == int:
                    payload[col] = 0
                elif dtype == str:
                    payload[col] = "Unknown"
            else:
                # Convert values to correct type
                try:
                    if dtype == int:
                        payload[col] = int(payload[col])
                    elif dtype == str:
                        payload[col] = str(payload[col])
                except Exception as e:
                    logger.error(f"Error converting {col} to {dtype}: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {payload[col]}")
        
        try:
            input_df = pd.DataFrame([payload])
            # Ensure columns are in the correct order
            input_df = input_df[list(required_columns.keys())]
            logger.info(f"Created input DataFrame with columns: {input_df.columns.tolist()}")
            logger.info(f"DataFrame shape: {input_df.shape}")
            logger.info(f"DataFrame dtypes: {input_df.dtypes}")
            logger.info(f"DataFrame head: {input_df.head().to_dict()}")
            
            pred = int(model.predict(input_df)[0])  # Convert numpy.int64 to Python int
            logger.info(f"Prediction successful: {pred}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        # Log prediction to DB if database connection exists
        if engine is not None:
            try:
                with engine.begin() as conn:
                    # First check if prediction already exists
                    check_sql = text(f"""
                    SELECT COUNT(*) FROM {DB_PREDICTION_LOG_TABLE}
                    WHERE EmployeeNumber = :emp AND SnapshotDate = :snap
                    """)
                    result = conn.execute(check_sql, {"emp": emp, "snap": snap}).scalar()
                    
                    if result == 0:
                        # Only insert if prediction doesn't exist
                        insert_sql = text(f"""
                        INSERT INTO {DB_PREDICTION_LOG_TABLE} (EmployeeNumber, SnapshotDate, Prediction)
                        VALUES (:emp, :snap, :pred)
                        """)
                        conn.execute(insert_sql, {"emp": emp, "snap": snap, "pred": str(pred)})
                    else:
                        logger.info(f"Prediction already exists for EmployeeNumber {emp} on {snap}")
            except Exception as e:
                logger.error(f"Failed to log prediction to DB (pymssql): {e}", exc_info=True)
                # Don't raise the error, just log it and continue
        
        return {
            "EmployeeNumber": emp,
            "SnapshotDate": snap,
            "prediction": pred
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        # Ensure HTTPException is raised for proper FastAPI error handling
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Main Block (for local testing if needed) ---
if __name__ == "__main__":
    import uvicorn
    # Note: When running directly, ensure DB is accessible and model can be loaded
    # Might need to set MLFLOW_TRACKING_URI env var
    logger.info("Starting API locally via uvicorn...")
    # Load DATABASE_URL_PYMSSQL for local testing of API
    if not DATABASE_URL_PYMSSQL:
        logger.warning("DATABASE_URL_PYMSSQL not set, DB logging disabled for local run.")
    uvicorn.run(app, host="127.0.0.1", port=8000)