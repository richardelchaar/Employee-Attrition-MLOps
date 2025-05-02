# MLOps Workflow Guide: Employee Attrition Project

This guide provides a comprehensive overview of the MLOps workflow implemented for the Employee Attrition prediction project. It details the components, data flow, automation, and key processes involved in training, deploying, monitoring, and retraining the machine learning model.

## 1. Overview

The goal is to create a system that reliably predicts employee attrition and keeps itself up-to-date with minimal manual intervention. The workflow automates:
1.  **Training & Selection**: Finding the best way to configure and train a model based on historical data.
2.  **Deployment**: Making the best model usable through an API (like a web service).
3.  **Batch Prediction**: Regularly (e.g., monthly) using the model to predict attrition for all current employees.
4.  **Monitoring for Changes (Drift)**: Automatically checking if the characteristics of current employees (features) or the model's prediction patterns are changing significantly compared to when the model was initially deployed.
5.  **Automatic Retraining**: Triggering a new training process if significant changes (drift) are detected.
6.  **Visualization**: Providing a web dashboard to see prediction examples and monitoring results.

The system uses Docker (to package the software components), MLflow (to track experiments and manage models), GitHub Actions (to automate the regular checks and retraining), FastAPI (for the prediction API), and Streamlit (for the web dashboard).

## 2. Core Components Explained

Think of the system as having several key parts:

### 2.1. Key Instruction Manuals (Scripts)

These Python scripts contain the detailed instructions for specific tasks:

*   **`scripts/optimize_train_select.py` (The Trainer & Selector)**:
    *   This is the main script for teaching the model.
    *   It fetches historical employee data.
    *   It tries different model types and settings (hyperparameter optimization with Optuna) to find the best combination.
    *   It trains the final model using the best settings found.
    *   It evaluates how well the model performs (checking accuracy, fairness, etc.).
    *   It records *everything* about this training process (settings, results, the model itself) in the MLflow library.
    *   It marks the best model found in MLflow as a candidate for official use.
*   **`scripts/save_reference_data.py` (The Baseline Setter)**:
    *   **Purpose**: To create a "snapshot" or "baseline" of what things looked like *when the current official model was approved*.
    *   **Action**: Finds the current official (Production) model in the MLflow library.
    *   Loads the data that was likely used to train that model.
    *   Applies necessary data adjustments (feature transformations like creating 'AgeGroup').
    *   Saves two baseline files *to the record of that specific model in the MLflow library*:
        1.  `reference_data.parquet`: A sample of typical employee data (features) from that time. Used later to check if **new employee data looks different (feature drift)**.
        2.  `reference_predictions.csv`: A set of predictions made by *that specific official model* on a sample of data. Used later to check if **the model's prediction patterns have changed (prediction drift)**.
    *   **When to Run**: This script needs to be run once *after* a model is officially approved (promoted to Production in MLflow) to set the baseline for monitoring it.
*   **`scripts/batch_predict.py` (The Monthly Predictor)**:
    *   **Purpose**: To get predictions for all *current* employees using the official model.
    *   **Action**: Asks MLflow for the current official (Production) model.
    *   Fetches the latest data *only for employees who haven't left yet* from the database.
    *   Applies necessary data adjustments (feature transformations).
    *   Uses the official model to generate predictions for these employees.
    *   **Output**: Saves the data and predictions into files (`reports/batch_features.json`, `reports/batch_predictions.json`) for the monitoring step. It also saves predictions to a database table.
*   **`check_drift_via_api.py` (The Manual Drift Checker)**:
    *   A helper script to manually trigger the monitoring checks.
    *   It reads the files created by `batch_predict.py`.
    *   It sends this data to the "Monitoring Desk" (`drift-api`) to perform the comparisons against the baseline.
    *   Saves the results (drift detected? yes/no) reported by the API into files (`reports/feature_drift_results.json`, `reports/prediction_drift_results.json`).
*   **`src/employee_attrition_mlops/data_processing.py` (The Data Handler)**:
    *   Contains helper functions for getting data from the database, cleaning it, and making adjustments (like creating age groups).
    *   **Smart Connection**: Knows whether it's running on your local computer (macOS) or inside a Docker container (Linux) and uses the correct database driver (`pyodbc` locally, `pymssql` in Docker).
*   **`src/employee_attrition_mlops/drift_detection.py` (The Drift Calculator)**:
    *   Contains the statistical code (using the Evidently AI library) to actually compare current data/predictions to the baseline data/predictions and calculate if there's significant drift.

### 2.2. Always-On Services (Docker Containers)

These run continuously to provide the system's functionality:

*   **`mlflow-server` (The Library)**:
    *   Keeps track of all training experiments and model versions.
    *   Stores the baseline files associated with the official model.
*   **`api` (The Prediction Desk)**:
    *   Runs a web service (FastAPI).
    *   Loads the current official (Production) model when it starts.
    *   Provides a web address (`/predict`) where other applications can send data for a single employee and get an attrition prediction back immediately.
*   **`drift-api` (The Monitoring Desk)**:
    *   Runs a separate web service (FastAPI) focused only on checking for drift.
    *   Provides web addresses (`/drift/feature`, `/drift/prediction`) where current data can be sent for comparison against the baseline.
    *   It fetches the necessary baseline files from the MLflow library (`mlflow-server`) when it receives a request.
    *   Also provides addresses (`/drift/feature/latest`, `/drift/prediction/latest`) where the *results* of the latest drift check can be retrieved.
*   **`frontend` (The Dashboard)**:
    *   Runs the user-friendly web dashboard (Streamlit).
    *   Talks to the `api` service to get predictions for display.
    *   Talks to the `drift-api` service (using the `/latest` addresses) to get and display the most recent drift monitoring results.

### 2.3. Automation & Configuration

*   **`docker-compose.yml`**: The master plan for starting all the services together, making sure they can talk to each other.
*   **`.env`**: A configuration file holding sensitive information like database passwords and specific settings (like which database driver to use locally vs. in Docker).
*   **`.github/workflows/production_automation.yml` (The Automated Manager)**:
    *   Instructions for GitHub Actions, the tool that automates tasks.
    *   Typically set to run automatically on a schedule (e.g., monthly).
    *   Performs the routine check-up: runs batch predictions, triggers drift detection via the API, checks the results, and conditionally starts retraining if needed.

## 3. The Automated Monthly Workflow in Action

Here’s how the system works automatically once set up, typically running monthly:

1.  **Scheduled Start (e.g., 1st of the Month)**:
    *   The Automated Manager (GitHub Actions) wakes up based on its schedule.
    *   It ensures it has the latest code and instructions.
2.  **Run Monthly Predictions**: 
    *   The manager runs the `scripts/batch_predict.py` script.
    *   This script uses the *current official model* (from the MLflow Library) to predict attrition for all *current employees* (data from the database).
    *   It saves the employee data (`batch_features.json`) and the predictions (`batch_predictions.json`) into the shared `reports` folder.
3.  **Check for Changes (Drift Detection)**:
    *   The manager runs `check_drift_via_api.py` (or similar commands).
    *   This script sends the data saved in step 2 to the Monitoring Desk (`drift-api`).
    *   The `drift-api` fetches the *baseline* data (saved when the official model was approved) from the MLflow Library.
    *   It compares the *current month's* data/predictions to the *baseline* data/predictions.
    *   The results (did feature data change? did prediction patterns change?) are saved to files (`feature_drift_results.json`, `prediction_drift_results.json`) in the `reports` folder.
4.  **Review Results & Decide**: 
    *   The Automated Manager looks at the drift result files.
    *   It checks if the detected changes exceed acceptable limits (thresholds defined in configuration).
5.  **Retrain if Needed (Conditional)**:
    *   **IF** significant drift was detected:
        *   The manager runs the `scripts/optimize_train_select.py` script.
        *   This retrains the model from scratch using the latest complete dataset.
        *   A *new candidate model* is saved to the MLflow Library.
        *   **(Manual Step Usually Required)**: A human typically reviews this new candidate model in MLflow before approving it to become the *new* official (Production) model. If approved, `scripts/save_reference_data.py` must be run again to set a *new baseline*.
    *   **IF** no significant drift was detected:
        *   No retraining is triggered.
6.  **Cycle Complete**: The automated check-up is finished until the next scheduled run.

**Continuous Dashboard Updates**: While the monthly check happens, the `frontend` dashboard is always running. It regularly asks the `drift-api` for the latest results stored in the `reports` folder, ensuring the displayed monitoring status is always up-to-date.

## 4. Environment Handling (Local vs. Docker)

*   **Database Connection**: The system needs to connect to your SQL Server database.
    *   **On your Mac (Local)**: It uses a driver called `pyodbc`. You need to install the official Microsoft ODBC driver first (`brew install msodbcsql18`). The connection details are stored in the `.env` file under `DATABASE_URL_PYODBC`.
    *   **Inside Docker Containers (Linux)**: It uses a different driver called `pymssql`, which works better in that environment. The connection details are stored in the `.env` file under `DATABASE_URL_PYMSSQL`.
    *   **Automatic Switching**: The code in `data_processing.py` and `batch_predict.py` is smart enough to detect where it's running and use the correct setting from the `.env` file.
*   **MLflow**: The system needs to know where the MLflow Library (`mlflow-server`) is. This address is set in the `.env` file (`MLFLOW_TRACKING_URI`).

This guide provides a detailed overview. For the absolute specifics, always refer to the code comments and the configuration files themselves.
