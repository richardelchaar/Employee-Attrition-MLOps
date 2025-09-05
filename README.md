# Employee Attrition MLOps Project

This project implements a full-stack, production-grade MLOps system to predict employee attrition. It automates the entire machine learning lifecycle—from data processing and model training to deployment, monitoring, and automated retraining—while incorporating principles of Responsible AI.

-----

## Key Features

  * **End-to-End Automation**: The MLOps workflow is fully automated using **GitHub Actions**, including model training with hyperparameter optimization (Optuna), validation, deployment, and scheduled monitoring.
  * **Advanced Drift Detection**: A robust system using **Evidently** continuously monitors for both feature drift and prediction drift. Results are logged to **MLflow** and visualized in a dedicated report.
  * **Interactive Frontend**: A **Streamlit** dashboard provides a user-friendly interface for stakeholders to get live predictions, view model performance metrics, and monitor drift status.
  * **Robust Backend & Database Integration**:
      * A high-performance **FastAPI** application serves model predictions and exposes endpoints for health checks and model information.
      * An **SQL Database** is fully integrated to act as the central source for training data, store all batch and real-time predictions, and power the workforce overview dashboard.
  * **Experiment Tracking & Governance**: **MLflow** is used for comprehensive experiment tracking, model versioning, artifact storage (e.g., fairness reports, SHAP plots), and managing the model registry (Staging/Production).
  * **Responsible AI**: Fairness assessment with **Fairlearn** and model explainability using **SHAP** are integrated into the training pipeline to ensure transparency and mitigate bias.
  * **Containerized Deployment**: The entire application stack (API, Frontend, MLflow) is containerized with **Docker** and orchestrated with **Docker Compose** for consistent, reproducible, and scalable deployments.

-----

## CI/CD Automation with GitHub Actions

The core of the project's automation is a **CI/CD pipeline** managed by GitHub Actions. This pipeline handles everything from code validation to the monthly production monitoring and retraining cycle.

```mermaid
flowchart LR
    Start([Start]) --> Trigger{Trigger Type}
    Trigger -->|Monthly Schedule| UnitTest[Unit Tests & Linting]
    Trigger -->|Manual Run| UnitTest
    
    UnitTest --> TestResult{Tests Pass?}
    TestResult -->|No| Fail([Fail])
    TestResult -->|Yes| Pipeline[Run MLOps Pipeline]
    
    Pipeline --> BatchPred[1. Batch Prediction]
    BatchPred --> DriftCheck[2. Drift Detection]
    
    DriftCheck --> DriftResult{3. Drift Detected?}
    DriftResult -->|No| CreateIssue[4. Create Summary Issue]
    DriftResult -->|Yes| Retrain[4. Retrain Model]
    
    Retrain --> CreateIssue
    CreateIssue --> End([End])
```

The workflow performs the following steps automatically:

1.  **Code Validation**: On every push or pull request, the pipeline runs unit tests and linting to ensure code quality.
2.  **Batch Prediction**: On a monthly schedule, it runs a batch prediction job on the latest employee data.
3.  **Drift Detection**: It compares the new data and predictions against a stored baseline to detect feature and prediction drift.
4.  **Automated Retraining**: If significant drift is detected, the workflow automatically triggers the training pipeline to create a new model candidate.
5.  **Reporting**: A GitHub issue is automatically created after each run to summarize the drift check results and any actions taken.

-----

## Technical Stack

  * **ML & Data Science**: Scikit-learn, Optuna, SHAP, Evidently, Fairlearn
  * **Backend & Frontend**: FastAPI, Streamlit
  * **MLOps & Tooling**: MLflow, Docker, Docker Compose, Poetry
  * **CI/CD**: GitHub Actions

-----

## Quickstart with Docker 🐳

This is the recommended method for running the project.

### Prerequisites

  * Docker & Docker Compose
  * Git

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/richardelchaar/Employee-Attrition-2.git
    cd Employee-Attrition-2
    ```

2.  **Configure Environment**
    Create a `.env` file from the example template and update it with your database credentials.

    ```bash
    cp .env.example .env
    # Edit the .env file with your configuration
    ```

3.  **Build and Run Services**
    This command will build the images and start all services in the background.

    ```bash
    docker-compose up --build -d
    ```

4.  **Access the Services**

      * **Frontend UI**: [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
      * **Prediction API Docs**: [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
      * **Drift API Docs**: [http://localhost:8001/docs](https://www.google.com/search?q=http://localhost:8001/docs)
      * **MLflow UI**: [http://localhost:5001](https://www.google.com/search?q=http://localhost:5001)

-----

## Project Structure

The repository is organized to separate concerns, making it clean and maintainable.

```
/
├── .github/workflows/  # CI/CD automation workflows
├── docs/               # All project documentation
├── scripts/            # Automation scripts (training, prediction, etc.)
├── src/                # Main source code
│   ├── employee_attrition_mlops/ # Core ML package (pipelines, API)
│   ├── frontend/         # Streamlit UI application
│   └── monitoring/       # Drift detection logic
├── tests/              # Test suite for the entire codebase
├── mlruns/             # MLflow experiment data (ignored by git)
├── reports/            # Generated drift and evaluation reports
├── docker-compose.yml  # Defines and orchestrates all services
└── pyproject.toml      # Project dependencies and metadata (Poetry)
```

-----

## Documentation

For more in-depth information, please refer to the comprehensive documentation.

  * **Core Concepts**
      * [System Architecture](docs/architecture.md)
      * [MLOps Workflow Guide](docs/mlops_workflow_guide.md)
      * [CI/CD Workflow](docs/ci_cd_workflow.md)
  * **Guides & Usage**
      * [Getting Started](docs/getting_started.md)
      * [Detailed Setup Guide](docs/setup_details.md)
      * [MLflow Usage](docs/mlflow_usage.md)
  * **Key Features**
      * [Monitoring & Governance Strategy](docs/monitoring.md)
      * [Drift Detection Guide](docs/drift_detection_guide.md)
      * [Responsible AI Guide](docs/responsible_ai.md)
  * **Reference**
      * [API Documentation](docs/api_documentation.md)
      * [Troubleshooting Guide](docs/troubleshooting.md)

-----

## License

This project is licensed under the MIT License.
