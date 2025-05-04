# Employee Attrition MLOps Project

## Project Overview

This project transforms a basic employee attrition prediction model into a full-fledged MLOps system, implementing industry best practices for machine learning operations. The system focuses on predicting employee attrition - a critical HR analytics problem that helps organizations identify employees at risk of leaving and take proactive measures to improve retention.

### Core Objectives

1. **Production-Ready ML System**
   - Robust data pipeline for consistent data processing
   - Automated model training and validation workflows
   - Reliable model deployment and serving infrastructure
   - Comprehensive monitoring and drift detection
   - Continuous Integration/Continuous Deployment (CI/CD)

2. **Responsible AI Implementation**
   - Fairness assessment and bias mitigation
   - Model explainability through SHAP values
   - Transparent decision-making process
   - Ethical considerations in predictions

3. **End-to-End MLOps Pipeline**
   - Data versioning and lineage tracking
   - Experiment tracking with MLflow
   - Automated model retraining triggers
   - Performance monitoring and alerting
   - Model versioning and promotion

### Key Features

- **Data Management**
  - Automated data preprocessing pipeline
  - Feature engineering and validation
  - Data quality monitoring
  - Reference data management

- **Model Development**
  - Automated model training pipeline
  - Hyperparameter optimization
  - Model evaluation and selection
  - Cross-validation and testing

- **Monitoring & Maintenance**
  - Real-time drift detection
  - Performance metric tracking
  - Automated alert system
  - Model health monitoring

- **Deployment & Serving**
  - FastAPI-based prediction service
  - Streamlit frontend for predictions
  - Model version management
  - A/B testing capability

- **Responsible AI**
  - Fairness metrics calculation
  - Bias detection and mitigation
  - SHAP-based feature importance
  - Prediction explanations

### Technical Stack

- **Backend**: Python 3.11+, FastAPI
- **ML Libraries**: scikit-learn, optuna, shap, evidently
- **Monitoring**: MLflow, custom drift detection
- **Frontend**: Streamlit
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

### Use Case: Employee Attrition Prediction

The system predicts the likelihood of employee attrition by analyzing various factors such as:
- Employee demographics
- Job characteristics
- Work environment metrics
- Performance indicators
- Compensation and benefits
- Career development opportunities

This prediction helps organizations:
- Identify at-risk employees
- Understand key factors driving attrition
- Develop targeted retention strategies
- Optimize HR policies and practices
- Improve employee satisfaction and engagement

A full-stack MLOps solution for employee attrition prediction with robust drift detection capabilities, featuring:
- Automated model training, retraining, and promotion
- Drift detection and monitoring
- API and Streamlit frontend
- MLflow tracking and artifact management
- CI/CD with GitHub Actions and Docker Compose

## Features

- **ML Model Training**: Automated training and validation of employee attrition prediction models
- **MLflow Integration**: Tracking experiments, model registration, and model versioning
- **Drift Detection System**: 
  - Feature drift monitoring with statistical tests
  - Prediction drift monitoring for model outputs
  - Automated alerts when drift is detected
  - Detailed HTML reports with feature-by-feature analysis
  - FastAPI endpoint for on-demand drift detection
  - MLflow integration for tracking drift metrics over time
  - Customizable drift thresholds for different sensitivity levels
- **GitHub Actions Workflows**:
  - Automated drift detection on schedule
  - Model promotion workflow
  - MLflow metadata maintenance
- **Visualization**: Comprehensive HTML reports for data and prediction drift
- **Frontend**: Streamlit app for live predictions and model info

## Architecture

- **API**: Serves predictions and model info (FastAPI)
- **Frontend**: Streamlit app for live predictions and model info
- **MLflow**: Model tracking and artifact storage
- **Automation**: All workflows managed by GitHub Actions

### Drift Detection Architecture

The drift detection system consists of:

1. **Reference Data Management**:
   - Saving baseline data for comparison
   - Storing feature distributions and statistics

2. **Drift Detection Pipeline**:
   - Feature drift detection using statistical tests
   - Prediction drift monitoring
   - HTML report generation

3. **Automation**:
   - GitHub Actions workflows for scheduled monitoring
   - Automatic issue creation for detected drift
   - Model retraining triggers

4. **API Layer**:
   - FastAPI endpoints for drift detection
   - Report generation and retrieval

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/BTCJULIAN/Employee-Attrition-2.git
cd Employee-Attrition-2

# Install dependencies with Poetry
poetry install
```

### Running Drift Detection

```bash
# Run drift detection with default settings
python check_production_drift.py

# Generate HTML report for current data
python scripts/generate_drift_report.py --current-data path/to/data.csv

# Save new reference data (baseline) for drift comparison
python save_reference_data.py
```

### Using the Drift Detection API

The project includes a FastAPI endpoint for drift detection:

```bash
# Start the API server
python drift_api.py

# Test the API with the test client
python test_drift_api_client.py

# Access the API documentation at http://localhost:8000/docs
```

Example API request:
```python
import requests
import pandas as pd

# Load data
df = pd.read_csv("path/to/data.csv")
data = df.to_dict(orient="records")

# Detect drift
response = requests.post(
    "http://localhost:8000/detect-drift",
    json={"data": data, "threshold": 0.05}
)

# Check results
result = response.json()
print(f"Drift detected: {result['drift_detected']}")
```

For comprehensive documentation on drift detection, see the [Drift Detection Guide](docs/drift_detection_guide.md).

### MLflow Maintenance

Repair and maintain MLflow metadata with:

```bash
python scripts/mlflow_maintenance.py --fix-run-metadata
```

## Quickstart with Docker

1. Clone the repo and set up `.env`:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Build and run all services:
   ```bash
   docker-compose up --build
   ```

3. Access:
   - API: http://localhost:8000
   - Frontend: http://localhost:8501
   - MLflow: http://localhost:5001

## Project Structure

/
├── .github/workflows/ # GitHub Actions workflows
├── scripts/ # Automation and utility scripts
├── src/ # Source code
│ ├── employee_attrition_mlops/ # Core ML logic
│ ├── monitoring/ # Drift detection
│ └── frontend/ # Streamlit app
├── tests/ # Test files
├── docs/ # Documentation
├── mlruns/ # MLflow tracking and artifacts
└── reports/ # Generated reports

## Repository Structure

This section provides a detailed explanation of the repository's organization and the purpose of each directory and key files.

### Core Directories

#### `.github/workflows/`
Contains GitHub Actions workflows for CI/CD automation:
- `production_automation.yml`: Main workflow for production deployment
- `drift_detection.yml`: Scheduled drift monitoring
- `model_promotion.yml`: Model version promotion pipeline
- `testing.yml`: Automated testing workflow

#### `docs/`
Project documentation and guides:
- Architecture diagrams and system design
- Setup and installation guides
- API documentation
- User manuals
- Development guidelines

#### `mlartifacts/` and `mlruns/`
MLflow tracking directories (typically in `.gitignore`):
- `mlruns/`: Experiment tracking data
- `mlartifacts/`: Model artifacts and metadata
- Contains run histories, metrics, and model versions

#### `models/`
Local model storage (outside MLflow registry):
- Saved model files
- Model checkpoints
- Pre-trained models
- Model configurations

#### `reports/`
Generated analysis and monitoring reports:
- Drift detection reports
- Model performance metrics
- Fairness analysis reports
- Confusion matrices
- Feature importance visualizations

#### `scripts/`
Standalone Python scripts for various tasks:
- `optimize_train_select.py`: Model training and selection
- `batch_predict.py`: Batch prediction processing
- `create_drift_reference.py`: Reference data generation
- `mlflow_maintenance.py`: MLflow metadata management
- `generate_reports.py`: Report generation utilities

#### `src/employee_attrition_mlops/`
Core Python package containing reusable code:
- `config.py`: Configuration management
- `data_processing.py`: Data preprocessing and feature engineering
- `pipelines.py`: ML pipeline implementation
- `utils.py`: Utility functions and helpers
- `api.py`: FastAPI endpoints

#### `src/frontend/`
Streamlit-based user interface:
- `app.py`: Main Streamlit application
- UI components and layouts
- Visualization utilities
- User interaction handlers

#### `src/monitoring/`
Monitoring and drift detection logic:
- Drift detection algorithms
- Alert generation
- Performance monitoring
- Statistical testing

#### `tests/`
Test suite for the codebase:
- Unit tests for individual components
- Integration tests for system workflows
- Test fixtures and utilities
- Performance benchmarks

### Key Configuration Files

#### `pyproject.toml`
Project configuration and dependency management:
- Package metadata
- Dependencies and versions
- Development tools configuration
- Build settings

#### `poetry.lock`
Lock file for Poetry dependency management:
- Exact dependency versions
- Hash verification
- Dependency resolution

#### `Dockerfile` and `Dockerfile.frontend`
Containerization configurations:
- Base image setup
- Dependency installation
- Application configuration
- Environment setup

#### `docker-compose.yml`
Multi-container application orchestration:
- Service definitions
- Network configuration
- Volume mappings
- Environment variables

#### `.env` (Template: `.env.example`)
Environment configuration (should be gitignored):
- Database credentials
- API keys and secrets
- Service endpoints
- Feature flags

## CI/CD & Automation

All automation is managed by GitHub Actions workflows:
- Testing and linting
- Drift detection
- Model retraining
- Batch prediction
- Model promotion
- API redeployment

## Environment Setup

### Required Environment Variables
Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DATABASE_URL_PYMSSQL=mssql+pymssql://username:password@hostname:1433/database

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5001  # MLflow server
MLFLOW_MODEL_NAME=employee_attrition_model
MLFLOW_MODEL_STAGE=Production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## Setup Instructions (macOS)

This guide provides detailed setup instructions for macOS, including solutions for common challenges.

### Prerequisites

Before starting, ensure you have the following installed:

1. **Git**
   ```bash
   # Verify Git installation
   git --version
   ```

2. **Python 3.11**
   ```bash
   # Install Python 3.11 using Homebrew
   brew install python@3.11
   
   # Verify Python version
   python3.11 --version
   ```

3. **Homebrew**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

### Common Challenges & Solutions

1. **ODBC Driver Installation**
   - **Challenge**: Microsoft ODBC driver installation can be tricky on macOS
   - **Solution**:
     ```bash
     # Install unixodbc first
     brew install unixodbc
     
     # Add Microsoft tap
     brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
     
     # Install driver
     brew install msodbcsql17 mssql-tools
     
     # Verify installation
     odbcinst -q -d
     ```
   - **Troubleshooting**: See [ODBC Troubleshooting Guide](docs/troubleshooting/odbc.md)

2. **Pydantic v2 Dependency Conflicts**
   - **Challenge**: Some packages may require specific Pydantic versions
   - **Solution**:
     ```bash
     # Check current Pydantic version
     poetry show pydantic
     
     # Update to compatible version
     poetry add "pydantic>=2.0.0,<3.0.0"
     
     # Resolve dependencies
     poetry install
     ```
   - **Troubleshooting**: See [Dependency Management Guide](docs/troubleshooting/dependencies.md)

3. **Python Version Management**
   - **Challenge**: Multiple Python versions can cause conflicts
   - **Solution**:
     ```bash
     # Use pyenv for version management
     brew install pyenv
     
     # Install Python 3.11
     pyenv install 3.11.0
     
     # Set local version
     pyenv local 3.11.0
     ```
   - **Troubleshooting**: See [Python Environment Guide](docs/troubleshooting/python.md)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/BTCJULIAN/Employee-Attrition-2.git
   cd Employee-Attrition-2
   ```

2. **Install Poetry**
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3.11 -
   
   # Add Poetry to your PATH (add to ~/.zshrc or ~/.bash_profile)
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **Install ODBC Drivers**
   ```bash
   # Install unixodbc
   brew install unixodbc
   
   # Install Microsoft ODBC Driver for SQL Server
   brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
   brew update
   brew install msodbcsql17 mssql-tools
   ```

4. **Install Project Dependencies**
   ```bash
   # Install all dependencies including development tools
   poetry install --with dev
   ```

5. **Configure Environment Variables**

   Create a `.env` file in the project root directory:
   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with your configuration:
   ```env
   # Database Configuration
   DATABASE_URL_PYMSSQL=mssql+pymssql://username:password@hostname:1433/database
   
   # MLflow Configuration
   MLFLOW_TRACKING_URI=http://localhost:5001
   MLFLOW_MODEL_NAME=employee_attrition_model
   MLFLOW_MODEL_STAGE=Production
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

   Note: The `.env` file is included in `.gitignore` for security reasons. Never commit sensitive information.

6. **Start MLflow Server**

   In a new terminal window, run:
   ```bash
   # Start MLflow UI server
   poetry run mlflow ui --host 0.0.0.0 --port 5001
   ```

   Access the MLflow UI at: http://localhost:5001

### Verifying the Setup

1. **Check Poetry Environment**
   ```bash
   poetry env info
   ```

2. **Test Database Connection**
   ```bash
   poetry run python -c "from src.employee_attrition_mlops.config import get_db_connection; get_db_connection()"
   ```

3. **Verify MLflow Connection**
   ```bash
   poetry run python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

### Troubleshooting

1. **ODBC Driver Issues**
   - Verify ODBC driver installation:
     ```bash
     odbcinst -q -d
     ```
   - Check ODBC configuration:
     ```bash
     odbcinst -q -s
     ```

2. **Python Version Issues**
   - Ensure correct Python version:
     ```bash
     poetry env use python3.11
     ```

3. **Dependency Installation Issues**
   - Clear Poetry cache and retry:
     ```bash
     poetry cache clear . --all
     poetry install --with dev
     ```

### Next Steps

After completing the setup:
1. Run the test suite: `poetry run pytest`
2. Start the API server: `poetry run python src/employee_attrition_mlops/api.py`
3. Launch the frontend: `poetry run streamlit run src/frontend/app.py`

## Workflow Overview

This section describes the end-to-end MLOps workflow implemented in this project, highlighting key MLOps principles and best practices for production-grade machine learning systems.

### MLOps Principles Implemented

1. **Comprehensive Logging & Governance**
   - MLflow tracking for experiment reproducibility
   - Detailed model metadata and lineage tracking
   - Version control for data, code, and models
   - Audit trails for model changes and deployments

2. **Automated Testing & Validation**
   - Unit tests for individual components
   - Integration tests for pipeline workflows
   - Data validation at each processing stage
   - Model performance validation

3. **Monitoring & Baselining**
   - Reference data generation for drift detection
   - Statistical baselines for feature distributions
   - Performance metric tracking over time
   - Automated alerting for anomalies

4. **Continuous Integration/Deployment**
   - Automated testing on code changes
   - Model versioning and promotion
   - Containerized deployment
   - Environment consistency

5. **Responsible AI Implementation**
   - Fairness assessment and monitoring
   - Model explainability tracking
   - Bias detection and mitigation
   - Ethical considerations in predictions

### 1. Data Handling

The data pipeline begins with loading and processing employee data:

```mermaid
graph LR
    A[Database] --> B[Data Loading]
    B --> C[Data Cleaning]
    C --> D[Feature Engineering]
    D --> E[Processed Data]
```

- **Initial Data Setup**: 
  - Use `scripts/seed_database_from_csv.py` to populate the database with initial data
  - Supports both CSV and direct database connections

- **Data Processing Pipeline**:
  - Data loaded via `src/employee_attrition_mlops/data_processing.py`
  - Automated cleaning and preprocessing
  - Feature engineering and validation
  - Train/test/validation splits

### 2. Training & Validation Pipeline

The model development pipeline is orchestrated by `scripts/optimize_train_select.py`:

```mermaid
graph TD
    A[Data] --> B[HPO with Optuna]
    B --> C[Model Selection]
    C --> D[Final Training]
    D --> E[Validation]
    E --> F[MLflow Logging]
```

- **Hyperparameter Optimization**:
  - Uses Optuna for efficient HPO
  - Cross-validation for robust evaluation
  - Multiple model architectures considered

- **Model Selection & Training**:
  - Best model selected based on validation metrics
  - Final training on full training set
  - Comprehensive validation suite

- **MLflow Artifacts**:
  - Performance metrics and plots
  - Fairness analysis reports
  - SHAP explanations
  - Drift detection baselines
  - Model registered to 'Staging' stage

### 3. CI/CD Pipeline

Automated workflows managed by GitHub Actions:

```mermaid
graph LR
    A[PR Created] --> B[Linting/Testing]
    B --> C[Training Pipeline]
    C --> D[Model Promotion]
    D --> E[API Deployment]
```

- **Pull Request Workflow**:
  - Code linting and formatting
  - Unit and integration tests
  - Documentation validation

- **Production Automation**:
  - Triggered on main branch updates
  - Runs full training pipeline
  - Builds and deploys API service
  - Updates model registry

### 4. Model Deployment

The trained model is served through a FastAPI service:

```mermaid
graph LR
    A[Model Registry] --> B[API Service]
    B --> C[Batch Prediction]
    B --> D[Real-time Prediction]
```

- **API Service**:
  - FastAPI implementation in `src/employee_attrition_mlops/api.py`
  - Docker containerization
  - Health checks and monitoring
  - Swagger documentation

- **Prediction Modes**:
  - Real-time predictions via API
  - Batch predictions using `scripts/batch_predict.py`
  - Support for both single and bulk requests

### 5. Monitoring & Retraining

Continuous monitoring and automated retraining:

```mermaid
graph LR
    A[Production Data] --> B[Drift Detection]
    B --> C{Drift Detected?}
    C -->|Yes| D[Retrain Trigger]
    C -->|No| E[Continue Monitoring]
    D --> F[Training Pipeline]
```

- **Drift Detection**:
  - Reference data generation via `scripts/create_drift_reference.py`
  - Statistical tests for feature drift
  - Prediction drift monitoring
  - Automated alert system

- **Retraining Triggers**:
  - Significant drift detection
  - Scheduled retraining
  - Performance degradation
  - Manual override capability

### 6. Responsible AI

Ethical considerations and model transparency:

```mermaid
graph TD
    A[Model] --> B[Fairness Analysis]
    A --> C[Explainability]
    A --> D[Bias Detection]
    B --> E[Reports]
    C --> E
    D --> E
```

- **Fairness Assessment**:
  - Multiple fairness metrics
  - Protected attribute analysis
  - Bias mitigation strategies

- **Explainability**:
  - SHAP value generation
  - Feature importance analysis
  - Prediction explanations
  - Decision boundary visualization

### Workflow Integration

All components are integrated through:

- **MLflow Tracking**:
  - Experiment management
  - Model versioning
  - Artifact storage
  - Metric tracking

- **Docker Compose**:
  - Service orchestration
  - Environment consistency
  - Easy deployment
  - Scalability

- **GitHub Actions**:
  - Automated workflows
  - Environment management
  - Deployment coordination
  - Monitoring integration

## Design Choices & Rationale

This section explains the key technology choices made for this MLOps pipeline and their benefits.

### MLflow: Experiment Tracking & Model Registry

**Why MLflow?**
- **Comprehensive Tracking**: MLflow provides a unified platform for tracking experiments, parameters, metrics, and artifacts
- **Model Registry**: Built-in model versioning and stage management (Staging/Production)
- **Reproducibility**: Detailed logging of environment, code versions, and dependencies
- **Integration**: Seamless integration with popular ML frameworks and cloud providers

**Key Benefits**:
- Centralized experiment management
- Easy model versioning and promotion
- Detailed model lineage tracking
- Built-in UI for experiment visualization

### Poetry: Dependency Management

**Why Poetry?**
- **Deterministic Builds**: Lock file ensures consistent dependency versions
- **Virtual Environment Management**: Automatic environment creation and activation
- **Dependency Resolution**: Efficient resolution of complex dependency trees
- **Development Workflow**: Built-in commands for building, publishing, and testing

**Key Benefits**:
- Consistent development environments
- Simplified dependency management
- Better security through version pinning
- Streamlined development workflow

### Fairlearn & SHAP: Responsible AI

**Why Fairlearn?**
- **Comprehensive Fairness Metrics**: Multiple fairness definitions and metrics
- **Bias Mitigation**: Built-in algorithms for bias reduction
- **Protected Attributes**: Support for analyzing multiple protected groups
- **Integration**: Works well with scikit-learn pipelines

**Why SHAP?**
- **Model-Agnostic**: Works with any ML model
- **Local & Global Explanations**: Individual and overall feature importance
- **Visualization**: Rich visualization capabilities
- **Trust**: Widely accepted in industry and research

**Key Benefits**:
- Ethical model development
- Transparent decision-making
- Regulatory compliance
- Stakeholder trust

### GitHub Actions: CI/CD

**Why GitHub Actions?**
- **Native Integration**: Tight integration with GitHub repositories
- **Flexible Workflows**: Customizable pipeline definitions
- **Matrix Testing**: Parallel testing across environments
- **Artifact Management**: Built-in artifact storage and sharing

**Key Benefits**:
- Automated testing and deployment
- Consistent build environments
- Easy integration with other tools
- Cost-effective for open-source projects

### FastAPI: API Framework

**Why FastAPI?**
- **Performance**: High-performance async framework
- **Type Safety**: Built-in data validation and type checking
- **Documentation**: Automatic OpenAPI/Swagger documentation
- **Modern Features**: Async support, dependency injection

**Key Benefits**:
- Fast and scalable API service
- Self-documenting endpoints
- Easy integration with ML models
- Strong type safety

### Docker: Containerization

**Why Docker?**
- **Isolation**: Consistent runtime environments
- **Portability**: Run anywhere with Docker installed
- **Scalability**: Easy horizontal scaling
- **Versioning**: Container version control

**Key Benefits**:
- Environment consistency
- Simplified deployment
- Easy scaling
- Version control for deployments

## Running Key Tasks

This section provides quick reference commands for common development and deployment tasks.

### Development Environment

1. **Activate Poetry Environment**
   ```bash
   poetry shell
   ```

2. **Install Dependencies**
   ```bash
   # Install all dependencies
   poetry install

   # Install with development tools
   poetry install --with dev
   ```

### Model Development

1. **Run Training Pipeline**
   ```bash
   # Run full training pipeline with HPO
   poetry run python scripts/optimize_train_select.py

   # Run with specific configuration
   poetry run python scripts/optimize_train_select.py --config config/training_config.yaml
   ```

2. **Generate Drift Reference Data**
   ```bash
   # Create new reference dataset
   poetry run python scripts/create_drift_reference.py

   # Specify custom reference data path
   poetry run python scripts/create_drift_reference.py --output data/reference/new_reference.csv
   ```

3. **Run Batch Predictions**
   ```bash
   # Process batch predictions
   poetry run python scripts/batch_predict.py --input data/predictions/input.csv --output data/predictions/results.csv
   ```

### Testing & Quality Assurance

1. **Run Test Suite**
   ```bash
   # Run all tests
   poetry run pytest

   # Run specific test file
   poetry run pytest tests/test_model.py

   # Run with coverage report
   poetry run pytest --cov=src --cov-report=html
   ```

2. **Code Quality Checks**
   ```bash
   # Run linter
   poetry run ruff check .

   # Run type checker
   poetry run mypy src/

   # Format code
   poetry run black .
   ```

### Monitoring & Experimentation

1. **Start MLflow UI**
   ```bash
   # Start MLflow UI server
   poetry run mlflow ui --host 0.0.0.0 --port 5001

   # Access at: http://localhost:5001
   ```

2. **Run Drift Detection**
   ```bash
   # Check for data drift
   poetry run python scripts/check_production_drift.py

   # Generate drift report
   poetry run python scripts/generate_drift_report.py --current-data data/current.csv
   ```

### Docker Deployment

1. **Build Docker Images**
   ```bash
   # Build API service
   docker build -t employee-attrition-api -f Dockerfile .

   # Build frontend
   docker build -t employee-attrition-frontend -f Dockerfile.frontend .
   ```

2. **Run with Docker Compose**
   ```bash
   # Start all services
   docker-compose up --build

   # Start in detached mode
   docker-compose up -d

   # View logs
   docker-compose logs -f

   # Stop services
   docker-compose down
   ```

3. **Access Services**
   - API: http://localhost:8000
   - Frontend: http://localhost:8501
   - MLflow: http://localhost:5001
   - API Documentation: http://localhost:8000/docs

### Database Operations

1. **Seed Database**
   ```bash
   # Seed from CSV
   poetry run python scripts/seed_database_from_csv.py --input data/raw/employees.csv

   # Verify database connection
   poetry run python -c "from src.employee_attrition_mlops.config import get_db_connection; get_db_connection()"
   ```

### CI/CD Operations

1. **Run GitHub Actions Locally**
   ```bash
   # Install act
   brew install act

   # Run specific workflow
   act -W .github/workflows/production_automation.yml
   ```

### Troubleshooting

1. **Common Issues**
   ```bash
   # Clear Poetry cache
   poetry cache clear . --all

   # Rebuild Docker images
   docker-compose build --no-cache

   # Check service logs
   docker-compose logs -f [service_name]
   ```

2. **Environment Verification**
   ```bash
   # Check Python version
   poetry run python --version

   # Verify dependencies
   poetry show --tree

   # Check MLflow connection
   poetry run python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

Note: All commands assume you're in the project root directory and have Poetry installed. For Docker commands, ensure Docker and Docker Compose are installed and running.

## License

MIT