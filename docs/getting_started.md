# Getting Started

This guide will help you set up and run the Employee Attrition MLOps project.

## Prerequisites

- Python 3.11
- Docker and Docker Compose
- Poetry (for dependency management)
- Git

## Setup

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd Employee-Attrition-2
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Run tests**
   ```bash
   poetry run pytest
   ```

5. **Run linting**
   ```bash
   poetry run black .
   poetry run isort .
   poetry run flake8 .
   poetry run mypy .
   ```

## Running the Project

### Using Docker Compose

1. **Start All Services**
   ```bash
   # Build and start all services
   docker-compose up --build
   
   # Or start in detached mode
   docker-compose up -d --build
   ```

2. **Access Services**
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - Drift API: http://localhost:8001
   - MLflow: http://localhost:5001

3. **Docker Services Overview**
   - **MLflow Server**: Model tracking and registry
   - **API**: Prediction service with FastAPI
   - **Drift API**: Drift detection service
   - **Frontend**: Streamlit interface

4. **Common Commands**
   ```bash
   # View running services
   docker-compose ps
   
   # View logs
   docker-compose logs -f
   
   # Restart services
   docker-compose restart
   
   # Stop all services
   docker-compose down
   
   # Rebuild specific service
   docker-compose build api
   ```

5. **Development Workflow**
   - Source code is mounted as volumes for live updates
   - Changes to Python files trigger automatic reload
   - MLflow artifacts persist between restarts
   - Reference data and predictions are stored in mounted volumes

### Running Locally

1. **Start MLflow server**
   ```bash
   poetry run mlflow server --host 127.0.0.1 --port 5001
   ```

2. **Start the API**
   ```bash
   poetry run uvicorn src.employee_attrition_mlops.api:app --reload
   ```

3. **Start the frontend**
   ```bash
   poetry run streamlit run src/frontend/app.py
   ```

## Project Structure

```
/
├── .github/workflows/    # GitHub Actions workflows
├── scripts/              # Automation and utility scripts
├── src/                  # Source code
│   ├── employee_attrition_mlops/  # Core ML logic
│   └── frontend/         # Streamlit app
├── tests/               # Test files
├── docs/                # Documentation
├── mlruns/              # MLflow tracking and artifacts
└── reports/             # Generated reports
```

## Key Components

### 1. API (FastAPI)
- Serves predictions and model info
- Endpoints:
  - `/predict`: Real-time predictions
  - `/model-info`: Model metadata
  - `/health`: System health check

### 2. Frontend (Streamlit)
- Interactive interface for predictions
- Real-time model info display
- User-friendly forms

### 3. MLflow
- Model tracking and versioning
- Experiment management
- Artifact storage

### 4. Automation
- All automation is managed by `.github/workflows/production_automation.yml`
- Includes:
  - Testing and linting
  - Drift detection
  - Model retraining
  - Batch prediction
  - Model promotion
  - API redeployment

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run tests and linting**
   ```bash
   poetry run pytest
   poetry run black .
   poetry run isort .
   poetry run flake8 .
   poetry run mypy .
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

5. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Documentation

- [CI/CD Workflow](ci_cd_workflow.md)
- [Monitoring](monitoring.md)

## Troubleshooting

### Common Issues

1. **MLflow Connection Issues**
   - See [MLflow Usage Guide](mlflow_usage.md) for configuration details
   - Check [Troubleshooting Guide](troubleshooting.md#mlflow-issues) for common solutions

2. **Database Connection Issues**
   - See [Setup Details](setup_details.md#database-setup) for configuration
   - Check [Troubleshooting Guide](troubleshooting.md#database-connection-issues) for solutions

3. **Docker Issues**
   - See [Setup Details](setup_details.md#docker-setup) for configuration
   - Check [Troubleshooting Guide](troubleshooting.md#docker-issues) for solutions

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [Setup Details](setup_details.md)
- Create an issue in the repository 