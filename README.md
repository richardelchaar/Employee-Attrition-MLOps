# Employee Attrition MLOps Project

A full-stack MLOps solution for employee attrition prediction, featuring:
- Automated model training, retraining, and promotion
- Drift detection and monitoring
- API and Streamlit frontend
- MLflow tracking and artifact management
- CI/CD with GitHub Actions and Docker Compose

## Architecture

- **API**: Serves predictions and model info (FastAPI)
- **Frontend**: Streamlit app for live predictions and model info
- **MLflow**: Model tracking and artifact storage
- **Automation**: All workflows managed by GitHub Actions (`production_automation.yml`)

## Quickstart

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

## Docker Services

| Service        | Purpose                | Port | Description                                    |
|----------------|------------------------|------|------------------------------------------------|
| api            | Backend API (FastAPI)  | 8000 | Serves predictions and model info              |
| frontend       | Frontend (Streamlit)   | 8501 | Interactive dashboard for predictions          |
| mlflow-server  | MLflow Tracking Server | 5001 | Model tracking, experiments, and artifacts     |

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

## CI/CD & Automation

All automation is managed by `.github/workflows/production_automation.yml`:
- Testing and linting
- Drift detection
- Model retraining
- Batch prediction
- Model promotion
- API redeployment

See [CI/CD & Automation](docs/ci_cd_automation.md) for details.

## Model Monitoring

- Drift detection runs on a schedule or on demand
- If drift is detected, the model is retrained and promoted automatically
- See [Monitoring & Retraining](docs/monitoring.md) for details

## API Redeployment

After model retraining/promotion:
1. New model is registered in MLflow
2. API container is restarted via Docker Compose
3. Frontend automatically picks up new model info

## Development

### Prerequisites
- Python 3.11
- Docker and Docker Compose
- Poetry (for dependency management)

### Setup
1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Run linting:
   ```bash
   poetry run black .
   poetry run isort .
   poetry run flake8 .
   poetry run mypy .
   ```

## Documentation

- [Getting Started](docs/getting_started.md)
- [CI/CD & Automation](docs/ci_cd_automation.md)
- [Monitoring & Retraining](docs/monitoring.md)

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a PR

## License

[Your License Here]

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

Replace the placeholder values with your actual configuration:
- `username`: Your database username
- `password`: Your database password
- `hostname`: Your database host
- `database`: Your database name
- `MLFLOW_TRACKING_URI`: Your MLflow server URL (default: http://localhost:5001)

### Running the Application

1. Start MLflow server (if not already running):
```bash
mlflow server --host 0.0.0.0 --port 5001
```

2. Start the FastAPI server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

3. Start the Streamlit app:
```bash
streamlit run src/frontend/app.py
```
