# Employee Attrition MLOps Project

A production-ready MLOps system for predicting employee attrition, incorporating robust data handling, automated training with integrated validation, deployment, monitoring, and clear governance.

## Project Structure

```
.
├── src/
│   ├── employee_attrition_mlops/
│   │   ├── api.py           # FastAPI backend for predictions and model info
│   │   ├── config.py        # Configuration management and environment variables
│   │   ├── data_processing.py # Data preprocessing and feature engineering
│   │   └── utils.py         # Utility functions for data handling
│   └── frontend/
│       └── app.py          # Streamlit frontend for user interaction
├── tests/
│   ├── test_api.py        # API endpoint tests (unit and integration)
│   ├── test_frontend.py   # Frontend logic tests (API integration)
│   └── test_data_processing.py # Data processing pipeline tests
├── scripts/
│   ├── optimize_train_select.py # Model training and optimization
│   ├── batch_predict.py    # Batch prediction processing
│   └── seed_database_from_csv.py # Database initialization
├── mlruns/                # MLflow tracking for experiments
├── mlartifacts/          # MLflow artifacts storage
├── models/               # Saved model artifacts
├── reports/             # Generated reports and visualizations
├── docs/                # Project documentation
├── Dockerfile           # Main application container
├── Dockerfile.mlflow    # MLflow tracking server container
├── docker-compose.yml   # Container orchestration
├── pyproject.toml       # Poetry dependencies
└── README.md
```

## Component Descriptions

### Backend Components

#### 1. FastAPI Backend (`src/employee_attrition_mlops/api.py`)
- **Purpose**: Provides RESTful API endpoints for predictions and model information
- **Key Features**:
  - `/predict` endpoint for real-time predictions
  - `/health` endpoint for system health monitoring
  - `/model-info` endpoint for model metadata
  - Database integration for prediction logging
  - MLflow model registry integration

#### 2. Configuration Management (`src/employee_attrition_mlops/config.py`)
- **Purpose**: Manages environment variables and configuration settings
- **Key Features**:
  - Database connection strings
  - MLflow tracking URI
  - API settings
  - Model registry settings

#### 3. Data Processing (`src/employee_attrition_mlops/data_processing.py`)
- **Purpose**: Handles data preprocessing and feature engineering
- **Key Features**:
  - Data cleaning and validation
  - Feature engineering
  - Data transformation pipelines
  - Data quality checks

#### 4. Utility Functions (`src/employee_attrition_mlops/utils.py`)
- **Purpose**: Provides helper functions for data handling and processing
- **Key Features**:
  - Data validation utilities
  - Logging functions
  - Error handling utilities

### Frontend Components

#### 1. Streamlit Frontend (`src/frontend/app.py`)
- **Purpose**: Provides a user-friendly interface for predictions
- **Key Features**:
  - Interactive form for prediction inputs
  - Real-time prediction display
  - Model information visualization
  - Error handling and user feedback

### Scripts

#### 1. Model Training (`scripts/optimize_train_select.py`)
- **Purpose**: Handles model training and optimization
- **Key Features**:
  - Automated model selection
  - Hyperparameter optimization
  - Cross-validation
  - Model evaluation and logging

#### 2. Batch Prediction (`scripts/batch_predict.py`)
- **Purpose**: Processes batch predictions
- **Key Features**:
  - Efficient batch processing
  - Error handling and logging
  - Results validation
  - Performance optimization

#### 3. Database Initialization (`scripts/seed_database_from_csv.py`)
- **Purpose**: Initializes the database with seed data
- **Key Features**:
  - Data validation
  - Database schema creation
  - Data seeding
  - Error handling

## Testing Strategy

### 1. Unit Tests
- **Purpose**: Test individual components in isolation
- **Coverage**:
  - API endpoint logic
  - Data processing functions
  - Utility functions
  - Configuration management

### 2. Integration Tests
- **Purpose**: Test component interactions
- **Coverage**:
  - API and database integration
  - Frontend and API integration
  - Data processing pipeline
  - Model training pipeline

### 3. End-to-End Tests
- **Purpose**: Test complete workflows
- **Coverage**:
  - Prediction workflow
  - Model training workflow
  - Batch prediction workflow

## Docker Setup

### 1. Main Application Container (`Dockerfile`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev
COPY . .
CMD ["poetry", "run", "uvicorn", "src.employee_attrition_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. MLflow Container (`Dockerfile.mlflow`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev
COPY . .
CMD ["poetry", "run", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

### 3. Docker Compose (`docker-compose.yml`)
```yaml
version: '3'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URL_PYMSSQL=${DATABASE_URL_PYMSSQL}
    depends_on:
      - mlflow

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
      - ./mlartifacts:/app/mlartifacts
```

## Streamlit Setup

### 1. Frontend Container
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev
COPY . .
CMD ["poetry", "run", "streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Docker Compose Addition
```yaml
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - api
      - mlflow
```

## Setup

1. **Install Dependencies**
   ```bash
   poetry install
   ```

2. **Environment Variables**
   Create a `.env` file with:
   ```
   MLFLOW_TRACKING_URI=your_mlflow_uri
   DATABASE_URL_PYMSSQL=your_database_url
   ```

3. **Run with Docker**
   ```bash
   docker-compose up -d
   ```

4. **Run Locally**
   ```bash
   # API
   poetry run uvicorn src.employee_attrition_mlops.api:app --reload
   
   # Frontend
   poetry run streamlit run src/frontend/app.py
   ```

## Testing

Run tests with:
```bash
# All tests
poetry run pytest

# Specific test file
poetry run pytest tests/test_api.py
poetry run pytest tests/test_frontend.py
poetry run pytest tests/test_data_processing.py
```

## Development

- **Code Style**: Uses black, isort, and flake8 for code formatting
- **Type Checking**: Uses mypy for static type checking
- **Testing**: Uses pytest for testing
- **Dependency Management**: Uses Poetry for dependency management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
