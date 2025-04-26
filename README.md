# Employee Attrition Prediction System

A machine learning system for predicting employee attrition using MLOps best practices.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Processing](#data-processing)
- [Model Training and Optimization](#model-training-and-optimization)
- [API and Deployment](#api-and-deployment)
- [Web Interface](#web-interface)
- [Testing](#testing)
- [Docker Support](#docker-support)
- [Development Tools](#development-tools)

## Overview

This project implements an end-to-end machine learning pipeline for predicting employee attrition. It follows MLOps best practices, including:

- Automated data processing and feature engineering
- Comprehensive testing (unit, integration, and performance)
- API deployment
- Docker containerization for reproducibility
- MLflow integration for experiment tracking
- Batch prediction capabilities
- Database integration and management
- Interactive web interface for predictions

## Project Structure

```
employee-attrition-mlops/
├── src/                   # Source code
│   ├── app.py             # Streamlit web application
│   └── employee_attrition_mlops/
│       ├── data_processing.py  # Data processing and transformers
│       ├── config.py           # Configuration settings
│       ├── api.py             # API endpoints
│       ├── pipelines.py       # ML pipelines
│       ├── utils.py           # Utility functions
│       └── __init__.py        # Package initialization
├── tests/                 # Test suite
│   ├── test_data_processing.py  # Data processing tests
│   ├── test_batch_predict.py    # Batch prediction tests
│   └── test_utils.py           # Utility function tests
├── scripts/               # Utility scripts
│   ├── batch_predict.py        # Batch prediction script
│   ├── seed_database_from_csv.py # Database seeding utility
│   ├── optimize_train_select.py  # Model optimization and training
│   └── __init__.py             # Package initialization
├── mlruns/               # MLflow experiment tracking
├── reports/              # Generated reports and visualizations
├── models/               # Trained models and artifacts
├── notebooks/            # Jupyter notebooks for analysis
├── docs/                 # Project documentation
├── references/           # Reference materials and datasets
├── mlartifacts/          # ML artifacts and model versions
├── pyproject.toml        # Project dependencies
├── poetry.lock          # Locked dependencies
├── pytest.ini           # pytest configuration
├── Dockerfile           # Main Docker configuration
├── Dockerfile.mlflow    # MLflow Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── Makefile             # Build automation
├── .gitignore          # Git ignore rules
└── LICENSE             # Project license
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip
- Git
- Docker (optional)
- ODBC driver for database connectivity

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/employee-attrition-mlops.git
   cd employee-attrition-mlops
   ```

2. Install dependencies:
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Database Setup

1. Set up your database (SQL Server, PostgreSQL, etc.)
2. Configure the connection string in your `.env` file:
   ```
   DATABASE_URL_PYODBC=mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
   ```
3. Install the appropriate ODBC driver:
   - On macOS: `brew install unixodbc`
   - On Windows: Install the appropriate ODBC driver for your database
   - On Linux: `apt-get install unixodbc-dev` or equivalent

4. Seed the database with initial data:
   ```bash
   python scripts/seed_database_from_csv.py
   ```

## Data Processing

The data processing pipeline includes:

### Custom Transformers
- `BoxCoxSkewedTransformer`: Handles skewed numerical columns
- `AddNewFeaturesTransformer`: Creates derived features (AgeAtJoining, TenureRatio, etc.)
- `AgeGroupTransformer`: Categorizes age into groups
- `CustomOrdinalEncoder`: Handles categorical variables
- `LogTransformSkewed`: Alternative transformation for skewed data

### Data Loading and Cleaning
- Database connectivity with error handling
- Data validation and cleaning
- Feature engineering
- Missing value handling
- Outlier detection

### Pipeline Configuration
- Configurable preprocessing steps
- Support for different encoding strategies
- Flexible scaling options

## Model Training and Optimization

The project includes comprehensive model training and optimization capabilities:

### Training Pipeline
- Automated model selection
- Hyperparameter optimization
- Cross-validation
- Model performance tracking with MLflow

### Optimization Script
The `optimize_train_select.py` script provides:
- Automated model selection
- Hyperparameter tuning
- Cross-validation
- Performance metrics tracking
- Model persistence

### Batch Prediction
The `batch_predict.py` script enables:
- Bulk prediction processing
- Database integration
- Result storage and reporting

## API and Deployment

The API module (`api.py`) provides:
- REST endpoints for model inference
- Data validation
- Error handling
- Response formatting

### Deployment Options
1. Local deployment
2. Docker container deployment
3. Cloud platform deployment (AWS, GCP, Azure)

## Web Interface

The project includes a Streamlit web application (`src/app.py`) that provides:
- Interactive user interface for making predictions
- Real-time model inference
- Visualization of prediction results
- Model information and performance metrics
- User-friendly forms for data input

### Running the Web Interface

```bash
# Start the Streamlit app
streamlit run src/app.py
```

## Testing

The project includes comprehensive tests:
- Unit tests for transformers and utilities
- Integration tests for database connectivity
- Batch prediction tests
- Performance tests
- Edge case handling

### Running Tests

```bash
# Run all tests except integration tests
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Run all tests
pytest
```

### Test Files
- `test_data_processing.py`: Tests for data processing pipeline and transformers
- `test_batch_predict.py`: Tests for batch prediction functionality
- `test_utils.py`: Tests for utility functions

### Database Connection Requirements for Tests

Integration tests require:
1. A `.env` file with `DATABASE_URL_PYODBC` variable
2. The `pyodbc` package installed
3. ODBC driver installed (see Setup and Installation section)

## Docker Support

The project includes Docker support with two configurations:
- Main application container
- MLflow tracking server container

### Building and Running with Docker

```bash
# Build and start all services
docker-compose up --build

# Or individual services
docker-compose up api
docker-compose up mlflow
```

## Development Tools

### Makefile Commands
The project includes a comprehensive Makefile with commands for:
- Building Docker images
- Running tests
- Cleaning build artifacts
- Database management
- Development setup

### MLflow Integration
- Experiment tracking
- Model versioning
- Performance metrics visualization
- Parameter tracking

### Documentation
- API documentation
- Development guides
- Deployment instructions
- Testing guidelines

## License

This project is licensed under the terms of the license included in the repository.
