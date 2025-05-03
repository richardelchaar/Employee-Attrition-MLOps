# CI/CD Workflow Documentation

This document describes the CI/CD workflow for the Employee Attrition project.

## Workflow Overview

The CI/CD pipeline is designed to run monthly checks for model drift and automatically retrain the model when necessary. It consists of three main jobs: unit testing, pipeline execution, and Docker image building.

## Workflow Diagram

```mermaid
graph TD
    A[Trigger] --> B{Event Type}
    B -->|Schedule| C[Monthly Run]
    B -->|Manual| D[Workflow Dispatch]
    
    C --> E[Unit Tests]
    D --> E
    
    E --> F{Tests Pass?}
    F -->|Yes| G[Run Pipeline]
    F -->|No| H[Fail]
    
    G --> I[Batch Prediction]
    I --> J[Feature Drift Check]
    J --> K[Prediction Drift Check]
    
    K --> L{Drift Detected?}
    L -->|Yes| M[Retrain Model]
    L -->|No| N[Skip Retraining]
    
    M --> O[Create Summary Issue]
    N --> O
    
    L -->|Yes| P[Docker Build & Push]
    P --> Q[MLflow Image]
    P --> R[API Image]
    P --> S[Frontend Image]
    P --> T[Drift API Image]
    
    subgraph "Unit Test Job"
        E --> E1[Install Dependencies]
        E1 --> E2[Run Tests]
        E2 --> E3[Run Linting]
    end
    
    subgraph "Pipeline Job"
        G --> G1[Setup Environment]
        G1 --> G2[Docker Compose]
        G2 --> G3[Batch Prediction]
        G3 --> G4[Drift Detection]
        G4 --> G5[Model Retraining]
        G5 --> G6[Create Issue]
    end
    
    subgraph "Docker Build Job"
        P --> P1[Setup Docker]
        P1 --> P2[Login to DockerHub]
        P2 --> P3[Build & Push Images]
    end
```

## Workflow Components

### 1. Triggers
- **Monthly Schedule**: Runs at midnight on the first day of every month
- **Manual Trigger**: Can be triggered manually through GitHub Actions

### 2. Unit Test Job
- Installs project dependencies
- Runs pytest for unit tests
- Performs code linting (black, isort, flake8, mypy)

### 3. Pipeline Job
- Sets up the environment with necessary secrets
- Runs Docker Compose for service orchestration
- Executes batch prediction
- Checks for feature and prediction drift
- Retrains model if drift is detected
- Creates a summary issue with results

### 4. Docker Build Job (Conditional)
- Only runs if drift is detected
- Builds and pushes four Docker images:
  - MLflow server
  - API service
  - Frontend application
  - Drift detection API

## Output and Monitoring

The workflow creates a GitHub issue after each run with:
- Feature drift results
- Prediction drift results
- Retraining status
- Batch prediction summary
- Link to the workflow run

## Security

The workflow uses GitHub secrets for sensitive information:
- MLflow tracking URI
- Database connection strings
- DockerHub credentials
- Drift API URL 