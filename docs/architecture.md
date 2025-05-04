# Architecture Documentation

This document provides detailed architectural diagrams and explanations for the Employee Attrition MLOps project.

## System Overview

```mermaid
graph TD
    A[Data Sources] --> B[Data Pipeline]
    B --> C[Training Pipeline]
    C --> D[Model Registry]
    D --> E[API Service]
    E --> F[Frontend]
    B --> G[Monitoring]
    G --> H[Drift Detection]
    H --> I[Retraining Trigger]
    I --> C
    C --> J[MLflow Tracking]
    G --> J
    E --> J
```

The system follows a standard MLOps architecture with the following key components:
- Data Pipeline: Handles data ingestion and preprocessing
- Training Pipeline: Manages model development and validation
- Model Registry: Stores and versions models
- API Service: Serves predictions
- Frontend: Provides user interface
- Monitoring: Tracks system health and model performance
- MLflow: Centralizes experiment tracking and artifacts

## End-to-End Workflow

```mermaid
sequenceDiagram
    participant DB as Database
    participant DP as Data Pipeline
    participant TP as Training Pipeline
    participant MR as Model Registry
    participant API as API Service
    participant MON as Monitoring
    participant GH as GitHub Actions
    
    DB->>DP: Load Raw Data
    DP->>DP: Preprocess
    DP->>TP: Processed Data
    TP->>TP: HPO & Training
    TP->>TP: Validation
    TP->>MR: Log to MLflow
    MR->>MR: Stage Model
    MR->>API: Deploy Model
    API->>MON: Log Predictions
    MON->>MON: Detect Drift
    MON->>GH: Trigger Retraining
    GH->>TP: Start Pipeline
```

The workflow follows these steps:
1. Data ingestion from database
2. Preprocessing and validation
3. Hyperparameter optimization and training
4. Model validation and logging to MLflow
5. Staging and deployment
6. Monitoring and drift detection
7. Automated retraining via GitHub Actions

## Training Pipeline

```mermaid
graph LR
    A[Data] --> B[HPO]
    B --> C[Training]
    C --> D[Validation]
    D --> E[MLflow]
    E --> F[Staging]
    
    subgraph "Pipeline Steps"
    B
    C
    D
    end
```

The training pipeline (`optimize_train_select.py`) includes:
1. Hyperparameter Optimization
   - Bayesian optimization
   - Cross-validation
   - Performance metrics
2. Model Training
   - Best hyperparameters
   - Full training set
   - Model serialization
3. Validation
   - Holdout set evaluation
   - Fairness assessment
   - Performance metrics
4. MLflow Integration
   - Parameter logging
   - Metric tracking
   - Artifact storage
5. Staging
   - Model registration
   - Version control
   - Quality checks

## Deployment Architecture

```mermaid
graph TD
    A[Docker Compose] --> B[API Container]
    A --> C[Frontend Container]
    A --> D[MLflow Container]
    
    B --> E[Database]
    C --> B
    D --> F[Artifact Storage]
    
    subgraph "Services"
    B
    C
    D
    end
```

The deployment architecture:
1. Uses Docker Compose for orchestration
2. Runs separate containers for each service
3. Connects to external databases
4. Manages artifact storage
5. Handles service communication

## Monitoring Loop

```mermaid
graph TD
    A[API Predictions] --> B[Drift Detection]
    B --> C[Statistical Tests]
    C --> D[Alert System]
    D --> E[GitHub Actions]
    E --> F[Retraining]
    
    subgraph "Monitoring"
    B
    C
    D
    end
```

The monitoring loop:
1. Collects prediction data
2. Performs drift detection
3. Runs statistical tests
4. Generates alerts
5. Triggers retraining via GitHub Actions
6. Updates model in production

## CI/CD Pipeline

```mermaid
graph LR
    A[Code Push] --> B[GitHub Actions]
    B --> C[Lint/Test]
    C --> D[Build Images]
    D --> E[Deploy Staging]
    E --> F[Run Tests]
    F --> G[Promote to Prod]
    G --> H[Update API]
    
    subgraph "Quality Gates"
    C
    F
    end
```

The CI/CD pipeline includes:
1. Code push triggers GitHub Actions
2. Linting and testing
3. Docker image building
4. Staging deployment
5. Integration testing
6. Production promotion
7. API update

Quality gates ensure:
- Code quality standards
- Test coverage requirements
- Performance benchmarks
- Security checks

## API Architecture

```mermaid
graph TD
    A[FastAPI] --> B[Model Loader]
    B --> C[Prediction Service]
    C --> D[Monitoring]
    D --> E[MLflow]
    
    F[Request] --> A
    A --> G[Response]
    
    subgraph "Services"
    B
    C
    D
    end
```

The API architecture includes:
- FastAPI application server
- Model loading service
- Prediction service
- Monitoring integration
- MLflow tracking

## Monitoring Architecture

```mermaid
graph TD
    A[Data Stream] --> B[Drift Detection]
    B --> C[Statistical Tests]
    C --> D[Alert System]
    D --> E[Retraining Trigger]
    
    F[Reference Data] --> B
    G[Current Data] --> B
    
    subgraph "Metrics"
    H[Feature Drift]
    I[Prediction Drift]
    J[Performance Metrics]
    end
    
    B --> H
    B --> I
    B --> J
```

The monitoring system:
1. Compares current data to reference data
2. Performs statistical tests for drift
3. Generates alerts when thresholds are exceeded
4. Triggers retraining when necessary
5. Tracks multiple metrics types

## Artifact Flow

```mermaid
graph LR
    A[Training] --> B[MLflow]
    C[Validation] --> B
    D[Monitoring] --> B
    E[API] --> B
    
    B --> F[Model Artifacts]
    B --> G[Metrics]
    B --> H[Plots]
    B --> I[Reports]
    
    subgraph "MLflow Storage"
    F
    G
    H
    I
    end
```

Artifacts are stored in MLflow:
- Model files and configurations
- Training and validation metrics
- Performance plots and visualizations
- Monitoring reports
- API usage statistics

## Security Architecture

```mermaid
graph TD
    A[API Gateway] --> B[Authentication]
    B --> C[Authorization]
    C --> D[Services]
    
    E[Request] --> A
    D --> F[Response]
    
    subgraph "Security Layers"
    B
    C
    end
```

Security measures include:
- API gateway for request routing
- Authentication service
- Authorization checks
- Secure service communication
- Data encryption

## Scaling Architecture

```mermaid
graph TD
    A[Load Balancer] --> B[API Instances]
    B --> C[Database]
    B --> D[Cache]
    
    subgraph "Scaling Group"
    B1[Instance 1]
    B2[Instance 2]
    B3[Instance 3]
    end
    
    B --> B1
    B --> B2
    B --> B3
```

Scaling considerations:
- Load balancing for API instances
- Database connection pooling
- Caching for performance
- Horizontal scaling capability
- Resource monitoring 