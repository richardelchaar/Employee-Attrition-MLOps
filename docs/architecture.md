# System Architecture

## Component Diagram

```mermaid
graph TB
    subgraph Frontend
        Streamlit[Streamlit App]
    end

    subgraph Backend
        FastAPI[FastAPI Service]
        MLflow[MLflow Server]
        DB[(Database)]
    end

    subgraph CI/CD
        GitHub[GitHub Actions]
        Docker[Docker Compose]
    end

    subgraph Monitoring
        Drift[Drift Detection]
        Retrain[Model Retraining]
    end

    %% Frontend to Backend
    Streamlit -->|API Calls| FastAPI
    Streamlit -->|Model Info| MLflow

    %% Backend Components
    FastAPI -->|Load Model| MLflow
    FastAPI -->|Log Predictions| DB
    MLflow -->|Store Artifacts| DB

    %% CI/CD Flow
    GitHub -->|Trigger| Docker
    Docker -->|Deploy| FastAPI
    Docker -->|Deploy| MLflow
    Docker -->|Deploy| Streamlit

    %% Monitoring Flow
    Drift -->|Detect Changes| DB
    Drift -->|Trigger| Retrain
    Retrain -->|Update Model| MLflow
    Retrain -->|Notify| GitHub

    %% Styling
    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px
    classDef cicd fill:#bfb,stroke:#333,stroke-width:2px
    classDef monitoring fill:#fbb,stroke:#333,stroke-width:2px

    class Streamlit frontend
    class FastAPI,MLflow,DB backend
    class GitHub,Docker cicd
    class Drift,Retrain monitoring
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant MLflow
    participant DB
    participant GitHub

    %% User Interaction
    User->>Streamlit: Submit Employee Data
    Streamlit->>FastAPI: POST /predict
    FastAPI->>MLflow: Load Model
    MLflow-->>FastAPI: Return Model
    FastAPI->>FastAPI: Make Prediction
    FastAPI->>DB: Log Prediction
    FastAPI-->>Streamlit: Return Prediction
    Streamlit-->>User: Display Result

    %% Monitoring Flow
    loop Daily
        GitHub->>DB: Run Drift Detection
        alt Drift Detected
            DB->>GitHub: Trigger Retraining
            GitHub->>MLflow: Train New Model
            MLflow->>DB: Store Model
            GitHub->>FastAPI: Deploy New Model
        end
    end
```

## Deployment Architecture

```mermaid
graph TB
    subgraph Docker Containers
        direction TB
        FastAPI[FastAPI Container]
        MLflow[MLflow Container]
        Streamlit[Streamlit Container]
    end

    subgraph External Services
        GitHub[GitHub Actions]
        DB[(Database)]
    end

    subgraph Volumes
        MLruns[MLruns Volume]
        MLartifacts[MLartifacts Volume]
    end

    %% Container Connections
    FastAPI -->|Load Model| MLflow
    FastAPI -->|Log Data| DB
    Streamlit -->|API Calls| FastAPI
    Streamlit -->|Model Info| MLflow

    %% Volume Mounts
    MLflow -->|Store Runs| MLruns
    MLflow -->|Store Artifacts| MLartifacts

    %% External Connections
    GitHub -->|Deploy| Docker Containers

    %% Styling
    classDef container fill:#bbf,stroke:#333,stroke-width:2px
    classDef external fill:#fbb,stroke:#333,stroke-width:2px
    classDef volume fill:#bfb,stroke:#333,stroke-width:2px

    class FastAPI,MLflow,Streamlit container
    class GitHub,DB external
    class MLruns,MLartifacts volume
```

## Component Descriptions

### Frontend (Streamlit)
- Interactive web interface for predictions
- Real-time model information display
- User-friendly forms for data input

### Backend (FastAPI)
- RESTful API for predictions
- Model loading and inference
- Prediction logging
- Health checks

### MLflow Server
- Model versioning and tracking
- Experiment management
- Artifact storage
- Model registry

### Database
- Stores historical data
- Logs predictions
- Tracks model performance

### CI/CD Pipeline
- Automated testing
- Drift detection
- Model retraining
- Deployment automation

### Monitoring
- Data drift detection
- Model performance monitoring
- Automated retraining triggers
- Health checks 