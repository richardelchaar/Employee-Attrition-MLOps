# Docker Guide

This guide provides detailed information about the Docker setup for the Employee Attrition MLOps project. For high-level architecture information, see [Architecture](architecture.md).

## Docker Files

### 1. Main Application (`Dockerfile`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
COPY . .
CMD ["poetry", "run", "uvicorn", "src.employee_attrition_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Frontend (`Dockerfile.frontend`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
COPY . .
EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. Drift Detection (`Dockerfile.drift`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
COPY . .
CMD ["poetry", "run", "python", "scripts/check_production_drift.py"]
```

### 4. MLflow Server (`Dockerfile.mlflow`)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
COPY . .
EXPOSE 5001
CMD ["poetry", "run", "mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]
```

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5001
    volumes:
      - ./mlartifacts:/app/mlartifacts
    depends_on:
      - mlflow

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    volumes:
      - ./mlartifacts:/app/mlartifacts
    depends_on:
      - api

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    ports:
      - "5001:5001"
    volumes:
      - ./mlruns:/app/mlruns
      - ./mlartifacts:/app/mlartifacts

  drift:
    build:
      context: .
      dockerfile: Dockerfile.drift
    volumes:
      - ./reference_data:/app/reference_data
      - ./reports:/app/reports
      - ./mlartifacts:/app/mlartifacts
    depends_on:
      - mlflow
```

## Volume Management

### Persistent Storage
- `mlruns/`: MLflow experiment tracking
- `mlartifacts/`: Model artifacts and metadata
- `reference_data/`: Drift detection baselines
- `reports/`: Generated drift reports

### Volume Permissions
```bash
# Set correct permissions for volumes
chmod -R 777 mlruns mlartifacts reference_data reports
```

## Environment Variables

Required environment variables for Docker services:
```env
# API Service
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=http://mlflow:5001

# MLflow
MLFLOW_MODEL_NAME=employee_attrition_model
MLFLOW_MODEL_STAGE=Production

# Drift Detection
DRIFT_THRESHOLD=0.05
REFERENCE_DATA_PATH=/app/reference_data
```

## Troubleshooting

### Common Issues

1. **Volume Mount Issues**
   ```bash
   # Check volume mounts
   docker-compose exec api ls -la /app/mlartifacts
   docker-compose exec mlflow ls -la /app/mlruns
   ```

2. **Permission Issues**
   ```bash
   # Fix permissions
   sudo chown -R 1000:1000 mlruns mlartifacts
   ```

3. **Network Issues**
   ```bash
   # Check network connectivity
   docker-compose exec api ping mlflow
   docker-compose exec frontend ping api
   ```

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs frontend
docker-compose logs mlflow
docker-compose logs drift

# Follow logs
docker-compose logs -f [service]
```

### Container Management
```bash
# Rebuild specific service
docker-compose build api

# Restart service
docker-compose restart api

# Remove containers and volumes
docker-compose down -v

# Clean up unused resources
docker system prune
``` 