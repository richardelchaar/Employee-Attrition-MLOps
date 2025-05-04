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
services:
  mlflow-server:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    ports:
      - "5001:5001"
    volumes:
      - ./mlruns:/mlflow_runs
      - ./mlartifacts:/mlflow_artifacts
    networks:
      - app_network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:5001"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${API_PORT:-8000}:8000"
    depends_on:
      mlflow-server:
        condition: service_healthy
    environment:
      - DATABASE_URL=${DATABASE_URL_PYMSSQL}
      - DATABASE_URL_PYMSSQL=${DATABASE_URL_PYMSSQL}
      - DATABASE_URL_PYODBC=${DATABASE_URL_PYODBC}
      - MLFLOW_TRACKING_URI=http://mlflow-server:5001
      - DB_PREDICTION_LOG_TABLE=${DB_PREDICTION_LOG_TABLE:-prediction_logs}
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
    volumes:
      - ./src:/app/src
    command: ["uvicorn", "employee_attrition_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
    networks:
      - app_network
    healthcheck:
      test: "wget -qO- http://localhost:8000/health > /dev/null || exit 1"
      interval: 10s
      timeout: 5s
      retries: 5

  drift-api:
    build:
      context: .
      dockerfile: Dockerfile.drift
    ports:
      - "${DRIFT_PORT:-8001}:8000"
    depends_on:
      mlflow-server:
        condition: service_healthy
      api:
        condition: service_healthy
    environment:
      - DATABASE_URL=${DATABASE_URL_PYMSSQL}
      - DATABASE_URL_PYMSSQL=${DATABASE_URL_PYMSSQL}
      - DATABASE_URL_PYODBC=${DATABASE_URL_PYODBC}
      - MLFLOW_TRACKING_URI=http://mlflow-server:5001
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
    volumes:
      - ./reference_data:/app/reference_data
      - ./reference_predictions:/app/reference_predictions
      - ./reports:/app/reports
    networks:
      - app_network
    healthcheck:
      test: ["CMD-SHELL", "curl --fail http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "${FRONTEND_PORT:-8501}:8501"
    depends_on:
      api:
        condition: service_healthy
      drift-api:
        condition: service_healthy
    environment:
      - API_URL=http://api:8000
      - DRIFT_API_URL=http://drift-api:8000
    volumes:
      - ./src/frontend:/app/src/frontend
    command: ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    networks:
      - app_network

networks:
  app_network:
    driver: bridge
```

## Volume Management

### Persistent Storage
- `mlruns/`: MLflow experiment tracking
- `mlartifacts/`: Model artifacts and metadata
- `reference_data/`: Drift detection baselines
- `reference_predictions/`: Reference predictions for drift comparison
- `reports/`: Generated drift reports
- `src/`: Source code (mounted for development)

### Volume Permissions
```bash
# Set correct permissions for volumes
chmod -R 777 mlruns mlartifacts reference_data reference_predictions reports
```

## Environment Variables

Required environment variables for Docker services:
```env
# Database Configuration
DATABASE_URL_PYMSSQL=mssql+pymssql://user:pass@host/db
DATABASE_URL_PYODBC=mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server

# Service Ports
API_PORT=8000
DRIFT_PORT=8001
FRONTEND_PORT=8501

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow-server:5001

# API Configuration
DB_PREDICTION_LOG_TABLE=prediction_logs
PYTHONPATH=/app
```

## Service Health Checks

Each service includes health checks to ensure proper startup order:

1. **MLflow Server**
   - Checks if the server is accessible on port 5001
   - Retries every 10 seconds
   - Maximum 5 retries

2. **API Service**
   - Checks the /health endpoint
   - Retries every 10 seconds
   - Maximum 5 retries

3. **Drift API**
   - Checks the /health endpoint
   - 20-second start period
   - Retries every 10 seconds
   - Maximum 5 retries

## Troubleshooting

### Common Issues

1. **Volume Mount Issues**
   ```bash
   # Check volume mounts
   docker-compose exec api ls -la /app/src
   docker-compose exec mlflow-server ls -la /mlflow_runs
   ```

2. **Permission Issues**
   ```bash
   # Fix permissions
   sudo chown -R 1000:1000 mlruns mlartifacts reference_data reference_predictions reports
   ```

3. **Service Health Issues**
   ```bash
   # Check service health
   docker-compose ps
   docker-compose logs mlflow-server
   docker-compose logs api
   docker-compose logs drift-api
   ```

4. **Network Issues**
   ```bash
   # Check network connectivity
   docker-compose exec api ping mlflow-server
   docker-compose exec drift-api ping api
   ```

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs frontend
docker-compose logs mlflow-server
docker-compose logs drift-api

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