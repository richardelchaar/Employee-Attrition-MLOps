# Troubleshooting Guide

This guide provides solutions to common issues you might encounter while working with the Employee Attrition MLOps project.

## Common Issues

### 1. Installation Issues

#### Poetry Installation
```bash
# If poetry install fails
poetry cache clear . --all
poetry install --no-cache

# If dependency resolution fails
poetry update
poetry install
```

#### Python Version Issues
```bash
# Check Python version
python --version

# If wrong version, use pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

### 2. Database Connection Issues

#### ODBC Driver Problems
```bash
# Check ODBC driver installation
odbcinst -q -d

# If missing, install drivers
brew install unixodbc
brew tap microsoft/mssql-release
brew install msodbcsql17 mssql-tools
```

#### Connection String Issues
```bash
# Verify connection string format
echo $DATABASE_URL_PYMSSQL

# Test connection
python -c "from src.employee_attrition_mlops.config import get_db_connection; get_db_connection()"
```

### 3. MLflow Issues

#### Server Connection
```bash
# Check MLflow server status
curl http://localhost:5001/health

# If down, restart server
poetry run mlflow ui --port 5001
```

#### Artifact Storage
```bash
# Check artifact storage permissions
ls -la mlruns/

# Fix permissions if needed
chmod -R 755 mlruns/
```

### 4. Docker Issues

#### Container Startup
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

#### Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :8501
lsof -i :5001

# Kill conflicting processes
kill -9 <PID>
```

### 5. API Issues

#### Service Health
```bash
# Check API health
curl http://localhost:8000/health

# Check model status
curl http://localhost:8000/model-info
```

#### Prediction Errors
```bash
# Check input format
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'

# View API logs
docker-compose logs -f api
```

### 6. Frontend Issues

#### Streamlit Problems
```bash
# Check Streamlit server
curl http://localhost:8501

# Restart Streamlit
streamlit run src/frontend/app.py
```

#### API Connection
```bash
# Check API URL
echo $API_URL

# Test API connection
curl $API_URL/health
```

## Debugging Tools

### 1. Logging
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f mlflow
```

### 2. Database Debugging
```bash
# Connect to database
python -c "from src.employee_attrition_mlops.config import get_db_connection; conn = get_db_connection(); print(conn.execute('SELECT 1').fetchone())"
```

### 3. MLflow Debugging
```bash
# Check MLflow tracking
poetry run mlflow ui

# View experiment runs
poetry run mlflow runs list
```

## Performance Issues

### 1. Slow Predictions
```bash
# Check API response time
curl -w "%{time_total}\n" -o /dev/null -s http://localhost:8000/health

# Profile API
python -m cProfile -o profile.prof src/api/main.py
```

### 2. High Memory Usage
```bash
# Check memory usage
docker stats

# Monitor container resources
docker-compose top
```

## Security Issues

### 1. Environment Variables
```bash
# Check sensitive variables
grep -r "API_KEY\|SECRET\|PASSWORD" .

# Update .env file
cp .env.example .env
```

### 2. File Permissions
```bash
# Check file permissions
ls -la

# Fix permissions
chmod 600 .env
chmod 755 scripts/
```

## Getting Help

1. **Check Documentation**
   - Review relevant documentation
   - Search for similar issues
   - Check troubleshooting guides

2. **Create Issue**
   - Provide detailed error message
   - Include environment details
   - Share relevant logs
   - Describe steps to reproduce

3. **Contact Support**
   - Open GitHub issue
   - Join project discussions
   - Contact maintainers

## Best Practices

1. **Regular Maintenance**
   - Keep dependencies updated
   - Monitor system health
   - Regular backups
   - Security updates

2. **Monitoring**
   - Set up alerts
   - Monitor logs
   - Track performance
   - Check security

3. **Documentation**
   - Keep notes
   - Update guides
   - Share solutions
   - Maintain changelog 