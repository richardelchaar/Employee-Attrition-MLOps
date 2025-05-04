# Detailed Setup Guide

This document provides in-depth setup instructions and troubleshooting guides for the Employee Attrition MLOps project.

## macOS Setup Guide

### ODBC Driver Installation

1. **Install unixodbc**
   ```bash
   brew install unixodbc
   ```

2. **Add Microsoft ODBC tap**
   ```bash
   brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
   brew update
   ```

3. **Install Microsoft ODBC Driver**
   ```bash
   brew install msodbcsql17 mssql-tools
   ```

4. **Verify Installation**
   ```bash
   # Check installed drivers
   odbcinst -q -d
   
   # Check data sources
   odbcinst -q -s
   ```

5. **Common Issues & Solutions**

   **Issue**: Driver not found
   ```bash
   # Check driver path
   ls /usr/local/lib/libmsodbcsql*
   
   # If missing, reinstall
   brew reinstall msodbcsql17
   ```

   **Issue**: Connection failures
   ```bash
   # Test connection
   sqlcmd -S your_server -U your_username -P your_password
   
   # Check error logs
   cat /var/log/odbc.log
   ```

### Poetry Setup

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3.11 -
   ```

2. **Configure Poetry**
   ```bash
   # Add to PATH
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   
   # Configure virtualenvs
   poetry config virtualenvs.in-project true
   ```

3. **Common Poetry Issues**

   **Issue**: Dependency resolution failures
   ```bash
   # Clear cache
   poetry cache clear . --all
   
   # Update lock file
   poetry lock --no-update
   
   # Reinstall dependencies
   poetry install
   ```

   **Issue**: Virtual environment problems
   ```bash
   # Remove existing venv
   rm -rf .venv
   
   # Recreate environment
   poetry env use python3.11
   poetry install
   ```

### GitHub Authentication

1. **Generate Personal Access Token (PAT)**
   - Go to GitHub Settings > Developer Settings > Personal Access Tokens
   - Generate new token with required scopes:
     - repo (full control)
     - workflow
     - read:packages
     - write:packages

2. **Configure Git**
   ```bash
   # Store credentials
   git config --global credential.helper store
   
   # Test authentication
   git push
   ```

3. **Environment Variables**
   ```bash
   # Add to .env
   GITHUB_TOKEN=your_pat_here
   ```

### Python Environment

1. **Install Python 3.11**
   ```bash
   # Using pyenv
   brew install pyenv
   pyenv install 3.11.0
   pyenv global 3.11.0
   ```

2. **Verify Installation**
   ```bash
   python --version
   which python
   ```

3. **Common Python Issues**

   **Issue**: Multiple Python versions
   ```bash
   # List installed versions
   pyenv versions
   
   # Set local version
   pyenv local 3.11.0
   ```

   **Issue**: Path conflicts
   ```bash
   # Check PATH
   echo $PATH
   
   # Reorder PATH
   export PATH="$(pyenv root)/shims:$PATH"
   ```

### Docker Setup

1. **Install Docker Desktop**
   ```bash
   brew install --cask docker
   ```

2. **Verify Installation**
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Common Docker Issues**

   **Issue**: Permission denied
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   
   # Restart Docker
   sudo systemctl restart docker
   ```

   **Issue**: Port conflicts
   ```bash
   # Check used ports
   lsof -i :8000
   
   # Stop conflicting services
   sudo lsof -t -i :8000 | xargs kill -9
   ```

### Database Setup

1. **Azure SQL Configuration**
   ```bash
   # Test connection
   sqlcmd -S your_server.database.windows.net -U your_username -P your_password
   ```

2. **Connection String Format**
   ```
   DATABASE_URL_PYMSSQL=mssql+pymssql://username:password@hostname:1433/database
   ```

3. **Common Database Issues**

   **Issue**: Connection timeout
   ```bash
   # Check firewall rules
   az sql server firewall-rule list --resource-group your_group --server your_server
   
   # Add your IP
   az sql server firewall-rule create --resource-group your_group --server your_server --name AllowMyIP --start-ip-address your_ip --end-ip-address your_ip
   ```

   **Issue**: Authentication failed
   ```bash
   # Reset password
   az sql server update --admin-password new_password --name your_server --resource-group your_group
   ```

### MLflow Setup

1. **Start MLflow Server**
   ```bash
   poetry run mlflow ui --host 0.0.0.0 --port 5001
   ```

2. **Verify Connection**
   ```bash
   # Check tracking URI
   poetry run python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

3. **Common MLflow Issues**

   **Issue**: Artifact storage
   ```bash
   # Set artifact location
   export MLFLOW_ARTIFACT_LOCATION=file:///path/to/artifacts
   
   # Start server with artifact location
   poetry run mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --artifacts-destination /path/to/artifacts
   ```

   **Issue**: Database connection
   ```bash
   # Use SQLite for local development
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   ```

### Testing Setup

1. **Run Tests**
   ```bash
   # All tests
   poetry run pytest
   
   # With coverage
   poetry run pytest --cov=src --cov-report=html
   ```

2. **Common Testing Issues**

   **Issue**: Missing dependencies
   ```bash
   # Install test dependencies
   poetry install --with test
   ```

   **Issue**: Test failures
   ```bash
   # Run specific test
   poetry run pytest tests/test_file.py::test_function -v
   
   # Debug mode
   poetry run pytest --pdb
   ```

### Troubleshooting Guide

1. **General Issues**
   - Check error logs in `logs/` directory
   - Verify environment variables
   - Check service status
   - Review configuration files

2. **Performance Issues**
   - Monitor resource usage
   - Check database performance
   - Review API response times
   - Analyze MLflow metrics

3. **Security Issues**
   - Verify authentication
   - Check firewall rules
   - Review access logs
   - Validate encryption

4. **Deployment Issues**
   - Check Docker logs
   - Verify network connectivity
   - Review deployment scripts
   - Check service health

### Support Resources

1. **Documentation**
   - [Project Documentation](docs/)
   - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
   - [FastAPI Documentation](https://fastapi.tiangolo.com/)
   - [Docker Documentation](https://docs.docker.com/)

2. **Community Support**
   - GitHub Issues
   - Stack Overflow
   - Project Discord
   - MLOps Community

3. **Professional Support**
   - Azure Support
   - GitHub Enterprise Support
   - Docker Enterprise Support 