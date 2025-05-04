# Employee Attrition MLOps Project Documentation

## Quick Links

- [Architecture](architecture.md)
- [Setup Guide](setup_details.md)
- [Getting Started](getting_started.md)
- [MLflow Usage](mlflow_usage.md)
- [Responsible AI](responsible_ai.md)
- [Monitoring](monitoring.md)
- [API Documentation](api_documentation.md)
- [Troubleshooting Guide](troubleshooting.md)
- [MLOps Workflow](mlops_workflow_guide.md)
- [CI/CD Workflow](ci_cd_workflow.md)
- [Drift Detection Guide](drift_detection_guide.md)

## Documentation Structure

### 1. Core Documentation
- **Architecture**: [System design, components, and workflows](architecture.md)
- **Setup Guide**: [Detailed installation and configuration](setup_details.md)
- **Getting Started**: [Quick start guide](getting_started.md)
- **API Documentation**: [Endpoint reference and usage](api_documentation.md)

### 2. MLOps Components
- **MLflow Usage**: [Experiment tracking and model management](mlflow_usage.md)
- **Monitoring**: [Drift detection and retraining](monitoring.md)
- **MLOps Workflow**: [End-to-end pipeline guide](mlops_workflow_guide.md)
- **CI/CD Workflow**: [Continuous integration and deployment](ci_cd_workflow.md)
- **Drift Detection**: [Implementation guide](drift_detection_guide.md)
- **Responsible AI**: [Fairness assessment and bias mitigation](responsible_ai.md)

### 3. Reference Materials
- **Troubleshooting**: [Common issues and solutions](troubleshooting.md)

## Getting Started

1. **Installation**
   ```bash
   # Clone repository
   git clone https://github.com/BTCJULIAN/Employee-Attrition-2.git
   cd Employee-Attrition-2

   # Install dependencies
   poetry install
   ```

2. **Configuration**
   ```bash
   # Copy and edit environment variables
   cp .env.example .env
   ```

3. **Start Services**
   ```bash
   # Start all services with Docker
   docker-compose up --build
   ```

## Key Components

### Data Pipeline
- Automated data ingestion and preprocessing
- Feature engineering and validation
- Data quality monitoring

### Model Development
- Hyperparameter optimization
- Model training and validation
- MLflow experiment tracking

### Production System
- FastAPI prediction service
- Streamlit frontend
- Docker containerization

### MLOps Infrastructure
- GitHub Actions CI/CD
- MLflow model registry
- Drift detection and monitoring

## Best Practices

### Development
- Follow PEP 8 style guide
- Use type hints
- Write comprehensive tests
- Document all public interfaces
- Follow semantic versioning

### Deployment
- Use semantic versioning
- Follow CI/CD pipeline
- Monitor performance
- Maintain documentation
- Regular security updates

### Monitoring
- Regular drift detection
- Performance tracking
- Automated alerts
- Model versioning
- Security monitoring

## Support

- [GitHub Issues](https://github.com/BTCJULIAN/Employee-Attrition-2/issues)
- [Documentation](../README.md)
- [Troubleshooting Guide](troubleshooting.md)

## License

MIT 