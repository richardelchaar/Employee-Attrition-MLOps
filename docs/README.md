# Employee Attrition MLOps Project Documentation

## Quick Links

- [Project Overview](index.md)
- [Architecture](architecture.md)
- [Setup Guide](setup_details.md)
- [MLflow Usage](mlflow_usage.md)
- [Responsible AI](responsible_ai.md)
- [Monitoring Strategy](monitoring_strategy.md)
- [API Documentation](api_documentation.md)
- [Troubleshooting Guide](troubleshooting.md)

## Documentation Structure

### 1. Core Documentation
- **Project Overview**: High-level project description, objectives, and features
- **Architecture**: System design, components, and workflows
- **Setup Guide**: Detailed installation and configuration instructions
- **API Documentation**: Endpoint reference and usage examples

### 2. MLOps Components
- **MLflow Usage**: Experiment tracking and model management
- **Monitoring Strategy**: Drift detection and retraining workflows
- **Responsible AI**: Fairness assessment and bias mitigation

### 3. Development Guides
- **Development Workflow**: Branching strategy and guidelines
- **Testing**: Test suite execution and coverage
- **Deployment**: CI/CD pipeline and containerization
- **Versioning**: Semantic versioning and release process

### 4. Reference Materials
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Development and deployment guidelines
- **Glossary**: Key terms and concepts
- **FAQ**: Frequently asked questions

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
- [Documentation](docs/)
- [Troubleshooting Guide](docs/troubleshooting.md)

## License

MIT 