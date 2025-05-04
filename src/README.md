# Employee Attrition MLOps - Source Code

This directory contains the core source code for the Employee Attrition MLOps project. The project is structured into several key components:

## Directory Structure

### `employee_attrition_mlops/`
Core ML pipeline implementation including:
- Data processing and feature engineering
- Model training and evaluation
- Prediction pipelines
- API endpoints for model serving

### `monitoring/`
Drift detection and model monitoring system:
- Feature drift detection
- Prediction drift monitoring
- Statistical tests implementation
- Alert generation

### `frontend/`
Streamlit-based web interface:
- Live prediction interface
- Model performance visualization
- User interaction components

### `config/`
Configuration management:
- Environment variables
- Model parameters
- System settings
- API configurations

### `utils/`
Utility functions and helpers:
- Common functions
- Data processing utilities
- Logging setup
- Error handling

## Key Components

### ML Pipeline
The ML pipeline is implemented in `employee_attrition_mlops/` and includes:
- Data preprocessing and feature engineering
- Model training with hyperparameter optimization
- Model evaluation and selection
- Prediction pipeline for serving

### Monitoring System
The monitoring system in `monitoring/` provides:
- Real-time drift detection
- Statistical analysis of data changes
- Automated alerts for model degradation
- Performance tracking

### Frontend
The Streamlit frontend in `frontend/` offers:
- Interactive prediction interface
- Model performance visualization
- User-friendly data input
- Results display

## Getting Started

1. Ensure all dependencies are installed (see root `pyproject.toml`)
2. Set up environment variables in `.env`
3. Run the ML pipeline:
   ```bash
   python -m src.employee_attrition_mlops.pipelines
   ```
4. Start the monitoring system:
   ```bash
   python -m src.monitoring.drift_detection
   ```
5. Launch the frontend:
   ```bash
   streamlit run src/frontend/app.py
   ```

## Development Guidelines

- Follow PEP 8 style guide
- Use type hints for all functions
- Document all public functions and classes
- Write unit tests for new features
- Update documentation when making changes

## Testing

Run the test suite:
```bash
pytest tests/
```

## Documentation

For detailed documentation of each component, see:
- [ML Pipeline Documentation](employee_attrition_mlops/README.md)
- [Monitoring System Documentation](monitoring/README.md)
- [Frontend Documentation](frontend/README.md)
- [Configuration Guide](config/README.md) 