# CI/CD and Model Lifecycle Automation

This document describes the complete CI/CD pipeline and model lifecycle management for the Employee Attrition project.

## CI/CD Pipeline Overview

The CI/CD pipeline consists of two main workflows:

### 1. Main CI/CD Workflow (`.github/workflows/ci.yml`)
- **Trigger Conditions**:
  - Push to main branch
  - Pull requests
  - Daily scheduled runs

- **Jobs**:
  1. **Test Job**
     - Runs unit tests
     - Performs code linting
     - Type checking

  2. **Deploy Job**
     - Builds Docker images
     - Pushes to container registry
     - Deploys to production

  3. **Drift Detection Job**
     - Runs daily
     - Checks for data drift
     - Triggers retraining if needed

  4. **Fairness Testing Job**
     - Runs daily
     - Monitors model fairness
     - Logs fairness metrics

### 2. Model Promotion Workflow (`.github/workflows/model_promotion.yml`)
- **Trigger Conditions**:
  - PR review approval
  - PR must have "model-promotion" label

- **Jobs**:
  1. **Promote Model Job**
     - Starts MLflow server
     - Promotes model to production
     - Handles success/failure notifications

## Model Lifecycle Management

### Model Monitoring

#### Drift Detection
- **Automated Monitoring**:
  - Daily runs via GitHub Actions
  - Configuration: `.github/workflows/ci.yml`
  - Compares current vs. reference data
  - Uses `scripts/drift_detection.py` for detection
  - Uses `scripts/create_drift_reference.py` for reference data

#### Drift Metrics
- Numerical feature statistical drift
- Key monitored features:
  - Age
  - MonthlyIncome
  - YearsAtCompany
  - Other numerical features

#### Alerting System
- Automatic GitHub issue creation
- Issue contents:
  - Drifted features list
  - Drift magnitude
  - Recommended actions

### Model Retraining Process

#### Automated Retraining
1. **Trigger Conditions**:
   - Significant feature drift (>30% features)
   - Critical feature drift (p-value < 0.01)
   - Performance degradation
   - Monthly scheduled retraining

2. **Retraining Workflow**:
   - Data validation
   - Model training
   - Performance evaluation
   - MLflow logging
   - Fairness testing

3. **Review Process**:
   - Performance comparison
   - Feature drift analysis
   - Model behavior changes
   - Fairness metrics review

#### Model Promotion

1. **Automated Promotion**:
   - PR with "model-promotion" label
   - Required approvals
   - Automated deployment
   - Health checks

2. **Manual Promotion**:
   ```bash
   # Promote latest staging model
   python scripts/promote_model.py

   # Promote specific version
   python scripts/promote_model.py --model-version <version>

   # Promote by run ID
   python scripts/promote_model.py --run-id <run_id>
   ```

### Model Governance

#### Model Cards
- Training data timeframe
- Feature drift history
- Performance metrics
- Known limitations
- Approved use cases

#### Monitoring Metrics
1. **Data Quality**:
   - Missing value patterns
   - Feature distributions
   - Data volume

2. **Model Performance**:
   - Prediction drift
   - Feature importance
   - Performance metrics

3. **Operational Metrics**:
   - Prediction latency
   - Processing time
   - Error rates

## Best Practices

### 1. Code Quality
- Follow PEP 8 guidelines
- Maintain test coverage
- Document all changes
- Use type hints

### 2. Model Management
- Regular drift monitoring
- Document all promotions
- Maintain test coverage
- Update model cards

### 3. Workflow Management
- Use PR templates
- Include performance metrics
- Follow review guidelines
- Document changes

## Troubleshooting

### Common Issues

1. **Drift Detection**:
   - MLflow server status
   - Reference data availability
   - Threshold configuration
   - Check drift detection logs

2. **Promotion Failures**:
   - Run ID/version verification
   - MLflow connection
   - GitHub Actions logs

3. **Environment Setup**:
   - Environment variables
   - Dependencies
   - MLflow server

### Baseline Profiles
- Stored in MLflow artifacts
- Include:
  - Feature distributions
  - Missing value patterns
  - Category distributions

## Manual Commands

### Drift Detection
```bash
# Normal drift check
python scripts/drift_detection.py

# Simulate drift
python scripts/drift_detection.py --simulate-drift

# Create reference data
python scripts/create_drift_reference.py
```

### MLflow Access
1. Start server: `mlflow server --host 127.0.0.1 --port 5001`
2. Access UI: `http://127.0.0.1:5001`
3. View metrics and artifacts

## Security Considerations

1. **Access Control**:
   - Role-based access
   - API key management
   - Environment isolation

2. **Data Protection**:
   - Sensitive data handling
   - Encryption at rest
   - Secure communication

3. **Audit Trail**:
   - Model version tracking
   - Change logging
   - Access monitoring

## MLflow Integration

### Model Tracking
- Model versions and stages
- Performance metrics
- Artifacts and drift references
- Experiment tracking

### Artifact Management
- Drift reference data
- Model artifacts
- Performance reports
- Fairness metrics

### Version Control
- Model versioning
- Stage transitions
- Rollback capabilities
- Version comparison 