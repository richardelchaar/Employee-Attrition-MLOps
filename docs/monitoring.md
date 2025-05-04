# Model Monitoring and Retraining

This document describes the automated monitoring and retraining process for the Employee Attrition model, including the overall strategy, drift detection methods, baseline generation, and retraining triggers.

## Automation Schedule

### Monthly Production Workflow
The complete production workflow runs automatically on the 1st of every month at midnight UTC. This workflow includes:

1. **Unit Testing and Linting**
   - Runs pytest for all test cases
   - Performs code linting (black, isort, flake8, mypy)
   - Ensures code quality before pipeline execution

2. **Pipeline Execution**
   - Builds and runs services with Docker Compose
   - Executes batch prediction
   - Performs drift detection
   - Triggers retraining if needed

3. **Drift Detection**
   - Compares current data against reference dataset
   - Calculates drift metrics for all features
   - Generates drift report
   - Creates GitHub issue with results

4. **Model Retraining (if needed)**
   - Triggered if drift exceeds thresholds
   - Optimizes hyperparameters
   - Trains new model
   - Evaluates performance

5. **Docker Image Management**
   - Builds and pushes Docker images if drift detected
   - Updates MLflow server image
   - Updates API image
   - Updates frontend image
   - Updates drift API image

### Manual Triggering
The workflow can also be triggered manually through GitHub Actions:
1. Go to the "Actions" tab in the repository
2. Select "Monthly MLOps Pipeline"
3. Click "Run workflow"

## Monitoring Strategy

### Types of Drift
- **Feature Drift**: Changes in feature distributions using Evidently's statistical tests
- **Prediction Drift**: Changes in prediction distributions and performance metrics
- **Concept Drift**: Changes in feature-target relationship

### Drift Detection Implementation
The system uses Evidently's statistical tests for drift detection:

1. **Feature Drift Detection**
   ```python
   from monitoring.drift_detection import DriftDetector
   
   detector = DriftDetector(drift_threshold=0.05)
   drift_detected, drift_score, drifted_features = detector.detect_drift(
       reference_data=reference_data,
       current_data=current_data
   )
   ```

2. **Prediction Drift Detection**
   ```python
   drift_detected, drift_score = detector.detect_prediction_drift(
       reference_data=reference_data,
       current_data=current_data,
       prediction_column="prediction"
   )
   ```

### Thresholds and Alerts

1. **Drift Thresholds**
   - Default threshold: 0.05 (5%)
   - Feature drift: Tested per feature using Evidently's statistical tests
   - Prediction drift: Monitored using distribution comparison
   - Overall drift: Share of drifted features

2. **Alert System**
   - Automatic GitHub issue creation with:
     - Feature drift results (drift detected, drift share, drifted features)
     - Prediction drift results (drift detected, drift score)
     - Retraining status
     - Batch prediction summary
   - MLflow metrics logging:
     - `drift_detected`: Binary indicator
     - `drift_score`: Overall drift score
     - `n_drifted_features`: Number of drifted features
     - `drifted_features`: List of drifted features

## Model Retraining and Promotion

### Automated Retraining Process
When drift is detected:
1. GitHub Actions workflow automatically triggers retraining
2. New model is trained and evaluated
3. Results are logged to MLflow
4. GitHub issue is updated with retraining results
5. Docker images are rebuilt and pushed if drift detected

### Model Promotion
Models are promoted to production through a controlled process:
- PR with "model-promotion" label
- Approval and merge triggers automated promotion
- Manual promotion via script is also supported

### Monitoring MLflow
- View model versions and their stages
- Compare model metrics
- Analyze drift reports
- Track model lineage

## Best Practices
- Regular monitoring and review of drift detection results
- Keep reference data up to date
- Document all model promotions
- Maintain test coverage for new models
- Use PR templates for model updates
- Include performance metrics in PRs
- Follow code review guidelines

## Troubleshooting
- Verify MLflow server is running
- Check reference dataset availability
- Review drift thresholds in configuration
- Ensure correct run ID/version for promotion
- Confirm environment variables and dependencies
- Check Docker service status
- Verify GitHub Actions workflow status

## Example Usage

### Manual Monitoring
```bash
# Normal drift check
python scripts/drift_detection.py

# Simulate drift (for testing)
python scripts/drift_detection.py --simulate-drift
```

### MLflow UI
1. Start MLflow server: `mlflow server --host 127.0.0.1 --port 5001`
2. Open `http://127.0.0.1:5001` in your browser
3. Navigate to the latest run to view drift metrics

### Manual Promotion
```bash
# Promote latest staging model
python scripts/promote_model.py

# Promote specific version
python scripts/promote_model.py --model-version <version>

# Promote by run ID
python scripts/promote_model.py --run-id <run_id>
```

## Model Governance

### Retraining Decision Process

1. **Automated Analysis**:
   - Drift detection runs monthly
   - Thresholds are checked automatically
   - Results logged to MLflow

2. **Review Requirements**:
   - Performance metrics comparison
   - Feature importance changes
   - Data quality checks
   - Model fairness metrics

3. **Approval Process**:
   - Create PR with "model-promotion" label
   - Required reviewers must approve
   - Performance comparison must be documented
   - Model cards must be updated

### Model Cards

Each model version should maintain an updated model card including:
- Training data timeframe
- Feature drift history
- Performance metrics over time
- Known limitations
- Approved use cases

### Monitoring Metrics

The following metrics are tracked:
1. **Data Quality**:
   - Missing value patterns
   - Feature distributions
   - Data volume

2. **Model Performance**:
   - Prediction drift
   - Feature importance stability
   - Performance metrics (AUC, precision, recall)

3. **Operational Metrics**:
   - Prediction latency
   - Data processing time
   - Error rates

### Retraining Workflow

1. **Trigger Conditions**:
   - Significant feature drift detected
   - Performance degradation observed
   - Scheduled retraining period reached

2. **Automated Steps**:
   - Data validation
   - Model training
   - Performance evaluation
   - Results logging to MLflow
   - Docker image updates

3. **Manual Review**:
   - Performance comparison
   - Feature drift analysis
   - Model behavior changes
   - Fairness metrics review

4. **Promotion Process**:
   - Create promotion PR
   - Document changes
   - Get approvals
   - Automated promotion on merge 