# Model Monitoring and Retraining Process

This document describes the automated monitoring and retraining process for the Employee Attrition model.

## Model Monitoring

### Drift Detection

The system automatically monitors for data drift using the following process:

1. **Scheduled Monitoring**: 
   - Runs daily via GitHub Actions
   - Configured in `.github/workflows/drift_detection.yml`
   - Compares current data against the reference dataset

2. **Drift Metrics**:
   - Monitors numerical features for statistical drift
   - Key features monitored include:
     - Age
     - MonthlyIncome
     - YearsAtCompany
     - Other numerical features

3. **Alerting**:
   - Creates GitHub issues automatically when drift is detected
   - Issues include:
     - List of drifted features
     - Drift magnitude
     - Recommended actions

### Manual Monitoring

You can manually check for drift using:

```bash
# Normal drift check
python scripts/drift_detection.py

# Simulate drift (for testing)
python scripts/drift_detection.py --simulate-drift
```

View detailed drift reports in MLflow UI:
1. Start MLflow server: `mlflow server --host 127.0.0.1 --port 5001`
2. Open `http://127.0.0.1:5001` in your browser
3. Navigate to the latest run to view drift metrics

## Model Retraining and Promotion

### Automated Retraining Process

When drift is detected:
1. Review the GitHub issue created by the drift detection workflow
2. Analyze the drift report in MLflow
3. If retraining is needed:
   - Create a new branch for model updates
   - Run retraining script
   - Review model performance metrics
   - Create a PR with the "model-promotion" label

### Model Promotion

Models are promoted to production through a controlled process:

1. **Via Pull Request**:
   - Create PR with "model-promotion" label
   - Include run ID in PR description: `Promoting model run: <run_id>`
   - Get PR approved
   - Automated promotion occurs on PR merge

2. **Manual Promotion**:
   ```bash
   # Promote latest staging model
   python scripts/promote_model.py

   # Promote specific version
   python scripts/promote_model.py --model-version <version>

   # Promote by run ID
   python scripts/promote_model.py --run-id <run_id>
   ```

### Monitoring MLflow

Access the MLflow UI to:
- View model versions and their stages
- Compare model metrics
- Analyze drift reports
- Track model lineage

URL: `http://127.0.0.1:5001`

## Best Practices

1. **Regular Monitoring**:
   - Review drift detection results daily
   - Analyze trends in feature drift
   - Document significant changes

2. **Model Updates**:
   - Keep reference data up to date
   - Document all model promotions
   - Maintain test coverage for new models

3. **Workflow Management**:
   - Use PR templates for model updates
   - Include performance metrics in PRs
   - Follow code review guidelines

## Troubleshooting

Common issues and solutions:

1. **Drift Detection Issues**:
   - Verify MLflow server is running
   - Check reference dataset availability
   - Review drift thresholds in configuration

2. **Promotion Failures**:
   - Ensure correct run ID/version
   - Verify MLflow connection
   - Check GitHub Actions logs

3. **Environment Setup**:
   - Confirm environment variables
   - Verify dependencies installation
   - Check MLflow server status

### Baseline Profiles
- Reference data profiles are stored in MLflow as artifacts
- Statistical profiles include:
  - Feature distributions
  - Missing value patterns
  - Category distributions
- Access baseline profiles via MLflow UI under artifacts

### Retraining Triggers

Automated retraining is triggered when:
1. **Feature Drift Thresholds**:
   - More than 30% of features show drift
   - Any critical feature (Age, MonthlyIncome, YearsAtCompany) shows significant drift (p-value < 0.01)

2. **Scheduled Triggers**:
   - Daily drift detection via GitHub Actions
   - Monthly scheduled retraining (fallback mechanism)

3. **Manual Triggers**:
   - Through GitHub Actions workflow dispatch
   - Via local script execution

## Model Governance

### Retraining Decision Process

1. **Automated Analysis**:
   - Drift detection runs daily
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