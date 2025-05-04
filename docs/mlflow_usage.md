# MLflow Usage Guide

This document explains how to use MLflow in the Employee Attrition MLOps project, including interpreting the UI, understanding artifacts, and managing the model registry.

## MLflow UI Overview

### Accessing the UI

1. **Start MLflow Server**
   ```bash
   poetry run mlflow ui --host 0.0.0.0 --port 5001
   ```

2. **Access the UI**
   - Open http://localhost:5001 in your browser
   - Default view shows experiment list

### Key UI Components

1. **Experiments**
   - List of all experiments
   - Filter by name, tags, or time
   - Create new experiments
   - Compare runs across experiments

2. **Runs**
   - Individual training runs
   - Parameters and metrics
   - Artifacts and models
   - Run status and duration

3. **Model Registry**
   - Registered models
   - Model versions
   - Stage transitions
   - Model descriptions

## Interpreting MLflow Artifacts

### Training Artifacts

1. **Model Files**
   - `model.pkl`: Serialized model
   - `model_metadata.json`: Model configuration
   - `requirements.txt`: Dependencies

2. **Metrics**
   - `metrics.json`: Training metrics
   - `validation_metrics.json`: Validation results
   - `test_metrics.json`: Test performance

3. **Plots**
   - `confusion_matrix.png`: Classification performance
   - `roc_curve.png`: ROC analysis
   - `feature_importance.png`: SHAP values
   - `fairness_metrics.png`: Fairness analysis

4. **Reports**
   - `data_drift_report.html`: Drift analysis
   - `model_card.md`: Model documentation
   - `fairness_report.html`: Fairness assessment

### Monitoring Artifacts

1. **Drift Detection**
   - `drift_metrics.json`: Drift scores
   - `feature_drift.png`: Feature distribution changes
   - `prediction_drift.png`: Prediction distribution changes

2. **Performance Tracking**
   - `performance_metrics.json`: Daily metrics
   - `error_analysis.html`: Error patterns
   - `latency_metrics.json`: API performance

## Model Registry Workflow

### Model Stages

1. **None**
   - Initial state after training
   - Not ready for deployment

2. **Staging**
   - Candidate for production
   - Passed validation tests
   - Ready for final review

3. **Production**
   - Currently deployed model
   - Serving predictions
   - Monitored for performance

### Stage Transitions

1. **To Staging**
   ```bash
   # Promote model to staging
   poetry run mlflow models transition-stage --model-name employee_attrition_model --version 1 --stage Staging
   ```

2. **To Production**
   ```bash
   # Promote model to production
   poetry run mlflow models transition-stage --model-name employee_attrition_model --version 1 --stage Production
   ```

3. **Archiving**
   ```bash
   # Archive old model
   poetry run mlflow models transition-stage --model-name employee_attrition_model --version 1 --stage Archived
   ```

### Quality Gates

1. **Staging Requirements**
   - Pass all validation tests
   - Meet performance thresholds
   - Complete fairness assessment
   - Documentation up to date

2. **Production Requirements**
   - Pass staging tests
   - Complete A/B testing
   - Performance monitoring setup
   - Rollback plan in place

## Using MLflow in Development

### Tracking Experiments

1. **Start Run**
   ```python
   import mlflow
   
   with mlflow.start_run():
       # Log parameters
       mlflow.log_param("param1", value1)
       
       # Log metrics
       mlflow.log_metric("metric1", value1)
       
       # Log artifacts
       mlflow.log_artifact("path/to/artifact")
   ```

2. **Log Model**
   ```python
   # Log scikit-learn model
   mlflow.sklearn.log_model(
       sk_model=model,
       artifact_path="model",
       registered_model_name="employee_attrition_model"
   )
   ```

### Comparing Runs

1. **Select Runs**
   - Use the UI to select multiple runs
   - Compare parameters and metrics
   - View differences in artifacts

2. **Generate Reports**
   ```bash
   # Generate comparison report
   poetry run python scripts/generate_comparison_report.py --run-ids run1,run2
   ```

## Best Practices

### Experiment Organization

1. **Naming Conventions**
   - Use descriptive experiment names
   - Include date and purpose
   - Tag related experiments

2. **Parameter Logging**
   - Log all relevant parameters
   - Include environment details
   - Document parameter choices

3. **Artifact Management**
   - Keep artifacts organized
   - Use consistent naming
   - Include documentation

### Model Registry

1. **Version Control**
   - Use semantic versioning
   - Document changes
   - Maintain changelog

2. **Stage Management**
   - Follow promotion process
   - Document decisions
   - Track deployments

3. **Monitoring**
   - Track model health
   - Monitor performance
   - Document issues

## Troubleshooting

### Common Issues

1. **Connection Problems**
   ```bash
   # Check tracking URI
   echo $MLFLOW_TRACKING_URI
   
   # Test connection
   poetry run python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

2. **Artifact Storage**
   ```bash
   # Check artifact location
   echo $MLFLOW_ARTIFACT_LOCATION
   
   # Verify permissions
   ls -la /path/to/artifacts
   ```

3. **Model Registry**
   ```bash
   # List registered models
   poetry run mlflow models list
   
   # Check model details
   poetry run mlflow models describe --name employee_attrition_model
   ```

### Performance Optimization

1. **Artifact Storage**
   - Use efficient storage backend
   - Clean up old artifacts
   - Compress large files

2. **Database Optimization**
   - Index frequently queried fields
   - Archive old runs
   - Monitor database size

3. **UI Performance**
   - Limit number of runs loaded
   - Use efficient queries
   - Cache frequently accessed data

## Reviewing Staging Models

### Accessing the UI

1. **Start MLflow Server**
   ```bash
   poetry run mlflow ui --host 0.0.0.0 --port 5001
   ```

2. **Navigate to Model Registry**
   - Open http://localhost:5001
   - Click "Models" in the top navigation
   - Select "employee_attrition_model"
   - Filter by "Staging" stage

### Key Artifacts to Review

1. **ROC Curve**
   - Location: `artifacts/roc_curve.png`
   - Purpose: Shows model's discrimination ability
   - Interpretation:
     - Higher AUC = better discrimination
     - Should be > 0.8 for good performance
     - Check for smooth curve without sharp drops

2. **Data Profile**
   - Location: `artifacts/ydata-profile.html`
   - Purpose: Shows data distribution and quality
   - Key Checks:
     - Feature distributions
     - Missing values
     - Outliers
     - Data types
     - Correlations

3. **Prediction Histogram**
   - Location: `artifacts/prediction_histogram.png`
   - Purpose: Shows prediction distribution
   - Interpretation:
     - Should match expected attrition rate
     - Check for prediction bias
     - Look for unusual patterns

4. **Fairness Report**
   - Location: `artifacts/fairness_report.html`
   - Purpose: Shows model fairness across groups
   - Key Metrics:
     - Statistical parity
     - Equal opportunity
     - Predictive parity
     - Group-wise performance

5. **SHAP Plots**
   - Location: `artifacts/shap_summary.png`
   - Purpose: Shows feature importance
   - Interpretation:
     - Global feature importance
     - Feature interactions
     - Direction of impact
     - Magnitude of effects

### Model Metrics

1. **Performance Metrics**
   - Accuracy: Should be > 0.85
   - F1 Score: Should be > 0.80
   - ROC AUC: Should be > 0.85
   - Precision: Should be > 0.80
   - Recall: Should be > 0.75

2. **Fairness Metrics**
   - Statistical Parity Difference: < 0.1
   - Equal Opportunity Difference: < 0.1
   - Predictive Parity Difference: < 0.1

3. **Data Quality Metrics**
   - Missing Value Rate: < 0.05
   - Outlier Rate: < 0.01
   - Feature Correlation: < 0.8

### Review Process

1. **Initial Check**
   - Verify model is in "Staging" stage
   - Check model version and timestamp
   - Review commit message and author

2. **Performance Review**
   - Compare metrics to thresholds
   - Check for performance degradation
   - Review error patterns

3. **Fairness Review**
   - Check fairness metrics
   - Review group-wise performance
   - Verify bias mitigation

4. **Data Quality Review**
   - Check data profile
   - Review feature distributions
   - Verify preprocessing

5. **Documentation Review**
   - Check model card
   - Review training parameters
   - Verify dependencies

### Decision Making

1. **Approval Criteria**
   - All metrics meet thresholds
   - No significant fairness issues
   - Good data quality
   - Complete documentation

2. **Promotion Process**
   ```bash
   # Promote to production
   poetry run mlflow models transition-stage \
       --model-name employee_attrition_model \
       --version 1 \
       --stage Production
   ```

3. **Rejection Process**
   - Document reasons for rejection
   - Create new experiment
   - Address identified issues

### Best Practices

1. **Review Process**
   - Follow checklist systematically
   - Document all findings
   - Consider business impact
   - Consult stakeholders

2. **Documentation**
   - Update model card
   - Document review process
   - Record decisions
   - Track changes

3. **Monitoring**
   - Set up alerts
   - Track performance
   - Monitor fairness
   - Watch for drift 