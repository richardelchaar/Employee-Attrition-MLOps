# Monitoring Strategy Guide

This document details the monitoring and drift detection strategy for the Employee Attrition MLOps project, including baseline generation, drift detection methods, and retraining triggers.

## Drift Detection Overview

### Types of Drift

1. **Feature Drift**
   - Changes in feature distributions
   - Statistical tests for each feature
   - Population stability index (PSI)
   - Kolmogorov-Smirnov test

2. **Prediction Drift**
   - Changes in prediction distributions
   - Target distribution shifts
   - Confidence score changes
   - Model performance degradation

3. **Concept Drift**
   - Changes in feature-target relationship
   - Performance metric changes
   - Error pattern analysis
   - Feature importance shifts

## Baseline Generation

### Reference Data

1. **Initial Baseline**
   ```python
   # Generate reference data
   poetry run python scripts/create_drift_reference.py \
       --input data/processed/training.csv \
       --output data/reference/baseline.csv
   ```

2. **Baseline Statistics**
   ```python
   # Calculate baseline statistics
   from evidently import ColumnMapping
   from evidently.test_suite import TestSuite
   from evidently.tests import *
   
   column_mapping = ColumnMapping(
       target='attrition',
       numerical_features=['age', 'salary'],
       categorical_features=['department', 'education']
   )
   
   reference_data = pd.read_csv('data/reference/baseline.csv')
   ```

### Statistical Tests

1. **Feature Distribution**
   ```python
   # Kolmogorov-Smirnov test
   from scipy.stats import ks_2samp
   
   def check_feature_drift(reference, current, feature):
       statistic, pvalue = ks_2samp(
           reference[feature],
           current[feature]
       )
       return pvalue < 0.05
   ```

2. **Population Stability**
   ```python
   # Population Stability Index
   def calculate_psi(expected, actual, buckets=10):
       expected_percents = np.histogram(expected, buckets)[0] / len(expected)
       actual_percents = np.histogram(actual, buckets)[0] / len(actual)
       return np.sum(
           (actual_percents - expected_percents) * 
           np.log(actual_percents / expected_percents)
       )
   ```

## Drift Detection Implementation

### Monitoring Pipeline

1. **Data Collection**
   ```python
   # Collect current data
   current_data = pd.read_sql(
       "SELECT * FROM predictions WHERE date >= DATEADD(day, -7, GETDATE())",
       con=db_connection
   )
   ```

2. **Drift Detection**
   ```python
   # Run drift detection
   poetry run python scripts/check_production_drift.py \
       --reference data/reference/baseline.csv \
       --current data/current/predictions.csv
   ```

### Statistical Analysis

1. **Feature Drift**
   ```python
   # Feature drift detection
   from evidently.analyzers import DataDriftAnalyzer
   
   data_drift_analyzer = DataDriftAnalyzer()
   data_drift_analyzer.calculate(
       reference_data,
       current_data,
       column_mapping
   )
   ```

2. **Prediction Drift**
   ```python
   # Prediction drift detection
   from evidently.analyzers import TargetDriftAnalyzer
   
   target_drift_analyzer = TargetDriftAnalyzer()
   target_drift_analyzer.calculate(
       reference_data,
       current_data,
       column_mapping
   )
   ```

## Thresholds and Alerts

### Drift Thresholds

1. **Feature Drift**
   - PSI > 0.25: Significant drift
   - KS p-value < 0.05: Statistical significance
   - Distribution difference > 10%: Notable change

2. **Prediction Drift**
   - Accuracy drop > 5%: Performance degradation
   - F1 score drop > 5%: Classification issues
   - ROC AUC drop > 0.05: Model quality decline

### Alert System

1. **Alert Generation**
   ```python
   def generate_drift_alert(drift_metrics):
       alerts = []
       for metric, value in drift_metrics.items():
           if value > THRESHOLDS[metric]:
               alerts.append(f"{metric} drift detected: {value}")
       return alerts
   ```

2. **Notification System**
   ```python
   def send_drift_notification(alerts):
       if alerts:
           message = "\n".join(alerts)
           # Send to monitoring system
           # Log to MLflow
           # Notify team
   ```

## Retraining Triggers

### Automatic Triggers

1. **Performance Degradation**
   - Accuracy below threshold
   - F1 score below threshold
   - ROC AUC below threshold

2. **Significant Drift**
   - Multiple features show drift
   - Prediction distribution changes
   - Concept drift detected

### Manual Triggers

1. **Scheduled Retraining**
   - Monthly model refresh
   - Quarterly full retraining
   - Annual model review

2. **Business Changes**
   - New data sources
   - Policy changes
   - Process updates

## Monitoring Reports

### Report Generation

1. **Daily Reports**
   ```python
   # Generate daily drift report
   poetry run python scripts/generate_drift_report.py \
       --current-data data/current/predictions.csv \
       --output reports/drift/daily_report.html
   ```

2. **Weekly Summary**
   ```python
   # Generate weekly summary
   poetry run python scripts/generate_summary_report.py \
       --start-date $(date -d "7 days ago" +%Y-%m-%d) \
       --output reports/summary/weekly_report.html
   ```

### Report Interpretation

1. **Drift Metrics**
   - Feature drift scores
   - Prediction drift metrics
   - Performance changes

2. **Action Items**
   - Features requiring attention
   - Model performance issues
   - Retraining recommendations

## Best Practices

### Monitoring Setup

1. **Data Quality**
   - Validate input data
   - Handle missing values
   - Check data types

2. **Performance Tracking**
   - Regular metric collection
   - Historical comparison
   - Trend analysis

### Maintenance

1. **Baseline Updates**
   - Regular baseline refresh
   - Historical baseline storage
   - Version control

2. **Threshold Tuning**
   - Regular threshold review
   - Business impact consideration
   - False positive management

## Troubleshooting

### Common Issues

1. **False Positives**
   - Check threshold settings
   - Verify data quality
   - Review statistical tests

2. **Missing Data**
   - Handle missing values
   - Update data collection
   - Modify monitoring logic

3. **Performance Issues**
   - Optimize calculations
   - Cache results
   - Schedule efficiently

### Solutions

1. **Data Issues**
   - Improve data quality
   - Update validation rules
   - Enhance error handling

2. **Monitoring Issues**
   - Adjust thresholds
   - Update baselines
   - Modify alert rules

3. **Performance Issues**
   - Optimize code
   - Use caching
   - Schedule jobs 