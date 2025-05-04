# Monitoring System

This directory contains the monitoring and drift detection system for the Employee Attrition prediction model. The system ensures model reliability by detecting and alerting on data and prediction drift.

## Components

### Drift Detection
- Feature drift monitoring
- Prediction drift detection
- Statistical tests implementation
- Alert generation

### Monitoring Dashboard
- Real-time metrics display
- Drift visualization
- Performance tracking
- Alert management

### Alert System
- Threshold configuration
- Alert generation
- Notification delivery
- Alert history

## Drift Detection Methods

### Feature Drift
- Kolmogorov-Smirnov test for numerical features
- Chi-square test for categorical features
- Population stability index (PSI)
- Feature distribution comparison

### Prediction Drift
- Prediction distribution comparison
- Performance metric monitoring
- Confidence score analysis
- Error rate tracking

## Implementation

### Statistical Tests
```python
from monitoring.drift_detection import detect_feature_drift

# Detect drift in a feature
drift_result = detect_feature_drift(
    reference_data=ref_data,
    current_data=curr_data,
    feature_name="age",
    test_type="ks"
)
```

### Alert Generation
```python
from monitoring.alerts import generate_alert

# Generate drift alert
alert = generate_alert(
    drift_type="feature",
    feature_name="age",
    drift_score=0.85,
    threshold=0.7
)
```

## Configuration

### Drift Thresholds
```yaml
drift_thresholds:
  feature_drift:
    ks_test: 0.7
    chi_square: 0.7
    psi: 0.2
  prediction_drift:
    performance: 0.1
    distribution: 0.7
```

### Alert Settings
```yaml
alerts:
  email:
    enabled: true
    recipients: ["team@example.com"]
  slack:
    enabled: true
    channel: "#model-monitoring"
```

## Usage

### Running Drift Detection
```bash
python -m monitoring.drift_detection \
    --reference-data path/to/reference.csv \
    --current-data path/to/current.csv \
    --output-dir reports/
```

### Viewing Monitoring Dashboard
```bash
streamlit run monitoring/dashboard.py
```

## Monitoring Metrics

### Feature Drift Metrics
- Kolmogorov-Smirnov statistic
- Chi-square statistic
- Population Stability Index
- Feature distribution changes

### Prediction Drift Metrics
- Prediction distribution changes
- Performance metric changes
- Error rate changes
- Confidence score changes

## Alert Types

1. **Feature Drift Alerts**
   - Significant distribution changes
   - Missing value increases
   - Outlier increases
   - Feature correlation changes

2. **Prediction Drift Alerts**
   - Performance degradation
   - Prediction distribution shifts
   - Error rate increases
   - Confidence score changes

## Best Practices

1. **Threshold Setting**
   - Set appropriate thresholds for different features
   - Consider business impact
   - Account for seasonality
   - Regular threshold review

2. **Alert Management**
   - Prioritize alerts by impact
   - Group related alerts
   - Maintain alert history
   - Regular alert review

3. **Monitoring Strategy**
   - Regular drift checks
   - Comprehensive metrics
   - Clear visualization
   - Actionable insights

4. **Documentation**
   - Document drift detection methods
   - Explain alert logic
   - Maintain runbook
   - Update thresholds

## Testing

Run the monitoring test suite:
```bash
pytest tests/test_monitoring.py
```

## Maintenance

Regular maintenance tasks:
- Update reference data
- Review thresholds
- Clean up old alerts
- Update documentation 