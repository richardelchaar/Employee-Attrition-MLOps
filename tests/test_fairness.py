import pytest
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_fairness_metrics():
    # Create more realistic sample data with some bias
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1])  # Introduce some bias
    sensitive_features = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'AgeGroup': ['Young', 'Old', 'Young', 'Old', 'Young', 'Old', 'Young', 'Old']
    })
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Test overall metrics
    assert 'accuracy' in metric_frame.overall
    assert 'precision' in metric_frame.overall
    assert 'recall' in metric_frame.overall
    assert 'f1' in metric_frame.overall
    
    # Test group metrics - check the structure of by_group
    group_metrics = metric_frame.by_group
    assert isinstance(group_metrics, pd.DataFrame)
    assert 'Gender' in group_metrics.index.names
    assert 'AgeGroup' in group_metrics.index.names
    
    # Test demographic parity
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    assert isinstance(dp_diff, float)
    assert 0 <= dp_diff <= 1
    
    # Test equalized odds
    eo_diff = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    assert isinstance(eo_diff, float)
    assert 0 <= eo_diff <= 1

def test_fairness_thresholds():
    # Create data with balanced predictions across groups
    # Each group (M/F) has equal number of positive and negative predictions
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Perfect predictions
    sensitive_features = pd.DataFrame({
        'Gender': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F']  # Balanced groups
    })
    
    # Calculate demographic parity difference
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    
    # Calculate equalized odds difference
    eo_diff = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    
    # Both metrics should be 0 for perfect predictions with balanced groups
    assert abs(dp_diff) < 1e-10  # Using small epsilon for floating point comparison
    assert abs(eo_diff) < 1e-10  # Using small epsilon for floating point comparison

def test_fairness_metrics_with_bias():
    # Test with more pronounced bias to ensure metrics detect it
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])  # More bias
    sensitive_features = pd.DataFrame({
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
    })
    
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    assert dp_diff > 0.1  # Should detect bias above threshold
    
    eo_diff = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features['Gender']
    )
    assert eo_diff > 0.1  # Should detect bias above threshold 