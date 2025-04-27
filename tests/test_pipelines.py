# tests/test_pipelines.py
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import warnings

from src.employee_attrition_mlops.pipelines import create_full_pipeline, create_preprocessing_pipeline
from src.employee_attrition_mlops.data_processing import (
    identify_column_types, find_skewed_columns,
    AddNewFeaturesTransformer, AgeGroupTransformer,
    BoxCoxSkewedTransformer, LogTransformSkewed,
    CustomOrdinalEncoder
)
from src.employee_attrition_mlops.config import TARGET_COLUMN, RANDOM_STATE, SKEWNESS_THRESHOLD, BUSINESS_TRAVEL_MAPPING

# --- Fixtures ---
@pytest.fixture
def sample_data():
    """Basic valid sample data for testing."""
    return pd.DataFrame({
        'EmployeeNumber': [98765, 98766, 98767, 98768, 98769, 98770, 98771, 98772, 98773, 98774],
        'Age': [41, 35, 28, 45, 31, 38, 42, 29, 33, 39],
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'MaritalStatus': ['Single', 'Married', 'Single', 'Divorced', 'Married', 'Single', 'Married', 'Single', 'Divorced', 'Married'],
        'Department': ['Research & Development', 'Sales', 'Research & Development', 'Human Resources', 'Research & Development',
                      'Sales', 'Research & Development', 'Human Resources', 'Sales', 'Research & Development'],
        'EducationField': ['Medical', 'Life Sciences', 'Technical', 'Medical', 'Life Sciences',
                          'Technical', 'Medical', 'Life Sciences', 'Technical', 'Medical'],
        'JobLevel': [3, 2, 1, 4, 2, 3, 4, 1, 2, 3],
        'JobRole': ['Research Scientist', 'Sales Executive', 'Research Scientist', 'Manager', 'Laboratory Technician',
                   'Sales Executive', 'Manager', 'Research Scientist', 'Sales Executive', 'Research Scientist'],
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently',
                          'Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
        'DistanceFromHome': [5, 8, 2, 10, 4, 7, 3, 6, 9, 1],
        'Education': [4, 3, 2, 5, 3, 4, 5, 2, 3, 4],
        'DailyRate': [1102, 950, 800, 1200, 900, 1050, 1150, 850, 1000, 1100],
        'HourlyRate': [94, 85, 75, 100, 80, 90, 95, 85, 88, 92],
        'MonthlyIncome': [8500, 7200, 3500, 9500, 4200, 7800, 9200, 3800, 6500, 8800],
        'MonthlyRate': [21410, 20000, 18000, 22000, 19000, 21000, 22500, 18500, 20500, 21500],
        'PercentSalaryHike': [12, 11, 13, 15, 10, 12, 14, 11, 13, 12],
        'StockOptionLevel': [0, 1, 0, 2, 1, 0, 2, 1, 0, 1],
        'OverTime': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'NumCompaniesWorked': [1, 2, 1, 3, 1, 2, 3, 1, 2, 1],
        'TotalWorkingYears': [15, 8, 5, 20, 7, 12, 18, 6, 10, 14],
        'TrainingTimesLastYear': [3, 2, 4, 1, 3, 2, 1, 4, 3, 2],
        'YearsAtCompany': [8, 6, 3, 12, 4, 7, 10, 2, 5, 9],
        'YearsInCurrentRole': [7, 4, 2, 10, 3, 6, 9, 1, 4, 8],
        'YearsSinceLastPromotion': [1, 2, 1, 3, 1, 2, 3, 1, 2, 1],
        'YearsWithCurrManager': [7, 5, 2, 10, 3, 6, 9, 1, 4, 8],
        'EnvironmentSatisfaction': [3, 2, 4, 1, 3, 2, 4, 1, 3, 2],
        'JobInvolvement': [3, 2, 4, 1, 3, 2, 4, 1, 3, 2],
        'JobSatisfaction': [2, 1, 4, 1, 3, 2, 4, 1, 3, 2],
        'PerformanceRating': [3, 3, 4, 3, 4, 3, 4, 3, 3, 4],
        'RelationshipSatisfaction': [4, 2, 4, 1, 3, 2, 4, 1, 3, 2],
        'WorkLifeBalance': [2, 3, 4, 1, 3, 2, 4, 1, 3, 2],
        TARGET_COLUMN: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Balanced binary target
    })

@pytest.fixture
def large_sample_data(sample_data):
    """Create a larger dataset for SMOTE testing."""
    # Create a larger dataset by duplicating the sample data
    data = pd.concat([sample_data] * 50)  # Create 50 copies for more samples
    return data.reset_index(drop=True)

@pytest.fixture
def imbalanced_data(large_sample_data):
    """Create imbalanced data for SMOTE testing."""
    data = large_sample_data.copy()
    # Make the target imbalanced (95% zeros, 5% ones)
    n_samples = len(data)
    n_ones = int(n_samples * 0.05)  # 5% ones
    data.loc[data.index[n_ones:], TARGET_COLUMN] = 0
    return data

@pytest.fixture
def preprocessor(sample_data):
    """Create a preprocessor for testing."""
    col_types = identify_column_types(sample_data, target_column=TARGET_COLUMN)
    numerical_cols = col_types['numerical']
    categorical_cols = col_types['categorical']
    ordinal_cols = col_types['ordinal']
    business_travel_col = col_types['business_travel']
    
    skewed_cols = find_skewed_columns(sample_data, num_cols=numerical_cols, threshold=SKEWNESS_THRESHOLD)
    
    return create_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        ordinal_cols=ordinal_cols,
        business_travel_col=business_travel_col,
        skewed_cols=skewed_cols,
        numeric_transformer_type='log',
        numeric_scaler_type='standard',
        business_encoder_type='onehot'
    )

# --- Test Cases ---
def test_create_full_pipeline_basic(sample_data, preprocessor):
    """Test basic pipeline creation and functionality."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    pipeline = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False  # Disable SMOTE for basic test
    )
    
    # Test pipeline creation
    assert pipeline is not None
    assert len(pipeline.steps) == 5  # feature_eng, preprocessor, feature_selection, smote, classifier
    
    # Test pipeline fitting
    pipeline.fit(X, y)
    
    # Test pipeline prediction
    y_pred = pipeline.predict(X)
    assert len(y_pred) == len(y)
    assert set(np.unique(y_pred)).issubset({0, 1})
    
    # Test pipeline probabilities
    y_proba = pipeline.predict_proba(X)
    assert y_proba.shape == (len(y), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)

def test_create_full_pipeline_feature_selection(sample_data, preprocessor):
    """Test pipeline with different feature selection methods."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    # Test RFE feature selection
    pipeline_rfe = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='rfe',
        feature_selector_params={'n_features_to_select': 5},
        smote_active=False
    )
    
    # Test Lasso feature selection
    pipeline_lasso = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='lasso',
        feature_selector_params={'C': 0.1},
        smote_active=False
    )
    
    # Test Tree-based feature selection
    pipeline_tree = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='tree',
        feature_selector_params={'threshold': 'median'},
        smote_active=False
    )
    
    # Test all pipelines
    for pipeline in [pipeline_rfe, pipeline_lasso, pipeline_tree]:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        assert len(y_pred) == len(y)
        assert set(np.unique(y_pred)).issubset({0, 1})
        
        # Test feature selection results
        if hasattr(pipeline, 'named_steps'):
            feature_selector = pipeline.named_steps['feature_selection']
            if feature_selector != 'passthrough':
                assert hasattr(feature_selector, 'get_support')
                selected_features = feature_selector.get_support()
                assert isinstance(selected_features, np.ndarray)
                assert selected_features.dtype == bool

@pytest.mark.parametrize("data_fixture", ["large_sample_data", "imbalanced_data"])
def test_create_full_pipeline_smote(data_fixture, preprocessor, request):
    """Test pipeline with and without SMOTE for both balanced and imbalanced data."""
    data = request.getfixturevalue(data_fixture)
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    
    # Test with SMOTE
    pipeline_with_smote = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=True
    )
    
    # Test without SMOTE
    pipeline_without_smote = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Test both pipelines
    for pipeline in [pipeline_with_smote, pipeline_without_smote]:
        # Fit the pipeline
        pipeline.fit(X, y)
        
        # Test predictions
        y_pred = pipeline.predict(X)
        assert len(y_pred) == len(y)
        assert set(np.unique(y_pred)).issubset({0, 1})
        
        # Test SMOTE effect on training data
        if pipeline.named_steps['smote'] != 'passthrough':
            # Get the preprocessed data
            X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
            # Get the SMOTE step from the pipeline
            smote_step = pipeline.named_steps['smote']
            # Apply SMOTE to preprocessed data
            X_resampled, y_resampled = smote_step.fit_resample(X_preprocessed, y)
            
            if data_fixture == "imbalanced_data":
                # For imbalanced data, SMOTE should increase samples
                assert len(X_resampled) > len(X)
                assert len(y_resampled) > len(y)
                # Check that minority class is oversampled
                assert sum(y_resampled == 1) > sum(y == 1)
            else:
                # For balanced data, SMOTE should not be applied
                assert len(X_resampled) == len(X)
                assert len(y_resampled) == len(y)
                # Check that class balance is maintained
                assert sum(y_resampled == 1) == sum(y == 1)

def test_create_full_pipeline_classifiers(sample_data, preprocessor):
    """Test pipeline with different classifier types."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    # Test LogisticRegression
    pipeline_lr = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Test RandomForest
    pipeline_rf = create_full_pipeline(
        classifier_class=RandomForestClassifier,
        model_params={'random_state': RANDOM_STATE, 'n_estimators': 10},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Test both pipelines
    for pipeline in [pipeline_lr, pipeline_rf]:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        assert len(y_pred) == len(y)
        assert set(np.unique(y_pred)).issubset({0, 1})
        
        # Test feature importance if available
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            assert len(importances) > 0
            assert np.all(importances >= 0)
            assert np.allclose(importances.sum(), 1.0)

def test_create_full_pipeline_edge_cases(sample_data, preprocessor):
    """Test pipeline with edge cases."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    # Test with empty feature selector params
    pipeline_empty_params = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        feature_selector_params={},
        smote_active=False
    )
    
    # Test with invalid feature selector type - should default to passthrough
    pipeline_invalid_selector = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='invalid_type',
        smote_active=False
    )
    assert pipeline_invalid_selector.named_steps['feature_selection'] == 'passthrough'
    
    # Test with invalid classifier params
    with pytest.raises(Exception):
        create_full_pipeline(
            classifier_class=LogisticRegression,
            model_params={'invalid_param': 1},
            preprocessor=preprocessor,
            feature_selector_type='passthrough',
            smote_active=False
        )
    
    # Test with empty data
    with pytest.raises(ValueError):
        pipeline_empty_params.fit(pd.DataFrame(), pd.Series())
    
    # Test with single sample
    with pytest.raises(ValueError):
        pipeline_empty_params.fit(X.iloc[:1], y.iloc[:1])
    
    # Test the valid pipeline
    pipeline_empty_params.fit(X, y)
    y_pred = pipeline_empty_params.predict(X)
    assert len(y_pred) == len(y)
    assert set(np.unique(y_pred)).issubset({0, 1})

def test_create_full_pipeline_not_fitted(sample_data, preprocessor):
    """Test pipeline before fitting."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    
    pipeline = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Test that predict raises NotFittedError before fitting
    with pytest.raises(NotFittedError):
        pipeline.predict(X)
    
    # Test that predict_proba raises NotFittedError before fitting
    with pytest.raises(NotFittedError):
        pipeline.predict_proba(X)

def test_create_full_pipeline_feature_engineering(sample_data, preprocessor):
    """Test that feature engineering steps are applied correctly."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    pipeline = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Get the transformed data after feature engineering
    feature_eng = pipeline.named_steps['feature_eng']
    X_transformed = feature_eng.transform(X)
    
    # Check that new features are created
    assert 'AgeAtJoining' in X_transformed.columns
    assert 'TenureRatio' in X_transformed.columns
    assert 'IncomePerYearExp' in X_transformed.columns
    
    # Verify calculations
    expected_age_at_joining = X['Age'] - X['YearsAtCompany']
    assert all(X_transformed['AgeAtJoining'] == expected_age_at_joining)
    
    expected_tenure_ratio = X['YearsAtCompany'] / X['TotalWorkingYears']
    assert all(X_transformed['TenureRatio'] == expected_tenure_ratio)
    
    expected_income_per_year = X['MonthlyIncome'] / X['TotalWorkingYears']
    assert all(X_transformed['IncomePerYearExp'] == expected_income_per_year)
    
    # Check that original columns are preserved
    for col in X.columns:
        assert col in X_transformed.columns

def test_create_full_pipeline_data_transformations(sample_data, preprocessor):
    """Test that data transformations are applied correctly."""
    X = sample_data.drop(columns=[TARGET_COLUMN])
    y = sample_data[TARGET_COLUMN]
    
    pipeline = create_full_pipeline(
        classifier_class=LogisticRegression,
        model_params={'random_state': RANDOM_STATE},
        preprocessor=preprocessor,
        feature_selector_type='passthrough',
        smote_active=False
    )
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Get the transformed data after preprocessing
    X_transformed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Check that categorical variables are one-hot encoded
    categorical_cols = ['Gender', 'MaritalStatus', 'Department', 'EducationField', 'JobRole']
    for col in categorical_cols:
        if col in X.columns:
            # Check that the original column is not present
            assert col not in X_transformed.columns
            # Check that one-hot encoded columns are present
            encoded_cols = [c for c in X_transformed.columns if c.startswith(col)]
            assert len(encoded_cols) > 0
            # Check that encoded columns are binary
            for encoded_col in encoded_cols:
                assert set(X_transformed[encoded_col].unique()).issubset({0, 1})
    
    # Check that BusinessTravel is encoded according to the mapping
    if 'BusinessTravel' in X.columns:
        bt_cols = [c for c in X_transformed.columns if c.startswith('BusinessTravel')]
        assert len(bt_cols) > 0
        for bt_col in bt_cols:
            assert set(X_transformed[bt_col].unique()).issubset({0, 1})
    
    # Check that numerical variables are scaled
    numerical_cols = ['Age', 'DistanceFromHome', 'DailyRate', 'HourlyRate', 'MonthlyIncome']
    for col in numerical_cols:
        if col in X.columns:
            # Check that the column is present
            assert col in X_transformed.columns
            # Check that the values are scaled (mean close to 0, std close to 1)
            # Allow for some numerical precision issues
            assert abs(X_transformed[col].mean()) < 0.1  # Relaxed tolerance
            assert abs(X_transformed[col].std() - 1) < 0.1  # Relaxed tolerance
    
    # Check that ordinal variables are preserved
    ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction']
    for col in ordinal_cols:
        if col in X.columns:
            assert col in X_transformed.columns
            # Check that the values are preserved
            assert set(X_transformed[col].unique()).issubset(set(X[col].unique())) 