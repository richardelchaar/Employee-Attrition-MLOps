# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError
import logging
from src.employee_attrition_mlops.data_processing import (
    BoxCoxSkewedTransformer,
    AddNewFeaturesTransformer,
    CustomOrdinalEncoder,
    LogTransformSkewed,
    load_and_clean_data_from_db,
    identify_column_types,
    find_skewed_columns
)
from src.employee_attrition_mlops.pipelines import create_preprocessing_pipeline

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Constants for testing
BUSINESS_TRAVEL_MAPPING = {
    'Non-Travel': 0, 
    'Travel_Rarely': 1, 
    'Travel_Frequently': 2
}

# Mock data for testing
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'YearsAtCompany': [1, 2, 3, 4, 5],
        'TotalWorkingYears': [2, 4, 6, 8, 10],
        'MonthlyIncome': [2000, 3000, 4000, 5000, 6000],
        'Attrition': ['No', 'Yes', 'No', 'No', 'Yes'],
        'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely'],
        'Education': [1, 2, 3, 4, 5],
        'EnvironmentSatisfaction': [3, 2, 4, 1, 2],
        'EmployeeCount': [1, 1, 1, 1, 1],
        'StandardHours': [80, 80, 80, 80, 80],
        'Over18': ['Y', 'Y', 'Y', 'Y', 'Y'],
        'EmployeeNumber': [1, 2, 3, 4, 5],
        'HighlySkewedCol': [0, 1, 1, 1, 1000]  # Intentionally skewed
    })

@pytest.fixture
def skewed_data():
    return pd.DataFrame({
        'NormalCol': np.random.normal(0, 1, 100),
        'SkewedCol': np.random.exponential(1, 100),
        'NegativeSkewCol': -np.random.exponential(1, 100)
    })

# Test BoxCoxSkewedTransformer
def test_boxcox_transformer(sample_data):
    transformer = BoxCoxSkewedTransformer(skewed_cols=['HighlySkewedCol'])
    
    # Test fit
    transformer.fit(sample_data)
    assert 'HighlySkewedCol' in transformer.lambdas_
    assert transformer.lambdas_['HighlySkewedCol'] is not None
    
    # Test transform
    transformed = transformer.transform(sample_data)
    assert 'HighlySkewedCol' in transformed.columns
    assert not transformed.isnull().values.any()

# Test AddNewFeaturesTransformer
def test_add_new_features_transformer(sample_data):
    transformer = AddNewFeaturesTransformer()
    
    # Test transform
    transformed = transformer.fit_transform(sample_data)
    assert 'AgeAtJoining' in transformed.columns
    assert 'TenureRatio' in transformed.columns
    assert 'IncomePerYearExp' in transformed.columns
    
    # Verify calculations
    assert transformed.loc[0, 'AgeAtJoining'] == 24  # 25 - 1
    assert transformed.loc[1, 'TenureRatio'] == 0.5  # 2 / 4
    assert transformed.loc[2, 'IncomePerYearExp'] == pytest.approx(666.666, 0.1)  # 4000 / 6

# Test CustomOrdinalEncoder - unit test
def test_custom_ordinal_encoder(sample_data):
    mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    encoder = CustomOrdinalEncoder(mapping=mapping, cols=['BusinessTravel'])
    
    # Test transform
    transformed = encoder.fit_transform(sample_data)
    assert transformed['BusinessTravel'].isin([0, 1, 2]).all()

# Integration Test to ensure business travel encoding is right
def test_business_travel_in_pipeline(sample_data):
    """Test the actual pipeline uses BUSINESS_TRAVEL_MAPPING correctly"""
    col_types = identify_column_types(sample_data, 'Attrition')
    
    preprocessor = create_preprocessing_pipeline(
        numerical_cols=col_types['numerical'],
        categorical_cols=col_types['categorical'],
        ordinal_cols=col_types['ordinal'],
        business_travel_col=col_types['business_travel'],
        skewed_cols=[],
        business_encoder_type='ordinal'  # Force ordinal encoding
    )
    
    transformed = preprocessor.fit_transform(sample_data)
    
    # Verify mapping matches config
    if isinstance(transformed, pd.DataFrame):
        expected_values = sample_data['BusinessTravel'].map(BUSINESS_TRAVEL_MAPPING)
        pd.testing.assert_series_equal(
            transformed['BusinessTravel'],
            expected_values,
            check_names=False
        )

def test_invalid_business_travel():
    """Test handling of invalid business travel categories"""
    bad_data = pd.DataFrame({
        'BusinessTravel': ['Invalid', 'Non-Travel', 'Unknown'],
        'MonthlyIncome': [1000, 2000, 3000],
        'Attrition': ['No', 'Yes', 'No']  # Required for identify_column_types
    })
    
    # Get column types normally
    col_types = identify_column_types(bad_data, 'Attrition')
    
    # Should complete without errors (graceful handling)
    preprocessor = create_preprocessing_pipeline(
        numerical_cols=col_types['numerical'],
        categorical_cols=col_types['categorical'],
        ordinal_cols=col_types['ordinal'],
        business_travel_col=col_types['business_travel'],
        skewed_cols=[],
        business_encoder_type='ordinal'
    )
    
    transformed = preprocessor.fit_transform(bad_data)
    # Verify unknown categories were mapped to -1 or similar
    if isinstance(transformed, pd.DataFrame):
        assert (transformed['BusinessTravel'] == -1).any()

# Test LogTransformSkewed
def test_log_transform_skewed(sample_data):
    transformer = LogTransformSkewed(skewed_cols=['MonthlyIncome'])
    transformed = transformer.fit_transform(sample_data)
    assert (transformed['MonthlyIncome'] > 0).all()

# Test identify_column_types
def test_identify_column_types(sample_data):
    col_types = identify_column_types(sample_data, target_column='Attrition')
    
    assert 'numerical' in col_types
    assert 'categorical' in col_types
    assert 'ordinal' in col_types
    assert 'MonthlyIncome' in col_types['numerical']
    assert 'BusinessTravel' in col_types['business_travel']
    assert 'Education' in col_types['ordinal']

# Test find_skewed_columns
def test_find_skewed_columns(skewed_data):
    skewed_cols = find_skewed_columns(skewed_data, num_cols=['NormalCol', 'SkewedCol', 'NegativeSkewCol'])
    assert 'SkewedCol' in skewed_cols
    assert 'NegativeSkewCol' in skewed_cols
    assert 'NormalCol' not in skewed_cols

# Test database loading with mock
@patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_db_url")
@patch('src.employee_attrition_mlops.data_processing.create_engine')
def test_load_and_clean_data_from_db(mock_engine, sample_data):
    # Setup mock
    mock_conn = MagicMock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = ['employees_history']
    
    # Mock read_sql_table to return our sample data
    with patch('pandas.read_sql_table', return_value=sample_data):
        result = load_and_clean_data_from_db()
    
    assert result is not None
    assert 'EmployeeCount' not in result.columns  # Should be dropped
    assert result['Attrition'].dtype == float  # Should be converted to numeric

def test_load_and_clean_data_from_db_failure():
    """Test failure when table doesn't exist"""
    with patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_db_url"), \
         patch('sqlalchemy.create_engine') as mock_engine:
        
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None  # Simulate missing table
        
        result = load_and_clean_data_from_db()
        assert result is None

@patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_url")
@patch('sqlalchemy.create_engine')
def test_db_connection_failure(mock_engine):
    """Test database connection failure"""
    mock_engine.side_effect = SQLAlchemyError("Connection failed")
    result = load_and_clean_data_from_db()
    assert result is None

@patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_url")
@patch('pandas.read_sql_table')
def test_empty_table(mock_read_sql, sample_data):
    '''Tests empty table handling'''
    mock_read_sql.return_value = pd.DataFrame()  # Empty DataFrame
    result = load_and_clean_data_from_db()
    assert result is None  # Or assert result.empty if you handle empty tables differently

@patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_url")
def test_missing_target_column(sample_data):
    """Test handling of missing target column"""
    corrupted_data = sample_data.drop(columns=['Attrition'])
    with patch('pandas.read_sql_table', return_value=corrupted_data):
        result = load_and_clean_data_from_db()
    assert result is None  # Or check for a specific warning/error

@patch('src.employee_attrition_mlops.data_processing.DATABASE_URL_PYODBC', "mock_url")
@patch('src.employee_attrition_mlops.data_processing.create_engine')
def test_missing_columns_to_drop(mock_engine, sample_data):
    # Setup mock connection
    mock_conn = MagicMock()
    mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = ['employees_history']
    
    # Create test data missing one of the drop columns
    test_data = sample_data.drop(columns=['EmployeeCount'])
    
    # Mock read_sql_table
    with patch('pandas.read_sql_table', return_value=test_data):
        result = load_and_clean_data_from_db()
    
    assert result is not None
    assert 'EmployeeCount' not in result.columns
    assert 'StandardHours' not in result.columns

# Test full pipeline
def test_create_preprocessing_pipeline(sample_data):
    # First identify column types
    col_types = identify_column_types(sample_data, target_column='Attrition')
    
    # Find skewed columns - suppress warnings during this operation
    with pytest.warns(RuntimeWarning, match="Precision loss occurred in moment calculation"):
        skewed_cols = find_skewed_columns(
            sample_data, 
            num_cols=col_types['numerical'],
            threshold=0.75
        )

    skewed_cols = find_skewed_columns(
        sample_data, 
        num_cols=col_types['numerical'],
        threshold=0.75
    )
    
    # Create the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_cols=col_types['numerical'],
        categorical_cols=col_types['categorical'],
        ordinal_cols=col_types['ordinal'],
        business_travel_col=col_types['business_travel'],
        skewed_cols=skewed_cols,
        numeric_transformer_type='boxcox',
        numeric_scaler_type='standard',
        business_encoder_type='ordinal'
    )
    
    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(sample_data)
    
    # Verify the output structure
    assert isinstance(transformed_data, (pd.DataFrame, np.ndarray))
    if isinstance(transformed_data, pd.DataFrame):
        # Check some expected columns
        assert any('BusinessTravel' in col for col in transformed_data.columns)
        assert any('MonthlyIncome' in col for col in transformed_data.columns)
        
        # Verify BusinessTravel was encoded correctly
        if 'BusinessTravel' in transformed_data.columns:
            assert transformed_data['BusinessTravel'].isin([0, 1, 2]).all()
    else:
        # For numpy array output, check shape
        assert transformed_data.shape[0] == len(sample_data)
        assert transformed_data.shape[1] > 0
    
    # Test different configurations
    for transform_type in ['log', 'boxcox', None]:
        for scaler_type in ['standard', 'minmax', None]:
            for encoder_type in ['ordinal', 'onehot']:
                try:
                    preprocessor = create_preprocessing_pipeline(
                        numerical_cols=col_types['numerical'],
                        categorical_cols=col_types['categorical'],
                        ordinal_cols=col_types['ordinal'],
                        business_travel_col=col_types['business_travel'],
                        skewed_cols=skewed_cols,
                        numeric_transformer_type=transform_type,
                        numeric_scaler_type=scaler_type,
                        business_encoder_type=encoder_type
                    )
                    # Just test that it runs with these parameters
                    preprocessor.fit(sample_data)
                except Exception as e:
                    pytest.fail(f"Failed with transform_type={transform_type}, "
                              f"scaler_type={scaler_type}, encoder_type={encoder_type}: {str(e)}")
