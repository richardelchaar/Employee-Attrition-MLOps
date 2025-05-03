import pytest
from unittest.mock import patch, MagicMock

# Patch MLflow and file I/O to avoid side effects
def test_save_reference_data_smoke():
    with patch('scripts.save_reference_data.mlflow') as mock_mlflow, \
         patch('scripts.save_reference_data.load_and_clean_data', return_value=MagicMock()), \
         patch('scripts.save_reference_data.AddNewFeaturesTransformer', return_value=MagicMock()), \
         patch('scripts.save_reference_data.AgeGroupTransformer', return_value=MagicMock()), \
         patch('scripts.save_reference_data.pd.DataFrame.to_parquet'), \
         patch('scripts.save_reference_data.pd.DataFrame.to_csv'):
        from scripts.save_reference_data import save_reference_data
        result = save_reference_data()
        assert result in [True, False]  # Should not raise, should return a boolean 