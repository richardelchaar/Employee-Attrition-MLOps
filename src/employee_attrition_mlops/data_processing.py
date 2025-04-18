# src/employee_attrition_mlops/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox, skew
import logging
from .config import INITIAL_COLS_TO_DROP, TARGET_COLUMN, BUSINESS_TRAVEL_MAPPING

logger = logging.getLogger(__name__)

# --- Custom Transformers (Implementations from Course 1 needed here) ---

class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Adds AgeAtJoining, TenureRatio, IncomePerYearExp features."""
    def __init__(self):
        self.new_feature_names = ['AgeAtJoining', 'TenureRatio', 'IncomePerYearExp']

    def fit(self, X, y=None):
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X_ = X.copy()
        logger.info("Adding new features: AgeAtJoining, TenureRatio, IncomePerYearExp")
        # Calculate AgeAtJoining
        if 'Age' in X_.columns and 'YearsAtCompany' in X_.columns:
            X_['AgeAtJoining'] = X_['Age'] - X_['YearsAtCompany']
        else:
            logger.warning("Could not create AgeAtJoining: Missing Age or YearsAtCompany")
            X_['AgeAtJoining'] = np.nan # Add NaN column if calculation fails

        # Calculate TenureRatio
        if 'YearsAtCompany' in X_.columns and 'TotalWorkingYears' in X_.columns:
            denominator = X_['TotalWorkingYears'].replace({0: np.nan}) # Avoid division by zero
            ratio = X_['YearsAtCompany'] / denominator
            X_['TenureRatio'] = ratio.fillna(0) # Fill NaNs resulting from 0 denominator or missing inputs
        else:
            logger.warning("Could not create TenureRatio: Missing YearsAtCompany or TotalWorkingYears")
            X_['TenureRatio'] = np.nan

        # Calculate IncomePerYearExp
        if 'MonthlyIncome' in X_.columns and 'TotalWorkingYears' in X_.columns:
            denominator = X_['TotalWorkingYears'].replace({0: np.nan})
            ratio2 = X_['MonthlyIncome'] / denominator
            X_['IncomePerYearExp'] = ratio2.fillna(0)
        else:
            logger.warning("Could not create IncomePerYearExp: Missing MonthlyIncome or TotalWorkingYears")
            X_['IncomePerYearExp'] = np.nan

        # Handle potential NaNs introduced if source columns were missing
        X_ = X_.fillna({'AgeAtJoining': 0, 'TenureRatio': 0, 'IncomePerYearExp': 0}) # Example fill strategy

        return X_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.concatenate([np.array(input_features), np.array(self.new_feature_names)])

class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    """Creates AgeGroup categorical feature."""
    def __init__(self):
        self.feature_name_out = "AgeGroup"

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X_ = X.copy()
        logger.info("Creating AgeGroup feature")
        if 'Age' in X_.columns:
                bins = [17, 30, 40, 50, 61] # Bins: 18-30, 31-40, 41-50, 51-60
                labels = ['18-30', '31-40', '41-50', '51-60']
                X_[self.feature_name_out] = pd.cut(X_['Age'], bins=bins, labels=labels, right=True)
                # Convert to string to ensure it's treated as categorical
                X_[self.feature_name_out] = X_[self.feature_name_out].astype(str)
                # Handle potential NaNs if age is outside bins (shouldn't happen with these bins)
                if X_[self.feature_name_out].isnull().any():
                    logger.warning("NaNs found in AgeGroup, filling with 'Unknown'")
                    X_[self.feature_name_out] = X_[self.feature_name_out].fillna('Unknown')
        else:
                logger.error("Column 'Age' not found for AgeGroupTransformer.")
                # Add NaN column to prevent downstream errors, or raise error
                X_[self.feature_name_out] = np.nan
        return X_

    def get_feature_names_out(self, input_features=None):
            if input_features is None:
                input_features = self.feature_names_in_
            return np.append(np.array(input_features), self.feature_name_out)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Applies a predefined mapping for ordinal encoding to specified columns."""
    def __init__(self, mapping=None, cols=None):
        self.mapping = mapping if mapping is not None else {}
        self.cols = cols if cols is not None else []

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        # Validate that provided columns exist in X
        self.valid_cols_ = [col for col in self.cols if col in X.columns]
        if len(self.valid_cols_) != len(self.cols):
            missing = set(self.cols) - set(self.valid_cols_)
            logger.warning(f"Columns not found for CustomOrdinalEncoder during fit: {missing}")
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.valid_cols_:
            logger.info(f"Applying custom ordinal encoding to {col}")
            original_nan_count = X_[col].isnull().sum()
            X_[col] = X_[col].map(self.mapping)
            # Handle unknown values (not in mapping) -> NaN, then fill
            nan_after_map = X_[col].isnull().sum()
            unknown_count = nan_after_map - original_nan_count
            if unknown_count > 0:
                logger.warning(f"{unknown_count} unknown values found in {col} during mapping. Filling with -1.")
                X_[col] = X_[col].fillna(-1) # Fill NaNs introduced by mapping
            # Fill pre-existing NaNs if any (optional, could be handled by imputer)
            if original_nan_count > 0 and X_[col].isnull().any():
                    logger.warning(f"Filling pre-existing NaNs in {col} with -1 after mapping.")
                    X_[col] = X_[col].fillna(-1) # Fill original NaNs too
        return X_

    def get_feature_names_out(self, input_features=None):
            # Assumes it doesn't change feature names
            names = input_features if input_features is not None else self.feature_names_in_
            return np.array(names)


class LogTransformSkewed(BaseEstimator, TransformerMixin):
    """Applies log1p transformation to specified skewed columns."""
    def __init__(self, skewed_cols=None):
        self.skewed_cols = skewed_cols if skewed_cols is not None else []

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.valid_skewed_cols_ = [col for col in self.skewed_cols if col in X.columns]
        if len(self.valid_skewed_cols_) != len(self.skewed_cols):
                missing = set(self.skewed_cols) - set(self.valid_skewed_cols_)
                logger.warning(f"Columns not found for LogTransformSkewed during fit: {missing}")
        return self

    def transform(self, X):
        X_ = X.copy()
        if not self.valid_skewed_cols_: return X_ # No columns to transform
        logger.info(f"Applying log1p transform to skewed columns: {self.valid_skewed_cols_}")
        for col in self.valid_skewed_cols_:
            # Ensure column is numeric and handle potential errors
            if pd.api.types.is_numeric_dtype(X_[col]):
                if (X_[col] < 0).any():
                    logger.warning(f"Column {col} contains negative values. Log1p might produce NaNs.")
                X_[col] = np.log1p(X_[col])
            else:
                logger.warning(f"Column {col} is not numeric. Skipping log transform.")
        return X_

    def get_feature_names_out(self, input_features=None):
            names = input_features if input_features is not None else self.feature_names_in_
            return np.array(names)

# --- Data Loading and Initial Processing ---

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Loads data, performs initial cleaning, and basic feature engineering."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {path}")
        raise

    # 1. Initial Cleaning
    cols_to_drop_present = [col for col in INITIAL_COLS_TO_DROP if col in df.columns]
    if cols_to_drop_present:
        df = df.drop(columns=cols_to_drop_present)
        logger.info(f"Dropped initial columns: {cols_to_drop_present}")
    initial_rows = len(df)
    df = df.drop_duplicates()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} duplicate rows.")

    # 2. Handle Missing Values (Example: Simple fill - enhance if needed)
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values detected. Applying simple fillna (median/mode).")
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Basic Feature Engineering (can be expanded or done in pipeline)
    # Example: Create AgeGroup here if not done via transformer in pipeline
    age_grouper = AgeGroupTransformer()
    df = age_grouper.fit_transform(df)
    # Example: Apply custom ordinal encoding for BusinessTravel here
    bt_encoder = CustomOrdinalEncoder(mapping=BUSINESS_TRAVEL_MAPPING, cols=['BusinessTravel'])
    if 'BusinessTravel' in df.columns:
        df = bt_encoder.fit_transform(df)

    logger.info(f"Data cleaned. Shape after initial processing: {df.shape}")
    return df

def identify_column_types(df: pd.DataFrame, target_column: str = None) -> dict:
    """Identifies numerical, categorical, and ordinal columns dynamically."""
    if target_column and target_column in df.columns:
        features_df = df.drop(columns=[target_column])
    else:
        features_df = df

    numerical_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Refine heuristic for ordinal (adjust nunique threshold or use explicit list)
    potential_ordinal = [
        col for col in numerical_cols
        if features_df[col].nunique() < 15 and features_df[col].min() >= 0 # Example heuristic
        # Add more conditions if needed, e.g., check if values are sequential integers
    ]
    # Example: Explicitly define known ordinal cols from domain knowledge
    known_ordinal = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                        'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                        'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
    actual_ordinal = [col for col in potential_ordinal if col in known_ordinal]

    # Update numerical_cols to exclude identified ordinals
    numerical_cols = [col for col in numerical_cols if col not in actual_ordinal]

    # Separate BusinessTravel if it's still categorical (might be pre-encoded)
    business_travel_col = []
    if 'BusinessTravel' in categorical_cols:
            business_travel_col = ['BusinessTravel']
            categorical_cols.remove('BusinessTravel')
    elif 'BusinessTravel' in numerical_cols: # If already ordinally encoded
            # Decide if it needs special handling or leave as numerical/ordinal
            logger.info("'BusinessTravel' found in numerical columns (likely pre-encoded).")
            # Optionally move to ordinal list if needed by pipeline
            # if 'BusinessTravel' not in actual_ordinal: actual_ordinal.append('BusinessTravel')
            # numerical_cols.remove('BusinessTravel')


    col_types = {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "ordinal": actual_ordinal,
        "business_travel": business_travel_col # List containing the col name or empty
    }

    logger.info(f"Identified Column Types: { {k: len(v) for k, v in col_types.items()} }")
    return col_types


def find_skewed_columns(df: pd.DataFrame, num_cols: list, threshold: float = 0.75) -> list:
    """Finds numerical columns with skewness above a threshold."""
    if not num_cols: return []
    skewed_features = []
    try:
        skewness = df[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        # logger.debug(f"Skewness calculated:\n{skewness}") # Use debug level
        skewed_features = skewness[abs(skewness) > threshold].index.tolist()
        logger.info(f"Found {len(skewed_features)} skewed features (threshold > {threshold}): {skewed_features}")
    except Exception as e:
        logger.error(f"Error calculating skewness: {e}")
    return skewed_features

