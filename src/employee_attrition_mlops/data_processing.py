# src/employee_attrition_mlops/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox, skew
import logging
from sqlalchemy import create_engine, text # Removed URL
from sqlalchemy.exc import SQLAlchemyError # Added for DB error handling
import os # Added
from dotenv import load_dotenv # Added

# Import relevant config variables
# This assumes config.py is in the same directory or correctly in the python path
try:
    from .config import (TARGET_COLUMN, BUSINESS_TRAVEL_MAPPING,
                         COLS_TO_DROP_POST_LOAD, DB_HISTORY_TABLE,
                         DATABASE_URL_PYODBC,
                         SNAPSHOT_DATE_COL, SKEWNESS_THRESHOLD) # Added DB related vars & SKEWNESS_THRESHOLD
except ImportError as e:
     # Fallback or error handling if config import fails
     logging.error(f"Could not import from .config: {e}. Using fallback values or defaults.")
     # Define fallbacks or raise error
     TARGET_COLUMN = "Attrition"
     BUSINESS_TRAVEL_MAPPING = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
     COLS_TO_DROP_POST_LOAD = ['EmployeeCount', 'StandardHours', 'Over18']
     DB_HISTORY_TABLE = "employees_history"
     DATABASE_URL_PYODBC = os.getenv("DATABASE_URL_PYODBC") # Try loading directly as fallback
     SNAPSHOT_DATE_COL = "SnapshotDate"
     SKEWNESS_THRESHOLD = 0.75


logger = logging.getLogger(__name__)

# --- Custom Transformers (Keep As Is - Ensure these are implemented) ---

class BoxCoxSkewedTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Box-Cox transformation to specified skewed columns.
    Handles non-positive values by adding a shift before transformation.
    """
    def __init__(self, skewed_cols=None):
        # Ensure skewed_cols is a list or None
        if skewed_cols is not None and not isinstance(skewed_cols, list):
             self.skewed_cols = [skewed_cols] # Handle single column name
        else:
             self.skewed_cols = skewed_cols if skewed_cols is not None else []
        self.lambdas_ = {} # Stores fitted lambda for each column
        self.shifts_ = {} # Stores shift applied for non-positive columns

    def fit(self, X, y=None):
        """
        Fits the Box-Cox transformation by finding the optimal lambda for each specified column.
        Calculates necessary shifts for non-positive columns.
        """
        # Store feature names for get_feature_names_out
        self.feature_names_in_ = list(X.columns)
        # Filter to only columns present in X
        self.valid_skewed_cols_ = [col for col in self.skewed_cols if col in X.columns]
        if len(self.valid_skewed_cols_) != len(self.skewed_cols):
            missing = set(self.skewed_cols) - set(self.valid_skewed_cols_)
            logger.warning(f"Columns not found for BoxCoxSkewedTransformer during fit: {missing}")

        logger.info(f"Fitting BoxCox for columns: {self.valid_skewed_cols_}")
        for col in self.valid_skewed_cols_:
            col_data = X[col]
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(col_data):
                logger.warning(f"Column '{col}' is not numeric. Skipping BoxCox fit.")
                self.lambdas_[col] = None
                self.shifts_[col] = 0
                continue

            min_val = col_data.min()
            shift = 0
            # Box-Cox requires positive values
            if min_val <= 0:
                shift = abs(min_val) + 1e-6 # Add a small epsilon to ensure positivity
                logger.warning(f"Column '{col}' contains non-positive values. Applying shift: {shift:.6f}")
            self.shifts_[col] = shift

            # Fit Box-Cox to find optimal lambda
            try:
                # Ensure no NaNs before fitting boxcox
                data_to_fit = col_data.dropna() + shift
                if data_to_fit.empty or not np.all(data_to_fit > 0):
                     logger.error(f"Cannot fit BoxCox on column '{col}' after shift/dropna (empty or still non-positive). Skipping.")
                     self.lambdas_[col] = None
                     self.shifts_[col] = 0 # Reset shift if fit fails
                     continue

                # lmbda=None finds optimal lambda
                # Note: boxcox returns (transformed_data, lambda)
                _, fitted_lambda = boxcox(data_to_fit, lmbda=None)
                self.lambdas_[col] = fitted_lambda # Store the lambda value
                logger.info(f"Fitted BoxCox for '{col}'. Lambda: {self.lambdas_[col]:.4f}, Shift: {self.shifts_[col]:.6f}")

            except ValueError as e:
                 # Box-Cox can fail if data is constant or has other issues
                 logger.error(f"BoxCox fit failed for column '{col}': {e}. Skipping transform for this column.")
                 self.lambdas_[col] = None # Mark as failed
                 self.shifts_[col] = 0 # Reset shift if fit fails
            except Exception as e:
                 logger.error(f"Unexpected error during BoxCox fit for column '{col}': {e}", exc_info=True)
                 self.lambdas_[col] = None
                 self.shifts_[col] = 0

        return self

    def transform(self, X):
        """Applies the fitted Box-Cox transformation to the specified columns."""
        X_ = X.copy()
        logger.info(f"Applying BoxCox transform to columns: {list(self.lambdas_.keys())}")

        for col, lmbda in self.lambdas_.items():
            if lmbda is not None and col in X_.columns:
                shift = self.shifts_.get(col, 0)
                col_data = X_[col]

                # Check if column is numeric before transforming
                if not pd.api.types.is_numeric_dtype(col_data):
                    logger.warning(f"Column '{col}' is not numeric. Skipping BoxCox transform.")
                    continue

                # Apply shift
                data_to_transform = col_data + shift

                # Handle potential NaNs introduced by shift or already present
                original_nans = col_data.isnull()
                if data_to_transform.isnull().any():
                     logger.warning(f"NaNs present in column '{col}' before BoxCox application.")
                     # BoxCox function might handle NaNs or raise error depending on version/usage
                     # Apply transform only to non-NaNs
                     not_nan_mask = ~data_to_transform.isnull()
                     if not_nan_mask.any(): # Only transform if there are non-NaN values
                          try:
                              # Apply boxcox only to the non-NaN part
                              transformed_values = boxcox(data_to_transform[not_nan_mask], lmbda=lmbda)
                              # Create a series with NaNs in original positions
                              result_col = pd.Series(np.nan, index=X_.index, dtype=float)
                              result_col.loc[not_nan_mask] = transformed_values # Use .loc for alignment
                              X_[col] = result_col
                          except Exception as e:
                               logger.error(f"Error applying BoxCox transform to non-NaN part of '{col}': {e}. Leaving column untransformed.")
                     else:
                          logger.warning(f"Column '{col}' contains only NaNs after shift. Leaving untransformed.")

                elif not np.all(data_to_transform > 0):
                     logger.error(f"Column '{col}' still contains non-positive values after shift ({data_to_transform.min()}). Cannot apply BoxCox. Leaving untransformed.")
                else:
                     # Apply Box-Cox transform directly if no NaNs and all positive
                     try:
                          X_[col] = boxcox(data_to_transform, lmbda=lmbda)
                     except Exception as e:
                          logger.error(f"Error applying BoxCox transform to '{col}': {e}. Leaving column untransformed.")

            elif col in X_.columns:
                 # Only log warning if lambda was expected but is None (fit failed)
                 if col in self.valid_skewed_cols_ and lmbda is None:
                      logger.warning(f"Skipping BoxCox transform for '{col}' as lambda was not successfully fitted.")
            # else: column not found, already warned in fit

        return X_

    def get_feature_names_out(self, input_features=None):
         """Returns feature names, which are unchanged by this transformer."""
         if input_features is None:
             # Use stored names from fit if available
             if hasattr(self, 'feature_names_in_'):
                 return np.array(self.feature_names_in_)
             else:
                 # This should ideally not happen if fit was called
                 logger.error("Transformer has not been fitted yet. Cannot determine output feature names.")
                 # Return an empty array or raise error, depending on desired behavior
                 return np.array([])
         else:
             # Input features provided, assume they are the output names
             return np.array(input_features)

class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Adds AgeAtJoining, TenureRatio, IncomePerYearExp features."""
    def __init__(self):
        # Define the names of the features this transformer will add
        self.new_feature_names = ['AgeAtJoining', 'TenureRatio', 'IncomePerYearExp']

    def fit(self, X, y=None):
        # Store input feature names during fit for use in get_feature_names_out
        self.feature_names_in_ = list(X.columns)
        # No actual fitting needed for this transformer
        return self

    def transform(self, X):
        """Calculates and adds the new features to the DataFrame."""
        X_ = X.copy() # Create a copy to avoid modifying the original DataFrame
        logger.info("Adding new features: AgeAtJoining, TenureRatio, IncomePerYearExp")

        # Calculate AgeAtJoining: Age when the employee joined the company
        if 'Age' in X_.columns and 'YearsAtCompany' in X_.columns:
            X_['AgeAtJoining'] = X_['Age'] - X_['YearsAtCompany']
        else:
            logger.warning("Could not create AgeAtJoining: Missing 'Age' or 'YearsAtCompany' column.")
            X_['AgeAtJoining'] = np.nan # Add NaN column if calculation fails to maintain structure

        # Calculate TenureRatio: Ratio of years at the company to total working years
        if 'YearsAtCompany' in X_.columns and 'TotalWorkingYears' in X_.columns:
            # Replace 0 in TotalWorkingYears with NaN to avoid division by zero errors
            denominator = X_['TotalWorkingYears'].replace({0: np.nan})
            ratio = X_['YearsAtCompany'] / denominator
            # Fill resulting NaNs (from division by zero or missing inputs) with 0
            X_['TenureRatio'] = ratio.fillna(0)
        else:
            logger.warning("Could not create TenureRatio: Missing 'YearsAtCompany' or 'TotalWorkingYears' column.")
            X_['TenureRatio'] = np.nan # Add NaN column if calculation fails

        # Calculate IncomePerYearExp: Ratio of monthly income to total working years
        if 'MonthlyIncome' in X_.columns and 'TotalWorkingYears' in X_.columns:
            # Replace 0 in TotalWorkingYears with NaN
            denominator = X_['TotalWorkingYears'].replace({0: np.nan})
            ratio2 = X_['MonthlyIncome'] / denominator
            # Fill resulting NaNs with 0
            X_['IncomePerYearExp'] = ratio2.fillna(0)
        else:
            logger.warning("Could not create IncomePerYearExp: Missing 'MonthlyIncome' or 'TotalWorkingYears' column.")
            X_['IncomePerYearExp'] = np.nan # Add NaN column if calculation fails

        # Handle potential NaNs introduced if source columns were missing or calculations failed
        # Fill any remaining NaNs in the new columns with 0 (or another appropriate strategy)
        X_ = X_.fillna({'AgeAtJoining': 0, 'TenureRatio': 0, 'IncomePerYearExp': 0})

        return X_

    def get_feature_names_out(self, input_features=None):
        """Returns the names of all features after transformation."""
        # If input_features are not provided, use the ones stored during fit
        if input_features is None:
            input_features = self.feature_names_in_
        # Concatenate the original feature names with the new feature names
        return np.concatenate([np.array(input_features), np.array(self.new_feature_names)])


class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    """Creates AgeGroup categorical feature based on Age."""
    def __init__(self):
        # Name of the new feature to be created
        self.feature_name_out = "AgeGroup"

    def fit(self, X, y=None):
        # Store input feature names during fit
        self.feature_names_in_ = list(X.columns)
        # No actual fitting needed
        return self

    def transform(self, X):
        """Bins the 'Age' column into predefined groups."""
        X_ = X.copy()
        logger.info("Creating AgeGroup feature")
        if 'Age' in X_.columns:
                # Define the age bins and corresponding labels
                bins = [17, 30, 40, 50, 61] # Bins: 18-30, 31-40, 41-50, 51-60
                labels = ['18-30', '31-40', '41-50', '51-60']
                # Use pandas.cut to create the bins
                X_[self.feature_name_out] = pd.cut(X_['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
                # Convert the resulting categorical type to string to ensure consistent handling
                X_[self.feature_name_out] = X_[self.feature_name_out].astype(str)
                # Handle potential NaNs if age is outside bins or was NaN initially
                if X_[self.feature_name_out].isnull().any():
                    # Check if NaNs were from original data or failed binning
                    original_nan_count = X_['Age'].isnull().sum()
                    new_nan_count = X_[self.feature_name_out].isnull().sum()
                    if new_nan_count > original_nan_count:
                        logger.warning(f"NaNs found in AgeGroup possibly due to ages outside defined bins [18-60]. Filling with 'Unknown'. Original Age NaNs: {original_nan_count}")
                    else:
                         logger.info(f"Original NaNs in 'Age' resulted in NaNs in 'AgeGroup'. Filling with 'Unknown'. Count: {new_nan_count}")
                    X_[self.feature_name_out] = X_[self.feature_name_out].fillna('Unknown')
        else:
                logger.error("Column 'Age' not found for AgeGroupTransformer.")
                # Add NaN column to prevent downstream errors, or raise error
                X_[self.feature_name_out] = np.nan
        return X_

    def get_feature_names_out(self, input_features=None):
            """Returns feature names including the newly added AgeGroup."""
            if input_features is None:
                input_features = self.feature_names_in_
            # Append the new feature name to the list of input features
            return np.append(np.array(input_features), self.feature_name_out)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Applies a predefined mapping for ordinal encoding to specified columns."""
    def __init__(self, mapping=None, cols=None):
        # Store the mapping dictionary (e.g., {'Low': 0, 'Medium': 1, 'High': 2})
        self.mapping = mapping if mapping is not None else {}
        # Store the list of columns to apply the mapping to
        self.cols = cols if cols is not None else []

    def fit(self, X, y=None):
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        # Validate that provided columns exist in X and store the valid ones
        self.valid_cols_ = [col for col in self.cols if col in X.columns]
        if len(self.valid_cols_) != len(self.cols):
            missing = set(self.cols) - set(self.valid_cols_)
            logger.warning(f"Columns not found for CustomOrdinalEncoder during fit: {missing}")
        # No actual fitting needed for the mapping itself
        return self

    def transform(self, X):
        """Applies the mapping to the specified columns."""
        X_ = X.copy()
        for col in self.valid_cols_:
            logger.info(f"Applying custom ordinal encoding to '{col}'")
            # Store count of NaNs before mapping to detect unknown values
            original_nan_count = X_[col].isnull().sum()
            # Apply the mapping using pandas .map()
            X_[col] = X_[col].map(self.mapping)
            # Check for NaNs introduced by values not present in the mapping keys
            nan_after_map = X_[col].isnull().sum()
            unknown_count = nan_after_map - original_nan_count
            if unknown_count > 0:
                logger.warning(f"{unknown_count} unknown value(s) found in '{col}' during mapping (not in mapping keys). Filling with -1.")
                # Fill NaNs introduced by the mapping (unknown values) with -1 (or other indicator)
                X_[col] = X_[col].fillna(-1)
            # Optionally, fill pre-existing NaNs as well if desired
            if original_nan_count > 0 and X_[col].isnull().any():
                    logger.warning(f"Filling {original_nan_count} pre-existing NaNs in '{col}' with -1 after mapping.")
                    X_[col] = X_[col].fillna(-1) # Fill original NaNs too
        return X_

    def get_feature_names_out(self, input_features=None):
            """Returns feature names (assumes names do not change)."""
            names = input_features if input_features is not None else self.feature_names_in_
            return np.array(names)


class LogTransformSkewed(BaseEstimator, TransformerMixin):
    """Applies log1p transformation (log(1+x)) to specified skewed columns."""
    def __init__(self, skewed_cols=None):
        # Store the list of columns identified as skewed
        self.skewed_cols = skewed_cols if skewed_cols is not None else []

    def fit(self, X, y=None):
        # Store input feature names
        self.feature_names_in_ = list(X.columns)
        # Validate that skewed columns exist in the DataFrame
        self.valid_skewed_cols_ = [col for col in self.skewed_cols if col in X.columns]
        if len(self.valid_skewed_cols_) != len(self.skewed_cols):
                missing = set(self.skewed_cols) - set(self.valid_skewed_cols_)
                logger.warning(f"Columns not found for LogTransformSkewed during fit: {missing}")
        # No actual fitting needed
        return self

    def transform(self, X):
        """Applies log1p transformation."""
        X_ = X.copy()
        if not self.valid_skewed_cols_:
            logger.info("No valid skewed columns provided for LogTransformSkewed. Skipping.")
            return X_ # No columns to transform

        logger.info(f"Applying log1p transform to skewed columns: {self.valid_skewed_cols_}")
        for col in self.valid_skewed_cols_:
            # Ensure column is numeric before applying log transform
            if pd.api.types.is_numeric_dtype(X_[col]):
                # Check for negative values, as log1p is undefined for x <= -1
                if (X_[col] < -1).any():
                    logger.error(f"Column '{col}' contains values <= -1. Log1p will produce NaNs or errors. Check data preprocessing.")
                    # Optional: Handle negative values (e.g., add shift, clip, or raise error)
                    # Example: Add a shift if minimum is negative but >= -1
                    # min_val = X_[col].min()
                    # if min_val < 0 and min_val >= -1:
                    #     shift = abs(min_val) + 1e-6
                    #     logger.warning(f"Applying shift {shift} to '{col}' before log1p due to negative values.")
                    #     X_[col] = np.log1p(X_[col] + shift)
                    # else: # Values <= -1 exist, skip or error
                    #     logger.error(f"Cannot apply log1p to '{col}' due to values <= -1.")
                    #     continue # Skip this column
                else:
                    # Apply log1p transformation
                    X_[col] = np.log1p(X_[col])
            else:
                logger.warning(f"Column '{col}' is not numeric. Skipping log transform.")
        return X_

    def get_feature_names_out(self, input_features=None):
            """Returns feature names (unchanged by this transformer)."""
            names = input_features if input_features is not None else self.feature_names_in_
            return np.array(names)


# --- Original Data Loading (Keep for reference or potential other uses) ---
def load_and_clean_data_from_csv(path: str) -> pd.DataFrame:
    """Loads data from CSV, performs initial cleaning."""
    # --- THIS IS THE ORIGINAL FUNCTION, RENAMED FOR CLARITY ---
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from CSV {path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"CSV data file not found at {path}")
        raise

    # 1. Initial Cleaning (using COLS_TO_DROP_POST_LOAD as example)
    cols_to_drop_present = [col for col in COLS_TO_DROP_POST_LOAD if col in df.columns]
    if cols_to_drop_present:
        df = df.drop(columns=cols_to_drop_present)
        logger.info(f"Dropped initial columns from CSV: {cols_to_drop_present}")
    initial_rows = len(df)
    df = df.drop_duplicates()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} duplicate rows from CSV.")

    # 2. Handle Missing Values (Example: Simple fill - enhance if needed)
    # NOTE: This might be better handled within the main ML pipeline's imputers
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values detected in CSV load. Applying simple fillna (median/mode). Consider imputation in pipeline.")
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                 median_val = df[col].median()
                 df[col] = df[col].fillna(median_val)
                 logger.debug(f"Filled NaNs in numeric column '{col}' with median ({median_val})")
        for col in df.select_dtypes(include='object').columns:
             if df[col].isnull().any():
                # Handle potential errors if mode() is empty
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    logger.debug(f"Filled NaNs in object column '{col}' with mode ('{mode_val}')")
                else:
                    df[col] = df[col].fillna('Unknown') # Or another placeholder
                    logger.debug(f"Filled NaNs in object column '{col}' with 'Unknown' (mode was empty)")

    # 3. Basic Feature Engineering (REMOVED - Handled by pipeline transformers)

    logger.info(f"CSV Data cleaned. Shape after initial processing: {df.shape}")
    return df


# --- NEW: Data Loading from Database ---
def load_and_clean_data_from_db(table_name: str = DB_HISTORY_TABLE) -> pd.DataFrame:
    """
    Loads data from the specified database table using SQLAlchemy with pyodbc.
    Handles potential connection errors, and performs initial cleaning.
    Uses DATABASE_URL_PYODBC from config.

    Args:
        table_name: The name of the table to load data from.

    Returns:
        A pandas DataFrame containing the loaded and initially cleaned data,
        or None if loading fails.
    """
    if not DATABASE_URL_PYODBC:
        logger.error("DATABASE_URL_PYODBC is not configured. Cannot load data from DB.")
        return None

    connection_string = DATABASE_URL_PYODBC # Use the specific URL for pyodbc
    logger.info(f"Attempting DB connection using pyodbc driver (URL from config).")

    engine = None
    df = None
    try:
        # Establish database connection using the pyodbc URL
        engine = create_engine(connection_string)
        logger.info(f"Successfully created SQLAlchemy engine using pyodbc.")

        # Check if table exists
        with engine.connect() as connection:
            logger.info(f"Checking if table '{table_name}' exists...")
            check_query = text(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = :table_name")
            result = connection.execute(check_query, {"table_name": table_name}).fetchone()
            if result:
                logger.info(f"Table '{table_name}' found. Loading data...")
                df = pd.read_sql_table(table_name, con=connection)
                logger.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns from '{table_name}'.")
            else:
                logger.error(f"Table '{table_name}' does not exist in the database.")
                return None

        # --- Initial Data Cleaning ---
        if df is not None:
            logger.info("Starting initial data cleaning...")
            original_cols = df.columns.tolist()
            df.columns = [col.replace(' ', '').replace('(', '').replace(')', '').replace('-', '') for col in df.columns]
            renamed_cols = df.columns.tolist()
            if original_cols != renamed_cols:
                 logger.info(f"Renamed columns: {dict(zip(original_cols, renamed_cols))}")
            cols_to_drop = [col for col in COLS_TO_DROP_POST_LOAD if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped columns: {cols_to_drop}")
            if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == 'object':
                unique_vals = df[TARGET_COLUMN].unique()
                if set(unique_vals) <= {'Yes', 'No', None, np.nan}:
                    logger.info(f"Converting target column '{TARGET_COLUMN}' ('Yes'/'No') to numeric (1/0).")
                    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).astype(float)
                else:
                    logger.warning(f"Target column '{TARGET_COLUMN}' is object type but contains unexpected values: {unique_vals}. Skipping automatic conversion.")
            if SNAPSHOT_DATE_COL in df.columns:
                 try:
                     df[SNAPSHOT_DATE_COL] = pd.to_datetime(df[SNAPSHOT_DATE_COL])
                     logger.info(f"Converted '{SNAPSHOT_DATE_COL}' column to datetime objects.")
                 except Exception as e:
                     logger.error(f"Error converting '{SNAPSHOT_DATE_COL}' to datetime: {e}. Check column format.")
            logger.info("Initial data cleaning finished.")

    except SQLAlchemyError as e:
        logger.error(f"Database error during connection or query (pyodbc): {e}", exc_info=True)
        df = None
    except ImportError as e:
        logger.error(f"ImportError: pyodbc driver might not be installed: {e}")
        logger.error("Please ensure pyodbc is installed ('poetry install')")
        df = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading/cleaning from DB (pyodbc): {e}", exc_info=True)
        df = None
    finally:
        if engine:
            engine.dispose()
            logger.info("Database engine (pyodbc) disposed.")

    if df is None:
        logger.error("Failed to load data from the database using pyodbc.")

    return df


# --- Utilities (Keep As Is - Ensure these are implemented correctly) ---
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
    # These should match the column names loaded from the DB
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
    elif 'BusinessTravel' in numerical_cols: # If already ordinally encoded by a previous step/DB
            # Decide if it needs special handling or leave as numerical/ordinal
            logger.info("'BusinessTravel' found in numerical columns (likely pre-encoded). Check pipeline needs.")
            # Optionally move to ordinal list if needed by pipeline
            # if 'BusinessTravel' not in actual_ordinal: actual_ordinal.append('BusinessTravel')
            # numerical_cols.remove('BusinessTravel')

    # Ensure EmployeeNumber is not treated as a feature if present
    id_col = "EmployeeNumber"
    if id_col in numerical_cols: numerical_cols.remove(id_col)
    if id_col in categorical_cols: categorical_cols.remove(id_col)
    if id_col in actual_ordinal: actual_ordinal.remove(id_col)


    col_types = {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "ordinal": actual_ordinal,
        "business_travel": business_travel_col # List containing the col name or empty
    }

    logger.info(f"Identified Column Types for Pipeline: { {k: len(v) for k, v in col_types.items()} }")
    logger.debug(f"Numerical Columns: {numerical_cols}")
    logger.debug(f"Categorical Columns: {categorical_cols}")
    logger.debug(f"Ordinal Columns: {actual_ordinal}")
    logger.debug(f"Business Travel Column: {business_travel_col}")
    return col_types


def find_skewed_columns(df: pd.DataFrame, num_cols: list, threshold: float = SKEWNESS_THRESHOLD) -> list:
    """Finds numerical columns with skewness above a threshold."""
    if not num_cols:
        logger.info("No numerical columns provided to find_skewed_columns.")
        return []
    skewed_features = []
    # Ensure only columns present in the dataframe are considered
    valid_num_cols = [col for col in num_cols if col in df.columns]
    if len(valid_num_cols) != len(num_cols):
        missing = set(num_cols) - set(valid_num_cols)
        logger.warning(f"Columns not found in DataFrame for skewness check: {missing}")

    if not valid_num_cols:
         logger.warning("No valid numerical columns left for skewness check.")
         return []

    try:
        # Calculate skewness only on valid numerical columns
        skewness = df[valid_num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        # logger.debug(f"Skewness calculated:\n{skewness}") # Use debug level
        skewed_features = skewness[abs(skewness) > threshold].index.tolist()
        logger.info(f"Found {len(skewed_features)} skewed features (threshold > {threshold}): {skewed_features}")
    except Exception as e:
        logger.error(f"Error calculating skewness: {e}", exc_info=True)
    return skewed_features


def load_and_clean_data(path: str = None) -> pd.DataFrame:
    """
    Load and clean data from either a CSV file or database.
    
    Args:
        path (str, optional): Path to CSV file. If None, loads from database.
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    if path is not None and path.endswith('.csv'):
        return load_and_clean_data_from_csv(path)
    else:
        return load_and_clean_data_from_db()
