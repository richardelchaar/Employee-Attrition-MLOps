# src/employee_attrition_mlops/pipelines.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Import RF here
from sklearn.feature_selection import RFE, SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
# Ensure AddNewFeaturesTransformer is imported correctly
from .data_processing import (AddNewFeaturesTransformer, CustomOrdinalEncoder,
                                LogTransformSkewed, BoxCoxSkewedTransformer)
from .config import (BUSINESS_TRAVEL_MAPPING, RANDOM_STATE)
# from .outlier_removal import IsolationForestRemover # If implementing
import logging
import numpy as np
from sklearn.base import BaseEstimator # Import BaseEstimator

logger = logging.getLogger(__name__)

def create_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list,
    ordinal_cols: list,
    business_travel_col: list,
    skewed_cols: list,
    numeric_transformer_type: str = 'log',
    numeric_scaler_type: str = 'standard',
    business_encoder_type: str = 'onehot',
    # outlier_remover_active: bool = False
) -> ColumnTransformer:
    """Creates the preprocessing ColumnTransformer based on dynamic inputs."""

    logger.info("Building preprocessing pipeline...")
    logger.info(f"Num cols: {len(numerical_cols)}, Cat cols: {len(categorical_cols)}, Ord cols: {len(ordinal_cols)}, BT col: {business_travel_col}")
    logger.info(f"Skewed cols: {len(skewed_cols)}")
    logger.info(f"Numeric Transform: {numeric_transformer_type}, Scaler: {numeric_scaler_type}, BT Encoder: {business_encoder_type}")

    # --- Numeric Pipeline ---
    if numeric_transformer_type == 'log':
        num_transform = LogTransformSkewed(skewed_cols=skewed_cols)
    elif numeric_transformer_type == 'boxcox':
        num_transform = BoxCoxSkewedTransformer(skewed_cols=skewed_cols)
    else:
        num_transform = 'passthrough'

    if numeric_scaler_type == 'standard':
        num_scaler = StandardScaler()
    elif numeric_scaler_type == 'minmax':
        num_scaler = MinMaxScaler()
    else:
        num_scaler = 'passthrough'

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('transformer', num_transform),
        ('scaler', num_scaler)
    ], verbose=False)

    # --- Categorical Pipelines ---
    # Use 'sparse_output' if sklearn >= 1.2, else 'sparse'
    try:
        OneHotEncoder(sparse_output=False)
        ohe_sparse_arg = {'sparse_output': False}
        logger.debug("Using sparse_output=False for OneHotEncoder (sklearn >= 1.2)")
    except TypeError:
        ohe_sparse_arg = {'sparse': False}
        logger.debug("Using sparse=False for OneHotEncoder (sklearn < 1.2)")


    if business_encoder_type == 'ordinal' and business_travel_col:
            business_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', CustomOrdinalEncoder(mapping=BUSINESS_TRAVEL_MAPPING, cols=business_travel_col))
            ])
    elif business_travel_col:
        business_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', **ohe_sparse_arg))
        ])
    else:
            business_pipeline = 'drop'


    other_categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', **ohe_sparse_arg))
    ])

    # --- Ordinal Pipeline ---
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    # --- Assemble ColumnTransformer ---
    transformers = []
    if numerical_cols:
        transformers.append(('num', numeric_pipeline, numerical_cols))
    if ordinal_cols:
        transformers.append(('ord', ordinal_pipeline, ordinal_cols))
    if categorical_cols:
        transformers.append(('cat', other_categorical_pipeline, categorical_cols))
    if business_travel_col and business_pipeline != 'drop':
            transformers.append(('bus', business_pipeline, business_travel_col))

    if not transformers:
            logger.warning("No transformers specified for ColumnTransformer!")
            return ColumnTransformer(transformers=[], remainder='passthrough')


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    try:
        preprocessor.set_output(transform="pandas")
    except AttributeError:
        logger.warning("set_output(transform='pandas') not available in this scikit-learn version. Output might be numpy array.")
    except Exception as e:
         logger.warning(f"Could not set pandas output for ColumnTransformer: {e}")


    return preprocessor


def create_full_pipeline(
    classifier_class,
    model_params: dict,
    preprocessor: ColumnTransformer,
    feature_selector_type: str = 'passthrough',
    feature_selector_params: dict = None, # Expects params *for* the selector
    smote_active: bool = True,
    # outlier_remover_active: bool = False
) -> ImbPipeline:
    """Creates the full modeling pipeline including feature selection, SMOTE, and classifier."""

    logger.info(f"Building full pipeline with Selector: {feature_selector_type}, SMOTE: {smote_active}")

    feature_selector_params = feature_selector_params or {}
    feature_selector = 'passthrough' # Default

    # --- Feature Selection ---
    try:
        if feature_selector_type == 'rfe':
            # Instantiate the RFE estimator internally
            # Use a simple default estimator for RFE itself
            rfe_base_estimator = LogisticRegression(max_iter=200, solver='liblinear', random_state=RANDOM_STATE)
            n_features = feature_selector_params.get('n_features_to_select', 10) # Get n_features from Optuna params
            feature_selector = RFE(estimator=rfe_base_estimator, n_features_to_select=n_features)
            logger.info(f"Using RFE feature selection with n_features={n_features}")

        elif feature_selector_type == 'lasso':
            # Instantiate the LogisticRegression estimator for SelectFromModel internally
            lasso_C = feature_selector_params.get('C', 1.0) # Get C from Optuna params, default 1.0
            # Note: SelectFromModel needs an estimator trained with L1 penalty
            lasso_estimator_instance = LogisticRegression(
                penalty='l1',
                solver='liblinear', # Required for L1 in older sklearn
                C=lasso_C,
                random_state=RANDOM_STATE,
                max_iter=1000 # Reasonable default
            )
            threshold = feature_selector_params.get('threshold', -np.inf) # Use -inf to select based on non-zero coeffs
            feature_selector = SelectFromModel(estimator=lasso_estimator_instance, threshold=threshold, prefit=False)
            logger.info(f"Using Lasso (SelectFromModel) feature selection with C={lasso_C}, threshold={threshold}")

        elif feature_selector_type == 'tree':
            # Instantiate the RandomForestClassifier estimator for SelectFromModel internally
            threshold = feature_selector_params.get('threshold', 'median') # Get threshold from Optuna params
            # Use a simple RF for feature selection step
            tree_estimator_instance = RandomForestClassifier(
                n_estimators=50, # Keep it relatively small for speed
                random_state=RANDOM_STATE,
                n_jobs=1
            )
            feature_selector = SelectFromModel(estimator=tree_estimator_instance, threshold=threshold, prefit=False)
            logger.info(f"Using Tree (SelectFromModel) feature selection with threshold={threshold}")

        else: # passthrough or unknown
            feature_selector = 'passthrough'
            if feature_selector_type != 'passthrough':
                 logger.warning(f"Unknown feature_selector_type '{feature_selector_type}'. Defaulting to 'passthrough'.")
            logger.info("No feature selection applied.")

    except Exception as fs_err:
         logger.error(f"Error setting up feature selector '{feature_selector_type}': {fs_err}. Defaulting to 'passthrough'.", exc_info=True)
         feature_selector = 'passthrough'


    # --- SMOTE ---
    smote_step = ('smote', SMOTE(random_state=RANDOM_STATE)) if smote_active else ('smote', 'passthrough')

    # --- Classifier ---
    try:
        classifier = classifier_class(**model_params)
        logger.info(f"Using classifier: {classifier_class.__name__} with params: {model_params}")
    except Exception as clf_err:
         logger.error(f"Error instantiating classifier {classifier_class.__name__} with params {model_params}: {clf_err}", exc_info=True)
         raise

    # --- Assemble Pipeline ---
    # AddNewFeaturesTransformer is now the first step
    full_pipeline = ImbPipeline([
        ('feature_eng', AddNewFeaturesTransformer()), # Ensure this runs first
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        smote_step,
        ('classifier', classifier)
    ], verbose=False)

    return full_pipeline
