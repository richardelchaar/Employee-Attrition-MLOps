# src/employee_attrition_mlops/pipelines.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from .data_processing import (AddNewFeaturesTransformer, CustomOrdinalEncoder,
                                LogTransformSkewed, BoxCoxSkewedTransformer)
from .config import (BUSINESS_TRAVEL_MAPPING, RANDOM_STATE)
# from .outlier_removal import IsolationForestRemover # If implementing
import logging

logger = logging.getLogger(__name__)

def create_preprocessing_pipeline(
    numerical_cols: list,
    categorical_cols: list,
    ordinal_cols: list,
    business_travel_col: list, # Expecting list ['BusinessTravel'] or []
    skewed_cols: list,
    numeric_transformer_type: str = 'log',
    numeric_scaler_type: str = 'standard',
    business_encoder_type: str = 'onehot',
    # outlier_remover_active: bool = False # Add if implementing outlier remover
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
    if business_encoder_type == 'ordinal' and business_travel_col:
            business_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Pass mapping and explicitly name the single column
                ('encoder', CustomOrdinalEncoder(mapping=BUSINESS_TRAVEL_MAPPING, cols=business_travel_col))
            ])
    elif business_travel_col: # Default to onehot if column exists
        business_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
    else: # No business travel column
            business_pipeline = 'drop' # Or 'passthrough' if handled differently


    other_categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    # --- Ordinal Pipeline ---
    # Simple imputation, no scaling by default
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
            # Use the specific pipeline created for business travel
            transformers.append(('bus', business_pipeline, business_travel_col))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop', # Ensure only specified columns are kept
        verbose_feature_names_out=False # Keep feature names cleaner
    )
    preprocessor.set_output(transform="pandas") # Keep as DataFrame

    return preprocessor


def create_full_pipeline(
    classifier_class, # The actual model class (e.g., LogisticRegression)
    model_params: dict,
    preprocessor: ColumnTransformer, # The result from create_preprocessing_pipeline
    feature_selector_type: str = 'passthrough',
    feature_selector_params: dict = None,
    smote_active: bool = True,
    # outlier_remover_active: bool = False # Add if implementing
) -> ImbPipeline:
    """Creates the full modeling pipeline including feature selection, SMOTE, and classifier."""

    logger.info(f"Building full pipeline with Selector: {feature_selector_type}, SMOTE: {smote_active}")

    # --- Feature Selection ---
    feature_selector_params = feature_selector_params or {}
    if feature_selector_type == 'rfe':
        rfe_estimator = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver='liblinear')
        n_features = feature_selector_params.get('n_features_to_select', 10)
        feature_selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features)
        logger.info(f"Using RFE feature selection with n_features={n_features}")
    elif feature_selector_type == 'lasso':
        lasso_estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE, **feature_selector_params)
        feature_selector = SelectFromModel(estimator=lasso_estimator, prefit=False)
        logger.info("Using Lasso (SelectFromModel) feature selection")
    elif feature_selector_type == 'tree':
        # Default to RF if no estimator specified in params
        tree_estimator_params = {k.replace('estimator__',''):v for k,v in feature_selector_params.items() if k.startswith('estimator__')}
        tree_estimator = feature_selector_params.get('estimator', RandomForestClassifier(random_state=RANDOM_STATE, **tree_estimator_params))
        threshold = feature_selector_params.get('threshold', 'median')
        feature_selector = SelectFromModel(estimator=tree_estimator, threshold=threshold, prefit=False)
        logger.info(f"Using Tree (SelectFromModel) feature selection with threshold={threshold}")
    else:
        feature_selector = 'passthrough'
        logger.info("No feature selection applied.")

    # --- SMOTE ---
    smote_step = ('smote', SMOTE(random_state=RANDOM_STATE)) if smote_active else ('smote', 'passthrough')

    # --- Classifier ---
    classifier = classifier_class(**model_params)
    logger.info(f"Using classifier: {classifier_class.__name__} with params: {model_params}")

    # --- Assemble Pipeline ---
    # Use ImbPipeline to handle SMOTE correctly
    full_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        # Add ('outlier', OutlierRemover(...)) here if implementing
        ('feature_selection', feature_selector),
        smote_step,
        ('classifier', classifier)
    ], verbose=False) # Set verbose=True for debugging steps

    return full_pipeline

