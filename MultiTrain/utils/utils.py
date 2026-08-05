import inspect
import logging
import platform
import time
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from MultiTrain.errors.errors import *

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    Ridge,
    RidgeCV,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    SGDRegressor,
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
    Lars,
    LarsCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    BayesianRidge,
    ARDRegression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, NuSVC, SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from dataclasses import dataclass
from xgboost import XGBClassifier, XGBRegressor

from sklearn.exceptions import FitFailedWarning, NotFittedError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Patch only estimators MultiTrain can accelerate.  Patching every sklearn
# entry also replaces train_test_split, whose sklearnex implementation does
# not preserve pandas DataFrame behavior in the supported dependency version.
_CLASSIFIER_ACCELERATED_PATCHES = [
    "ridge",
    "svc",
    "nusvc",
    "logisticregression",
    "kneighborsclassifier",
    "extratreesclassifier",
    "randomforestclassifier",
]
_REGRESSOR_ACCELERATED_PATCHES = [
    "ridge",
    "elasticnet",
    "lasso",
    "svr",
    "nusvr",
    "linearregression",
    "kneighborsregressor",
    "extratreesregressor",
    "randomforestregressor",
]


def _models_classifier(
    random_state=None,
    n_jobs=None,
    max_iter=None,
    use_gpu=False,
    device="0",
):
    """
    Generate a dictionary of classifier models from various libraries.

    Each entry in the dictionary maps a classifier's name to an instance of the classifier.

    Args:
        random_state (int, optional): Seed for the random number generator. Defaults to None.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to None.
        max_iter (int, optional): Maximum number of iterations for iterative algorithms. Defaults to None.

    Returns:
        dict: A dictionary mapping classifier names to their instances.
    """
    if random_state is not None and (
        not isinstance(random_state, int) or isinstance(random_state, bool)
    ):
        raise MultiTrainTypeError("random_state must be an integer or None")
    if n_jobs is not None and (
        not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0
    ):
        raise MultiTrainTypeError("n_jobs must be a non-zero integer or None")
    if max_iter is not None and (
        not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0
    ):
        raise MultiTrainTypeError("max_iter must be a positive integer or None")
    if not isinstance(use_gpu, bool):
        raise MultiTrainTypeError("use_gpu must be a boolean")
    if not isinstance(device, str) or not device:
        raise MultiTrainTypeError("device must be a non-empty string")

    if use_gpu and platform.system() != "Darwin":
        from sklearnex import patch_sklearn
        patch_sklearn(name=_CLASSIFIER_ACCELERATED_PATCHES)
        
    models_dict = {
        LogisticRegression.__name__: LogisticRegression(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1, max_iter=max_iter if max_iter is not None else 100
        ),
        LogisticRegressionCV.__name__: LogisticRegressionCV(
            n_jobs=n_jobs if n_jobs is not None else 1,
            max_iter=max_iter if max_iter is not None else 100,
            cv=5,
        ),
        SGDClassifier.__name__: SGDClassifier(n_jobs=n_jobs if n_jobs is not None else 1, max_iter=max_iter if max_iter is not None else 1000),
        PassiveAggressiveClassifier.__name__: PassiveAggressiveClassifier(
            n_jobs=n_jobs if n_jobs is not None else 1, max_iter=max_iter if max_iter is not None else 1000
        ),
        RidgeClassifier.__name__: RidgeClassifier(max_iter=max_iter if max_iter is not None else 1000),
        RidgeClassifierCV.__name__: RidgeClassifierCV(cv=5),
        Perceptron.__name__: Perceptron(n_jobs=n_jobs if n_jobs is not None else 1, max_iter=max_iter if max_iter is not None else 1000),
        LinearSVC.__name__: LinearSVC(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        NuSVC.__name__: NuSVC(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        SVC.__name__: SVC(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        KNeighborsClassifier.__name__: KNeighborsClassifier(n_jobs=n_jobs if n_jobs is not None else 1),
        MLPClassifier.__name__: MLPClassifier(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 1000
        ),
        GaussianNB.__name__: GaussianNB(),
        BernoulliNB.__name__: BernoulliNB(),
        MultinomialNB.__name__: MultinomialNB(),
        ComplementNB.__name__: ComplementNB(),
        DecisionTreeClassifier.__name__: DecisionTreeClassifier(
            random_state=random_state
        ),
        ExtraTreeClassifier.__name__: ExtraTreeClassifier(random_state=random_state),
        GradientBoostingClassifier.__name__: GradientBoostingClassifier(
            random_state=random_state
        ),
        ExtraTreesClassifier.__name__: ExtraTreesClassifier(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        BaggingClassifier.__name__: BaggingClassifier(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        CatBoostClassifier.__name__: CatBoostClassifier(
            random_state=random_state,
            thread_count=n_jobs if n_jobs is not None else 1,
            silent=True,
            iterations=max_iter if max_iter is not None else 1000,
        ),
        RandomForestClassifier.__name__: RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        AdaBoostClassifier.__name__: AdaBoostClassifier(
            random_state=random_state, n_estimators=max_iter if max_iter is not None else 50
        ),
        HistGradientBoostingClassifier.__name__: HistGradientBoostingClassifier(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 100
        ),
        LGBMClassifier.__name__: LGBMClassifier(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1, verbose=-1, n_estimators=max_iter if max_iter is not None else 100
        ),
        XGBClassifier.__name__: XGBClassifier(
            random_state=random_state,
            n_jobs=n_jobs if n_jobs is not None else 1,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter if max_iter is not None else 100,
        ),
    }
    
    if use_gpu and platform.system() != "Darwin":
        models_dict[CatBoostClassifier.__name__].set_params(task_type='GPU', devices=device)
        models_dict[XGBClassifier.__name__].set_params(tree_method='gpu_hist', predictor='gpu_predictor')

    return models_dict

def _models_regressor(
    random_state=None,
    n_jobs=None,
    max_iter=None,
    use_gpu=False,
    device="0",
):
    """
    Generate a dictionary of regressor models from various libraries.

    Each entry in the dictionary maps a regressor's name to an instance of the regressor.

    Args:
        random_state (int, optional): Seed for the random number generator. Defaults to None.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to None.
        max_iter (int, optional): Maximum number of iterations for iterative algorithms. Defaults to None.

    Returns:
        dict: A dictionary mapping regressor names to their instances.
    """

    if random_state is not None and (
        not isinstance(random_state, int) or isinstance(random_state, bool)
    ):
        raise MultiTrainTypeError("random_state must be an integer or None")
    if n_jobs is not None and (
        not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0
    ):
        raise MultiTrainTypeError("n_jobs must be a non-zero integer or None")
    if max_iter is not None and (
        not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0
    ):
        raise MultiTrainTypeError("max_iter must be a positive integer or None")
    if not isinstance(use_gpu, bool):
        raise MultiTrainTypeError("use_gpu must be a boolean")
    if not isinstance(device, str) or not device:
        raise MultiTrainTypeError("device must be a non-empty string")

    if use_gpu and platform.system() != "Darwin":
        from sklearnex import patch_sklearn
        patch_sklearn(name=_REGRESSOR_ACCELERATED_PATCHES)
        
    models_dict = {
        LinearRegression.__name__: LinearRegression(n_jobs=n_jobs if n_jobs is not None else 1),
        Ridge.__name__: Ridge(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        RidgeCV.__name__: RidgeCV(cv=5),
        Lasso.__name__: Lasso(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        LassoCV.__name__: LassoCV(max_iter=max_iter if max_iter is not None else 1000, cv=5),
        ElasticNet.__name__: ElasticNet(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        ElasticNetCV.__name__: ElasticNetCV(max_iter=max_iter if max_iter is not None else 1000, cv=5),
        Lars.__name__: Lars(random_state=random_state),
        LarsCV.__name__: LarsCV(),
        OrthogonalMatchingPursuit.__name__: OrthogonalMatchingPursuit(),
        # ``max_iter`` means "maximum selected atoms" for OMP, not optimizer
        # iterations.  Reusing MultiTrain's global iteration budget here can
        # exceed the feature count and make otherwise valid datasets fail.
        OrthogonalMatchingPursuitCV.__name__: OrthogonalMatchingPursuitCV(
            max_iter=None,
            n_jobs=n_jobs if n_jobs is not None else 1,
        ),
        BayesianRidge.__name__: BayesianRidge(n_iter=max_iter if max_iter is not None else 300),
        ARDRegression.__name__: ARDRegression(n_iter=max_iter if max_iter is not None else 300),
        HuberRegressor.__name__: HuberRegressor(max_iter=max_iter if max_iter is not None else 100),
        TheilSenRegressor.__name__: TheilSenRegressor(random_state=random_state, max_iter=max_iter if max_iter is not None else 300),
        RANSACRegressor.__name__: RANSACRegressor(random_state=random_state, max_trials=max_iter if max_iter is not None else 100),
        PoissonRegressor.__name__: PoissonRegressor(max_iter=max_iter if max_iter is not None else 100),
        GammaRegressor.__name__: GammaRegressor(max_iter=max_iter if max_iter is not None else 100),
        TweedieRegressor.__name__: TweedieRegressor(max_iter=max_iter if max_iter is not None else 100),
        SGDRegressor.__name__: SGDRegressor(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 1000
        ),
        PassiveAggressiveRegressor.__name__: PassiveAggressiveRegressor(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 1000
        ),
        KNeighborsRegressor.__name__: KNeighborsRegressor(n_jobs=n_jobs if n_jobs is not None else 1),
        DecisionTreeRegressor.__name__: DecisionTreeRegressor(
            random_state=random_state
        ),
        ExtraTreeRegressor.__name__: ExtraTreeRegressor(random_state=random_state),
        RandomForestRegressor.__name__: RandomForestRegressor(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        ExtraTreesRegressor.__name__: ExtraTreesRegressor(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        GradientBoostingRegressor.__name__: GradientBoostingRegressor(
            random_state=random_state, n_estimators=max_iter if max_iter is not None else 100
        ),
        AdaBoostRegressor.__name__: AdaBoostRegressor(
            random_state=random_state, n_estimators=max_iter if max_iter is not None else 50
        ),
        BaggingRegressor.__name__: BaggingRegressor(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1
        ),
        MLPRegressor.__name__: MLPRegressor(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 1000
        ),
        SVR.__name__: SVR(max_iter=max_iter if max_iter is not None else -1),
        LinearSVR.__name__: LinearSVR(random_state=random_state, max_iter=max_iter if max_iter is not None else 1000),
        NuSVR.__name__: NuSVR(max_iter=max_iter if max_iter is not None else -1),
        CatBoostRegressor.__name__: CatBoostRegressor(
            random_state=random_state,
            thread_count=n_jobs if n_jobs is not None else 1,
            silent=True,
            iterations=max_iter if max_iter is not None else 1000,
        ),
        LGBMRegressor.__name__: LGBMRegressor(
            random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1, verbose=-1, n_estimators=max_iter if max_iter is not None else 100
        ),
        XGBRegressor.__name__: XGBRegressor(
            random_state=random_state,
            n_jobs=n_jobs if n_jobs is not None else 1,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter if max_iter is not None else 100,
        ),
        HistGradientBoostingRegressor.__name__: HistGradientBoostingRegressor(
            random_state=random_state, max_iter=max_iter if max_iter is not None else 100
        ),
    }
    
    if use_gpu and platform.system() != "Darwin":
        models_dict[CatBoostRegressor.__name__].set_params(task_type='GPU', devices=device)
        models_dict[XGBRegressor.__name__].set_params(tree_method='gpu_hist', predictor='gpu_predictor')

    return models_dict

def _cat_encoder(cat_data, auto_cat_encode):
    """
    Encode categorical columns in the dataset using Label Encoding.

    Args:
        cat_data (pd.DataFrame): The dataset containing categorical data.
        auto_cat_encode (bool): If True, automatically encodes all categorical columns.

    Returns:
        pd.DataFrame: The dataset with encoded categorical columns.
    """
    cat_columns = list(cat_data.select_dtypes(include=["object", "category"]).columns)

    if auto_cat_encode:
        le = LabelEncoder()
        cat_data_copy = cat_data.copy()
        for col in cat_columns:
            cat_data_copy[col] = le.fit_transform(cat_data_copy[col].astype(str))
        return cat_data_copy
    else:
        # Raise an error if columns are not encoded
        raise MultiTrainEncodingError(
            f"Ensure that all columns are encoded before splitting the dataset. Set "
            "auto_cat_encode to True or specify manual_encode"
        )
        
def _init_metrics():
    """
    Initialize a list of default metric names.

    Returns:
        list: A list of metric names.
    """
    return [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "roc_auc_score",
        "balanced_accuracy_score",
        "mean_squared_error",
        "r2_score",
        "mean_absolute_error",
        "median_absolute_error",
        "mean_squared_log_error",
        "explained_variance_score"
    ]

def _metrics(custom_metric: str, metric_type: str):
    """
    Retrieve a dictionary of metric functions from sklearn.

    Each entry in the dictionary maps a metric's name to its function.

    Args:
        custom_metric (str): Name of a custom metric to include.
        metric_type (str): 'classification' or 'regression'

    Returns:
        dict: A dictionary mapping metric names to their functions.
    """
    valid_metrics = {
        "classification": {
            "precision": precision_score,
            "recall": recall_score,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        },
        "regression": {
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "mean_absolute_error": mean_absolute_error,
            "median_absolute_error": median_absolute_error,
            "mean_squared_log_error": mean_squared_log_error,
            "explained_variance_score": explained_variance_score,
        },
    }

    if metric_type not in valid_metrics:
        raise MultiTrainTypeError(
            "metric_type must be either 'classification' or 'regression'"
        )
    if custom_metric is not None and not isinstance(custom_metric, str):
        raise MultiTrainTypeError("custom_metric must be a string or None")

    if custom_metric:
        # Check if the custom metric is a valid sklearn metric
        valid_sklearn_metrics = [
            name
            for name, obj in inspect.getmembers(sklearn.metrics, inspect.isfunction)
        ]
        if custom_metric not in valid_sklearn_metrics:
            raise MultiTrainMetricError(
                f"Custom metric ({custom_metric}) is not a valid metric. Please check the sklearn documentation for a valid list of metrics."
            )
        # Add the custom metric to the appropriate metric type
        metrics = valid_metrics.get(metric_type, {}).copy()
        metrics[custom_metric] = getattr(sklearn.metrics, custom_metric)
        return metrics

    return valid_metrics.get(metric_type, {})

def _manual_encoder(manual_encode, dataset):
    """
    Manually encode specified columns in the dataset using specified encoding types.

    Args:
        manual_encode (dict): Dictionary specifying encoding types and columns.
        dataset (pd.DataFrame): The dataset to encode.

    Returns:
        pd.DataFrame: The dataset with manually encoded columns.
    """
    if not isinstance(manual_encode, dict):
        raise MultiTrainTypeError("manual_encode must be a dictionary")
    if not isinstance(dataset, pd.DataFrame):
        raise MultiTrainDatasetTypeError("dataset must be a pandas DataFrame")

    encoder_types = ["label", "onehot"]
    dataset_copy = dataset.copy()

    for encode_type, encode_columns in manual_encode.items():
        if encode_type not in encoder_types:
            raise MultiTrainEncodingError(
                f"Encoding type {encode_type} not found. Use one of the following: {encoder_types}"
            )
        if not isinstance(encode_columns, (list, tuple)):
            raise MultiTrainTypeError(
                f"Columns for {encode_type} encoding must be a list or tuple"
            )
        missing_columns = [
            column for column in encode_columns if column not in dataset_copy.columns
        ]
        if missing_columns:
            raise MultiTrainColumnMissingError(
                f"Columns not found in the dataset: {missing_columns}"
            )

    if "label" in manual_encode.keys():
        le = LabelEncoder()
        for col in manual_encode["label"]:
            dataset_copy[col] = le.fit_transform(dataset_copy[col].astype(str))

    if "onehot" in manual_encode.keys():
        encode_columns = manual_encode["onehot"]
        for column in encode_columns:
            # Apply one-hot encoding
            dummies = pd.get_dummies(dataset_copy[column], prefix=column)
            # Concatenate the new dummy columns with the original dataset
            dataset_copy = pd.concat([dataset_copy, dummies], axis=1)
            # Drop the original categorical column
            dataset_copy.drop(column, axis=1, inplace=True)

    return dataset_copy


def _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode):
    """
    Check for non-encoded categorical columns and raise an error if found.

    Args:
        dataset (pd.DataFrame): The dataset to check.
        auto_cat_encode (bool): Indicates if automatic encoding is enabled.
        manual_encode (dict): Manual encoding instructions.

    Raises:
        MultiTrainEncodingError: If non-encoded columns are found.
    """
    if auto_cat_encode:
        return

    manually_encoded = {
        column
        for columns in (manual_encode or {}).values()
        for column in columns
    }
    categorical_columns = dataset.select_dtypes(
        include=["object", "category"]
    ).columns
    unencoded_columns = [
        column for column in categorical_columns if column not in manually_encoded
    ]
    if unencoded_columns:
        raise MultiTrainEncodingError(
            "Ensure that all categorical columns are encoded before splitting the "
            f"dataset. Unencoded columns: {unencoded_columns}. Set auto_cat_encode "
            "to True or include every column in manual_encode."
        )


def _fill_missing_values(dataset, column):
    """
    Fill missing values in a specified column of the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing missing values.
        column (str): The column to fill missing values in.

    Returns:
        pd.Series: The column with missing values filled.
    """
    if dataset[column].dtype in ["object", "category"]:
        modes = dataset[column].mode()
        mode_val = modes.iloc[0] if not modes.empty else ""
        dataset[column] = dataset[column].fillna(mode_val)
        return dataset[column]
    else:
        dataset[column] = dataset[column].fillna(0)
        return dataset[column]


def _handle_missing_values(
    dataset: pd.DataFrame, fix_nan_custom: Optional[Dict] = False
) -> pd.DataFrame:
    """
    Handle missing values in the dataset using specified strategies.

    Args:
        dataset (pd.DataFrame): The dataset to handle missing values in.
        fix_nan_custom (dict, optional): Custom strategies for handling missing values.

    Returns:
        pd.DataFrame: The dataset with missing values handled.

    Raises:
        MultiTrainNaNError: If missing values are found and no strategy is provided.
        MultiTrainTypeError: If fix_nan_custom is not a dictionary.
        MultiTrainColumnMissingError: If a specified column is not found in the dataset.
    """
    dataset_copy = dataset.copy()

    if (
        fix_nan_custom is not False
        and fix_nan_custom is not None
        and not isinstance(fix_nan_custom, dict)
    ):
        raise MultiTrainTypeError(
            f"fix_nan_custom should be a dictionary. Got {type(fix_nan_custom)}"
        )
    
    # Check for missing values in the dataset and raise an error if found
    if dataset_copy.isna().values.any():
        if not fix_nan_custom:
            raise MultiTrainNaNError(
                f"Missing values found in the dataset. Please handle missing values before proceeding."
                "Pass a value to the fix_nan_custom parameter to handle missing values. i.e. "
                'fix_nan_custom={"column1": "ffill", "column2": "bfill"}'
            )

    if fix_nan_custom:
        fill_list = ["ffill", "bfill", "interpolate"]

        for column, strategy in fix_nan_custom.items():
            if column not in dataset_copy.columns:
                raise MultiTrainColumnMissingError(
                    f"Column {column} not found in list of columns. Please pass in a valid column."
                )
            
            if strategy not in fill_list:
                raise MultiTrainNaNError(
                    f"Strategy {strategy} not found in list of stragies. Please pass one of {fill_list}"
                )

            if strategy in fill_list[:2]:
                dataset_copy[column] = getattr(
                    dataset_copy[column], strategy
                )()  # dataset[column].ffill()
                if dataset_copy[column].isnull().any():
                    dataset_copy[column] = _fill_missing_values(dataset_copy, column)

            elif strategy == "interpolate":
                dataset_copy[column] = getattr(dataset_copy[column], strategy)(method="linear")
                if dataset_copy[column].isnull().any():
                    dataset_copy[column] = _fill_missing_values(dataset_copy, column)

    remaining_missing = list(dataset_copy.columns[dataset_copy.isna().any()])
    if remaining_missing:
        raise MultiTrainNaNError(
            f"Missing values remain in columns: {remaining_missing}"
        )

    return dataset_copy


def _prepare_train_test(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    auto_cat_encode: bool = False,
    manual_encode: Optional[Dict] = None,
    fix_nan_custom: Optional[Dict] = False,
):
    """Preprocess a split without learning categories or fill values from test data."""
    if not isinstance(train_dataset, pd.DataFrame) or not isinstance(
        test_dataset, pd.DataFrame
    ):
        raise MultiTrainDatasetTypeError(
            "train_dataset and test_dataset must both be pandas DataFrames"
        )
    if set(train_dataset.columns) != set(test_dataset.columns):
        raise MultiTrainColumnMissingError(
            "Train and test datasets must contain the same columns"
        )
    if manual_encode is not None and not isinstance(manual_encode, dict):
        raise MultiTrainTypeError("manual_encode must be a dictionary or None")
    if (
        fix_nan_custom is not False
        and fix_nan_custom is not None
        and not isinstance(fix_nan_custom, dict)
    ):
        raise MultiTrainTypeError("fix_nan_custom must be a dictionary or False")

    train_copy = train_dataset.copy()
    test_copy = test_dataset.reindex(columns=train_dataset.columns).copy()

    invalid_encoder_types = set((manual_encode or {})) - {"label", "onehot"}
    if invalid_encoder_types:
        raise MultiTrainEncodingError(
            f"Unsupported encoding types: {sorted(invalid_encoder_types)}. "
            "Use only 'label' and 'onehot'."
        )

    if train_copy.isna().values.any() or test_copy.isna().values.any():
        if not fix_nan_custom:
            raise MultiTrainNaNError(
                "Missing values found in the dataset. Pass fix_nan_custom with a "
                "strategy for every column containing missing values."
            )

    if fix_nan_custom:
        valid_strategies = {"ffill", "bfill", "interpolate"}
        for column, strategy in fix_nan_custom.items():
            if column not in train_copy.columns or column not in test_copy.columns:
                raise MultiTrainColumnMissingError(
                    f"Column {column} not found in the dataset."
                )
            if strategy not in valid_strategies:
                raise MultiTrainNaNError(
                    f"Strategy {strategy} is invalid. Use one of {sorted(valid_strategies)}"
                )

            if strategy == "interpolate":
                train_copy[column] = train_copy[column].interpolate(method="linear")
                test_copy[column] = test_copy[column].interpolate(method="linear")
            else:
                train_copy[column] = getattr(train_copy[column], strategy)()
                test_copy[column] = getattr(test_copy[column], strategy)()

            non_null_train = train_copy[column].dropna()
            if train_copy[column].dtype.name in {"object", "category"}:
                fill_value = (
                    non_null_train.mode().iloc[0] if not non_null_train.empty else ""
                )
            else:
                # Preserve the package's existing numeric fallback while ensuring
                # the test split never supplies the value used for imputation.
                fill_value = 0

            train_copy[column] = train_copy[column].fillna(fill_value)
            test_copy[column] = test_copy[column].fillna(fill_value)

    remaining_missing = [
        column
        for column in train_copy.columns
        if train_copy[column].isna().any() or test_copy[column].isna().any()
    ]
    if remaining_missing:
        raise MultiTrainNaNError(
            "Missing values remain in columns without a configured strategy: "
            f"{remaining_missing}"
        )

    if auto_cat_encode:
        label_columns = list(
            train_copy.select_dtypes(include=["object", "category"]).columns
        )
        onehot_columns = []
    else:
        for encoding_type, columns in (manual_encode or {}).items():
            if not isinstance(columns, (list, tuple)):
                raise MultiTrainTypeError(
                    f"Columns for {encoding_type} encoding must be a list or tuple"
                )
        label_columns = list((manual_encode or {}).get("label", []))
        onehot_columns = list((manual_encode or {}).get("onehot", []))

    for column in label_columns:
        if column not in train_copy.columns:
            raise MultiTrainColumnMissingError(
                f"Column {column} not found in the dataset."
            )
        train_values = train_copy[column].astype(str)
        test_values = test_copy[column].astype(str)
        mapping = {
            value: index for index, value in enumerate(pd.unique(train_values))
        }
        train_copy[column] = train_values.map(mapping).astype(int)
        # Categories seen only in the test split share one reserved code and are
        # deliberately not learned from the held-out rows.
        test_copy[column] = (
            test_values.map(mapping).fillna(len(mapping)).astype(int)
        )

    for column in onehot_columns:
        if column not in train_copy.columns:
            raise MultiTrainColumnMissingError(
                f"Column {column} not found in the dataset."
            )
        train_dummies = pd.get_dummies(
            train_copy[column], prefix=column, dtype=int
        )
        test_dummies = pd.get_dummies(
            test_copy[column], prefix=column, dtype=int
        ).reindex(columns=train_dummies.columns, fill_value=0)
        train_copy = pd.concat(
            [train_copy.drop(columns=[column]), train_dummies], axis=1
        )
        test_copy = pd.concat(
            [test_copy.drop(columns=[column]), test_dummies], axis=1
        )

    return train_copy, test_copy


def _check_custom_models(custom_models, models):
    """
    Check and retrieve custom models from the provided models dictionary.

    Args:
        custom_models (list): List of custom model names.
        models (dict): Dictionary of available models.

    Returns:
        tuple: A tuple containing a list of model names and a list of model instances.

    Raises:
        MultiTrainTypeError: If custom_models is not a list.
        MultiTrainModelError: If a custom model name is not found in available models.
    """
    if custom_models is None:
        model_names = list(models.keys())
        model_list = list(models.values())
    elif isinstance(custom_models, list):
        if not custom_models:
            raise MultiTrainModelError("custom_models cannot be an empty list")
        if len(custom_models) != len(set(custom_models)):
            raise MultiTrainModelError("custom_models cannot contain duplicates")
        model_names = custom_models
        model_list = []
        for model_name in custom_models:
            if model_name not in models:
                raise MultiTrainModelError(f"Model {model_name} not found in available models")
            model_list.append(models[model_name])
    else:
        raise MultiTrainTypeError(
            f"custom_models must be a list or None. Got {type(custom_models)}"
        )

    return model_names, model_list


def _prep_model_names_list(
    datasplits: tuple, 
    custom_metric: str, 
    random_state: int, 
    n_jobs: int, 
    custom_models: list, 
    class_type: str, 
    max_iter: int, 
    use_gpu: bool = False,
    device: str = "0",
) -> tuple:
    """
    Prepare model names and lists based on provided data splits and parameters.

    Args:
        datasplits (tuple): A tuple containing training and testing data splits.
        custom_metric (str): A custom metric to include.
        random_state (int): Seed for the random number generator.
        n_jobs (int): Number of parallel jobs to run.
        custom_models (list): List of custom model names.
        class_type (str): Type of classification or regression.
        max_iter (int): Maximum number of iterations for iterative algorithms.

    Returns:
        tuple: A tuple containing model names, model list, and data splits.

    Raises:
        MultiTrainSplitError: If datasplits is not a tuple of length 4.
        MultiTrainMetricError: If the custom metric already exists in default metrics.
    """
    if custom_metric is not None and not isinstance(custom_metric, str):
        raise MultiTrainTypeError("custom_metric must be a string or None")

    # Validate the datasplits parameter
    if not isinstance(datasplits, tuple) or len(datasplits) != 4:
        raise MultiTrainSplitError(
            'The "datasplits" parameter can only be of type tuple and must have 4 values. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.'
        )

    # Unpack the datasplits tuple
    X_train, X_test, y_train, y_test = datasplits

    # Check if the custom metric is already in the default metrics
    if custom_metric in _init_metrics():
        raise MultiTrainMetricError(
            f"Metric already exists. Please pass in a different metric."
            f"Default metrics are: {_init_metrics()}"
        )

    # Initialize models
    if class_type == "classification":
        models = _models_classifier(
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
            use_gpu=use_gpu,
            device=device,
        )
    elif class_type == "regression":
        models = _models_regressor(
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
            use_gpu=use_gpu,
            device=device,
        )
    else:
        raise MultiTrainTypeError(f"Invalid class_type: {class_type}. Must be 'classification' or 'regression'")

    # Check for custom models and get model names and list
    model_names, model_list = _check_custom_models(custom_models, models)

    return model_names, model_list, X_train, X_test, y_train, y_test

def _format_time(seconds):
    """
    Convert a duration (in seconds) into a human-readable string.
    The output will include hours, minutes, seconds, milliseconds, or microseconds.
    
    Examples:
      5400.1234 -> "1 hr 30 m 0.12 s"
      30.205     -> "30.21 s"
      0.02       -> "20.00 ms"
      0.00005    -> "50.00us"
    """
    if not isinstance(seconds, (int, float, np.number)) or isinstance(seconds, bool):
        raise MultiTrainTypeError("seconds must be a numeric value")
    if seconds < 0:
        raise MultiTrainError("seconds cannot be negative")

    hrs = int(seconds // 3600)
    seconds_remaining = seconds % 3600
    mins = int(seconds_remaining // 60)
    secs = seconds_remaining % 60

    parts = []
    if hrs > 0:
        parts.append(f"{hrs}hr")
    if mins > 0:
        parts.append(f"{mins}m")
    
    # If seconds are at least one, display seconds.
    if secs >= 1:
        parts.append(f"{secs:.2f}s")
    # If less than 1 second but at least 1 millisecond, display milliseconds.
    elif secs >= 0.001:
        ms = secs * 1000
        parts.append(f"{ms:.2f}ms")
    else:
        us = secs * 1000000
        parts.append(f"{us:.2f}us")
    
    return " ".join(parts)

def _sub_fit(
    current_model,
    X_train,
    y_train,
    X_test,
    pca_scaler,
    raise_on_error=False,
):
    """Fit one model pipeline and return predictions for the test data."""
    if not hasattr(X_train, "shape"):
        raise MultiTrainTypeError("X_train must be an array-like object with a shape")

    n_components = (
        1
        if len(X_train.shape) == 1
        else min(X_train.shape[0], X_train.shape[1])
    )
    steps = [(current_model.__class__.__name__, current_model)]
    if pca_scaler:
        steps.insert(0, (PCA.__name__, PCA(n_components=n_components, random_state=42)))
        steps.insert(0, (pca_scaler.__class__.__name__, pca_scaler))

    current_model_pipeline = None
    try:
        current_model_pipeline = Pipeline(steps)
        current_model_pipeline.fit(X_train, y_train)
        current_prediction = current_model_pipeline.predict(X_test)
    except Exception as exc:
        if raise_on_error:
            raise
        logger.error(
            f"{current_model.__class__.__name__} unable to fit or predict. Reason: {exc}"
        )
        current_prediction = np.full(len(X_test), np.nan)

    return current_model_pipeline, current_prediction
    
    
    
def _fit_pred(
    current_model,
    model_names,
    idx,
    X_train,
    y_train,
    X_test,
    pca_scaler,
    use_gpu=False,
    device="0",
):
    """
    Fit a model and predict on the test set, measuring the time taken.

    Args:
        current_model: The model to fit.
        model_names (list): List of model names.
        idx (int): Index of the current model.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.
        pca_scaler (object or bool): A scaler object for PCA transformation or False if not used.

    Returns:
        tuple: A tuple containing the fitted model, predictions, and time taken.
    """
    
    start = time.time()

    if use_gpu and platform.system() != 'Darwin':
        from sklearnex import config_context
        with config_context(target_offload=f"gpu:{device}"):
            current_model, current_prediction = _sub_fit(current_model, X_train, y_train, X_test, pca_scaler)    
    else:
        current_model, current_prediction = _sub_fit(current_model, X_train, y_train, X_test, pca_scaler)    
    
    end = time.time() - start
    
    time_ = _format_time(end)

    return current_model, current_prediction, time_


def _calculate_metric(metric_func, y_true, y_pred, average=None, task=None):
    """
    Calculate a metric using the provided metric function.

    Args:
        metric_func (callable): The metric function to use for calculation.
        y_true (array-like): True labels or target values.
        y_pred (array-like): Predicted labels or target values.
        average (str, optional): The type of averaging to perform on the data. 
            Common options include 'micro', 'macro', 'samples', 'weighted', and 'binary'.
        task (str, optional): The task type, e.g., 'classification' or 'regression'. 
            This parameter is currently not used in the function.

    Returns:
        float: The calculated metric value. Returns NaN if an error occurs during calculation.
    """
    try:
        if pd.isna(np.asarray(y_pred)).any():
            return np.nan
            
        if average:
            val = metric_func(y_true, y_pred, average=average)
        else:
            val = metric_func(y_true, y_pred)
    except Exception as e:
        val = np.nan
    return val


def _classification_roc_auc(model, X, y_true):
    """Calculate ROC AUC from scores, never from predicted class labels."""
    try:
        observed_classes = np.unique(y_true)
        model_classes = np.asarray(
            getattr(model, "classes_", observed_classes)
        )
        if len(observed_classes) < 2:
            return np.nan

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)
            if len(model_classes) == 2:
                scores = scores[:, 1]
                return roc_auc_score(y_true, scores)
            return roc_auc_score(
                y_true, scores, multi_class="ovr", average="weighted"
            )

        if len(model_classes) == 2 and hasattr(model, "decision_function"):
            return roc_auc_score(y_true, model.decision_function(X))
    except Exception:
        return np.nan

    return np.nan


def _fit_pred_text(
    vectorizer,
    pipeline_dict,
    model,
    X_train,
    y_train,
    X_test,
    pca,
    use_gpu=False,
    device="0",
):
    """
    Fit a text processing pipeline and predict on the test set, measuring the time taken.

    Args:
        vectorizer (str): The type of vectorizer to use ('count' or 'tfidf').
        pipeline_dict (dict): Dictionary containing pipeline parameters.
        model: The model to fit.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.
        pca (object or bool): A scaler object for PCA transformation or False if not used.

    Returns:
        tuple: A tuple containing the fitted pipeline, predictions, and time taken.

    Raises:
        MultiTrainPCAError: If PCA is attempted for NLP tasks.
    """
    
    if pca:
        raise MultiTrainPCAError('You cannot use pca for nlp tasks (when text is set to True)')
    
    
    vectorizer_map = {"count": CountVectorizer, "tfidf": TfidfVectorizer}

    if vectorizer not in vectorizer_map:
        raise MultiTrainTextError(f"Unsupported vectorizer type: {vectorizer}")

    required_pipeline_keys = {"ngram_range", "encoding", "max_features", "analyzer"}
    if not isinstance(pipeline_dict, dict):
        raise MultiTrainTextError("pipeline_dict must be a dictionary")
    missing_pipeline_keys = required_pipeline_keys - set(pipeline_dict)
    if missing_pipeline_keys:
        raise MultiTrainTextError(
            f"pipeline_dict is missing required keys: {sorted(missing_pipeline_keys)}"
        )

    VectorizerClass = vectorizer_map[vectorizer]
    start = time.time()
    try:
        
        pipeline = make_pipeline(
            VectorizerClass(
                ngram_range=pipeline_dict["ngram_range"],
                encoding=pipeline_dict["encoding"],
                max_features=pipeline_dict["max_features"],
                analyzer=pipeline_dict["analyzer"],
            ),
            model,
        )
        
        pipeline, predictions = _sub_fit(
            pipeline,
            X_train,
            y_train,
            X_test,
            pca,
            raise_on_error=True,
        )

    except Exception:
        pipeline = make_pipeline(
            VectorizerClass(
                ngram_range=pipeline_dict["ngram_range"],
                encoding=pipeline_dict["encoding"],
                max_features=pipeline_dict["max_features"],
                analyzer=pipeline_dict["analyzer"],
            ),
            FunctionTransformer(
                lambda x: np.asarray(x.todense()),
                accept_sparse=True,
            ),
            model,
        )
        if use_gpu and platform.system() != 'Darwin':
            from sklearnex import config_context
            with config_context(target_offload=f"gpu:{device}"):
                pipeline, predictions = _sub_fit(pipeline, X_train, y_train, X_test, pca)
        else:
            pipeline, predictions = _sub_fit(pipeline, X_train, y_train, X_test, pca)
            
    end = time.time() - start
        
    time_ = _format_time(end)
    return pipeline, predictions, time_


def _display_table(
    results: dict,
    sort: Optional[str] = None,
    custom_metric: Optional[str] = None,
    return_best_model: Optional[str] = None,
    task: Optional[str] = None,
) -> pd.DataFrame:
    """
    Displays a sorted table of results.

    Args:
        results (dict): The results to display.
        sort (str, optional): The metric to sort by.
        custom_metric (str, optional): A custom metric to include in sorting.
        return_best_model (str, optional): The metric to return the best model by, e.g., 'accuracy'.
        task (str, optional): The task type, e.g., 'classification' or 'regression'.

    Returns:
        pd.DataFrame: A DataFrame of sorted results.

    Raises:
        MultiTrainError: If both sorting and returning the best model are requested simultaneously.
    """
    
    valid_tasks = {"classification", "regression"}
    if task not in valid_tasks:
        raise MultiTrainTypeError(
            "task must be either 'classification' or 'regression'"
        )
    if not isinstance(results, dict) or not results:
        raise MultiTrainTypeError("results must be a non-empty dictionary")

    # Convert the results dictionary to a DataFrame and transpose it.
    results_df = pd.DataFrame(results).T
    
    # Define the default sorting mapping for each task.
    # Metrics that should be sorted in descending order (higher is better)
    descending_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy", "r2_score", "explained_variance_score", 
                          "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted", 
                          "f1_micro", "f1_macro", "f1_weighted", "roc_auc_ovr", "roc_auc_ovo", "jaccard", "matthews_corrcoef", "top_k_accuracy", 
                          "average_precision", "neg_log_loss", "adjusted_rand_score", "adjusted_mutual_info_score", "normalized_mutual_info_score", 
                          "homogeneity_score", "completeness_score", "v_measure_score", "fowlkes_mallows_score"]

    # Metrics that should be sorted in ascending order (lower is better) 
    ascending_metrics = ["mean_squared_error", "mean_absolute_error", "median_absolute_error", "mean_squared_log_error", "max_error", 
                         "mean_poisson_deviance", "mean_gamma_deviance", "mean_absolute_percentage_error", "d2_absolute_error_score", 
                         "d2_pinball_score", "d2_tweedie_score", "hamming_loss", "zero_one_loss", "hinge_loss", "log_loss", "brier_score_loss"]

    sorted_ = {
        "classification": {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1", 
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",
        },
        "regression": {
            "mean_squared_error": "mean_squared_error",
            "r2_score": "r2_score",
            "mean_absolute_error": "mean_absolute_error",
            "median_absolute_error": "median_absolute_error",
            "mean_squared_log_error": "mean_squared_log_error",
            "explained_variance_score": "explained_variance_score",
        },
    }

    # If a custom metric is provided, add it to the sorted_ dictionary for the specified task.
    if custom_metric and task:
        sorted_[task][custom_metric] = custom_metric

    # If a sorting metric is requested
    if sort is not None:
        if not isinstance(sort, str) or not sort:
            raise MultiTrainMetricError("sort must be a non-empty metric name or None")
        # Ensure that sorting and returning the best model are not both requested simultaneously.
        if return_best_model:
            raise MultiTrainError("You can only either sort or return a best model")
        
        if task not in sorted_ or sort not in sorted_[task]:
            raise MultiTrainMetricError(
                f"Metric {sort!r} is not available for {task}. "
                f"Choose one of {list(sorted_.get(task, {}))}."
            )
        if sort in ascending_metrics:
            ascending = True
        elif sort in descending_metrics:
            ascending = False
        else:
            raise MultiTrainMetricError(
                f"The preferred sort direction for custom metric {sort!r} is unknown."
            )

        results_df = results_df.sort_values(
            by=sorted_[task][sort],
            ascending=ascending
        )

        # Bring the sorted column to the front of the DataFrame for better visibility.
        column_to_move = sorted_[task][sort]
        first_column = results_df.pop(column_to_move)
        results_df.insert(0, column_to_move, first_column)

        return results_df
        
    else:
        
        if return_best_model:
            if return_best_model not in results_df.columns:
                raise MultiTrainMetricError(
                    f"Metric {return_best_model!r} is not present in the results. "
                    f"Choose one of {list(results_df.columns)}."
                )
            if task == "classification":
                if return_best_model not in descending_metrics:
                    raise MultiTrainMetricError(
                        f"The preferred ranking direction for {return_best_model!r} is unknown."
                    )
                results_df = results_df.sort_values(by=return_best_model, ascending=False).head(1)
                
            elif task == "regression":
                if return_best_model in ascending_metrics:
                    ascending_order = True
                elif return_best_model in descending_metrics:
                    ascending_order = False
                else:
                    raise MultiTrainMetricError(
                        f"The preferred ranking direction for {return_best_model!r} is unknown."
                    )
                
                results_df = results_df.sort_values(by=return_best_model, ascending=ascending_order).head(1)
                
            return results_df
        else:
            return results_df
