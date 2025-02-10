import inspect
import logging
import time
import warnings
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
    Perceptron,
    Ridge,
    RidgeCV,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    SGDRegressor,
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from dataclasses import dataclass
from xgboost import XGBClassifier, XGBRegressor

from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, NotFittedError
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def _models_classifier(random_state, n_jobs, max_iter):
    """
    Generate a dictionary of classifier models from various libraries.

    Each entry in the dictionary maps a classifier's name to an instance of the classifier.

    Args:
        random_state (int): Seed for the random number generator.
        n_jobs (int): Number of parallel jobs to run.
        max_iter (int): Maximum number of iterations for iterative algorithms.

    Returns:
        dict: A dictionary mapping classifier names to their instances.
    """
    from MultiTrain.classification.classification_models import subMultiClassifier
    
    use_gpu_classifier = subMultiClassifier().use_gpu
    device_classifier = subMultiClassifier().device
    if use_gpu_classifier:
        from sklearnex import patch_sklearn
        patch_sklearn(global_patch=True)
        
    models_dict = {
        LogisticRegression.__name__: LogisticRegression(
            random_state=random_state, n_jobs=n_jobs, max_iter=max_iter
        ),
        LogisticRegressionCV.__name__: LogisticRegressionCV(
            n_jobs=n_jobs,
            max_iter=max_iter,
            cv=5,
        ),
        SGDClassifier.__name__: SGDClassifier(n_jobs=n_jobs, max_iter=max_iter),
        PassiveAggressiveClassifier.__name__: PassiveAggressiveClassifier(
            n_jobs=n_jobs, max_iter=max_iter
        ),
        RidgeClassifier.__name__: RidgeClassifier(max_iter=max_iter),
        RidgeClassifierCV.__name__: RidgeClassifierCV(cv=5),
        Perceptron.__name__: Perceptron(n_jobs=n_jobs, max_iter=max_iter),
        LinearSVC.__name__: LinearSVC(random_state=random_state, max_iter=max_iter),
        NuSVC.__name__: NuSVC(random_state=random_state, max_iter=max_iter),
        SVC.__name__: SVC(random_state=random_state, max_iter=max_iter),
        KNeighborsClassifier.__name__: KNeighborsClassifier(n_jobs=n_jobs),
        MLPClassifier.__name__: MLPClassifier(
            random_state=random_state, max_iter=max_iter
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
            random_state=random_state, n_jobs=n_jobs
        ),
        BaggingClassifier.__name__: BaggingClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        CatBoostClassifier.__name__: CatBoostClassifier(
            random_state=random_state,
            thread_count=n_jobs,
            silent=True,
            iterations=max_iter,
        ),
        RandomForestClassifier.__name__: RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        AdaBoostClassifier.__name__: AdaBoostClassifier(
            random_state=random_state, n_estimators=max_iter
        ),
        HistGradientBoostingClassifier.__name__: HistGradientBoostingClassifier(
            random_state=random_state, max_iter=max_iter
        ),
        LGBMClassifier.__name__: LGBMClassifier(
            random_state=random_state, n_jobs=n_jobs, verbose=-1, n_estimators=max_iter
        ),
        XGBClassifier.__name__: XGBClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter,
        ),
    }
    
    if use_gpu_classifier:
        models_dict[CatBoostClassifier.__name__].set_params(task_type='GPU', devices=device_classifier)
        models_dict[XGBClassifier.__name__].set_params(tree_method='gpu_hist', predictor='gpu_predictor')

    return models_dict

def _models_regressor(random_state, n_jobs, max_iter):
    """
    Generate a dictionary of regressor models from various libraries.

    Each entry in the dictionary maps a regressor's name to an instance of the regressor.

    Args:
        random_state (int): Seed for the random number generator.
        n_jobs (int): Number of parallel jobs to run.
        max_iter (int): Maximum number of iterations for iterative algorithms.

    Returns:
        dict: A dictionary mapping regressor names to their instances.
    """

    from MultiTrain.regression.regression_models import subMultiRegressor
    
    use_gpu_regressor = subMultiRegressor().use_gpu
    device_regressor = subMultiRegressor().device
    if use_gpu_regressor:
        from sklearnex import patch_sklearn
        patch_sklearn()
        
    models_dict = {
        LinearRegression.__name__: LinearRegression(n_jobs=n_jobs),
        Ridge.__name__: Ridge(random_state=random_state),
        RidgeCV.__name__: RidgeCV(),
        Lasso.__name__: Lasso(random_state=random_state),
        LassoCV.__name__: LassoCV(),
        ElasticNet.__name__: ElasticNet(random_state=random_state),
        ElasticNetCV.__name__: ElasticNetCV(),
        SGDRegressor.__name__: SGDRegressor(
            random_state=random_state, max_iter=max_iter
        ),
        KNeighborsRegressor.__name__: KNeighborsRegressor(n_jobs=n_jobs),
        DecisionTreeRegressor.__name__: DecisionTreeRegressor(
            random_state=random_state
        ),
        ExtraTreeRegressor.__name__: ExtraTreeRegressor(random_state=random_state),
        RandomForestRegressor.__name__: RandomForestRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        ExtraTreesRegressor.__name__: ExtraTreesRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        GradientBoostingRegressor.__name__: GradientBoostingRegressor(
            random_state=random_state
        ),
        AdaBoostRegressor.__name__: AdaBoostRegressor(
            random_state=random_state, n_estimators=max_iter
        ),
        BaggingRegressor.__name__: BaggingRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        CatBoostRegressor.__name__: CatBoostRegressor(
            random_state=random_state,
            thread_count=n_jobs,
            silent=True,
            iterations=max_iter,
        ),
        LGBMRegressor.__name__: LGBMRegressor(
            random_state=random_state, n_jobs=n_jobs, verbose=-1, n_estimators=max_iter
        ),
        XGBRegressor.__name__: XGBRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter,
        ),
        HistGradientBoostingRegressor.__name__: HistGradientBoostingRegressor(
            random_state=random_state, max_iter=max_iter
        ),
    }
    
    if use_gpu_regressor:
        models_dict[CatBoostRegressor.__name__].set_params(task_type='GPU', devices=device_regressor)
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
        metrics[custom_metric] = globals().get(custom_metric)
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
    encoder_types = ["label", "onehot"]
    dataset_copy = dataset.copy()

    for encode_type, encode_columns in manual_encode.items():
        if encode_type not in encoder_types:
            raise MultiTrainEncodingError(
                f"Encoding type {encode_type} not found. Use one of the following: {encoder_types}"
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
    for i in dataset.columns:
        if dataset[i].dtype == "object":
            if not auto_cat_encode and manual_encode is None:
                raise MultiTrainEncodingError(
                    f"Ensure that all columns are encoded before splitting the dataset. Column '{i}' is not encoded.\nSet "
                    "auto_cat_encode to True or pass in a manual encoding dictionary."
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
        mode_val = dataset[column].mode()[0]
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
    
    # Check for missing values in the dataset and raise an error if found
    if dataset_copy.isna().values.any():
        if not fix_nan_custom:
            raise MultiTrainNaNError(
                f"Missing values found in the dataset. Please handle missing values before proceeding."
                "Pass a value to the fix_nan_custom parameter to handle missing values. i.e. "
                'fix_nan_custom={"column1": "ffill", "column2": "bfill"}'
            )

    if fix_nan_custom:
        if not isinstance(fix_nan_custom, dict):
            raise MultiTrainTypeError(
                f"fix_nan_custom should be a dictionary of type {dict}. Got {type(fix_nan_custom)}. "
                '\nExample: {"column1": "ffill", "column2": "bfill", "column3": ["value"]}'
            )

        fill_list = ["ffill", "bfill", "interpolate"]

        for column, strategy in fix_nan_custom.items():
            if column not in dataset_copy.columns:
                raise MultiTrainColumnMissingError(
                    f"Column {column} not found in list of columns. Please pass in a valid column."
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

    return dataset_copy


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
    elif custom_models:
        if not isinstance(custom_models, list):
            raise MultiTrainTypeError(
                f"You must pass a list of models to the custom models parameter. Got type {type(custom_models)}"
            )

        model_names = custom_models
        model_list = []
        for model_name in custom_models:
            if model_name not in models:
                raise MultiTrainModelError(f"Model {model_name} not found in available models")
            model_list.append(models[model_name])

    return model_names, model_list


def _prep_model_names_list(
    datasplits: tuple, 
    custom_metric: str, 
    random_state: int, 
    n_jobs: int, 
    custom_models: list, 
    class_type: str, 
    max_iter: int, 
    
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
            random_state=random_state, n_jobs=n_jobs, max_iter=max_iter
        )
    elif class_type == "regression":
        models = _models_regressor(
            random_state=random_state, n_jobs=n_jobs, max_iter=max_iter
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
      0.00005    -> "50.00 µs"
    """
    hrs = int(seconds // 3600)
    seconds_remaining = seconds % 3600
    mins = int(seconds_remaining // 60)
    secs = seconds_remaining % 60

    parts = []
    if hrs > 0:
        parts.append(f"{hrs} hr")
    if mins > 0:
        parts.append(f"{mins} m")
    
    # If seconds are at least one, display seconds.
    if secs >= 1:
        parts.append(f"{secs:.2f} s")
    # If less than 1 second but at least 1 millisecond, display milliseconds.
    elif secs >= 0.001:
        ms = secs * 1000
        parts.append(f"{ms:.2f} ms")
    else:
        us = secs * 1000000
        parts.append(f"{us:.2f} µs")
    
    return " ".join(parts)

def _sub_fit(current_model, X_train, y_train, X_test, pca_scaler):
    """
    Fit a model to the training data and predict on the test data using an optional PCA scaler.

    Args:
        current_model: The machine learning model to be fitted.
        X_train (pd.DataFrame or np.ndarray): The training feature set.
        y_train (pd.Series or np.ndarray): The training target set.
        X_test (pd.DataFrame or np.ndarray): The test feature set.
        pca_scaler (object or bool): A scaler object for PCA transformation or False if not used.

    Returns:
        tuple: A tuple containing the fitted model pipeline and the predictions on the test set.
    """
    # Initialize the steps for the pipeline with the current model
    steps = [(current_model.__class__.__name__, current_model)]
    
    # Determine the number of components for PCA based on the shape of X_train
    if isinstance(X_train, pd.DataFrame):
        n_components = min(X_train.shape[0], X_train.shape[1])
    elif isinstance(X_train, np.ndarray):
        n_components = 1 if X_train.ndim == 1 else min(X_train.shape[0], X_train.shape[1])
        
    if pca_scaler:
        steps.insert(0, (PCA.__name__, PCA(n_components=n_components, random_state=42)))  # Add PCA with fixed components
        steps.insert(0, (pca_scaler.__class__.__name__, pca_scaler))  # Add the provided scaler before PCA

    try:
        # Create a pipeline with the specified steps and fit it to the training data
        current_model_pipeline = Pipeline(steps)
        current_model_pipeline.fit(X_train, y_train)
        
        # Predict on the test data using the fitted pipeline
        current_prediction = current_model_pipeline.predict(X_test)
        
    except (ValueError, NotFittedError, FitFailedWarning) as e:
        logger.error(f"{current_model.__class__.__name__} unable to fit properly. Reason: {e}")
        current_prediction = np.full(len(X_test), np.nan)
        
    except AttributeError as e:
        logger.error(f"{current_model.__class__.__name__} unable to predict properly. Reason: {e}")
        current_prediction = np.full(len(X_test), np.nan)
        
    return current_model_pipeline, current_prediction
    
    
    
def _fit_pred(current_model, model_names, idx, X_train, y_train, X_test, pca_scaler):
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
    
    from MultiTrain.classification.classification_models import subMultiClassifier
    use_gpu = subMultiClassifier().use_gpu
    if use_gpu:
        from sklearnex import config_context
        
    device = subMultiClassifier().device
    start = time.time()

    if use_gpu:
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
        if any(np.isnan(y_pred)):
            return np.nan
            
        if average:
            val = metric_func(y_true, y_pred, average=average)
        else:
            val = metric_func(y_true, y_pred)
    except Exception as e:
        val = np.nan
    return val


def _fit_pred_text(vectorizer, pipeline_dict, model, X_train, y_train, X_test, pca):
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
    
    from MultiTrain.classification.classification_models import subMultiClassifier
    
    use_gpu = subMultiClassifier().use_gpu
    if use_gpu:
        from sklearnex import config_context
    device = subMultiClassifier().device
    vectorizer_map = {"count": CountVectorizer, "tfidf": TfidfVectorizer}

    if vectorizer not in vectorizer_map:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer}")

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
        
        pipeline, predictions = _sub_fit(pipeline, X_train, y_train, X_test, pca)

    except Exception:
        pipeline = make_pipeline(
            VectorizerClass(
                ngram_range=pipeline_dict["ngram_range"],
                encoding=pipeline_dict["encoding"],
                max_features=pipeline_dict["max_features"],
                analyzer=pipeline_dict["analyzer"],
            ),
            FunctionTransformer(
                lambda x: x.todense(),
                accept_sparse=True,
            ),
            model,
        )
        if use_gpu:
            with config_context(target_offload=f"gpu:{device}"):
                pipeline, predictions = _sub_fit(pipeline, X_train, y_train, X_test, pca)
        else:
            pipeline, predictions = _sub_fit(pipeline, X_train, y_train, X_test, pca)
            
    end = time.time() - start
        
    time_ = _format_time(end)
    return pipeline, predictions, time_


def _display_table(
    results,
    sort=None,
    custom_metric=None,
    return_best_model: Optional[str] = None,
    task: str = None,
):
    """
    Display a sorted table of results.

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
    results_df = pd.DataFrame(results).T
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
            "mean squared error": "mean squared error",
            "r2 score": "r2 score",
            "mean absolute error": "mean absolute error",
            "median absolute error": "median absolute error",
            "mean squared log error": "mean squared log error",
            "explained variance score": "explained variance score",
        },
    }

    # If a custom metric is provided, add it to the sorted_ dictionary for the specified task.
    if custom_metric:
        sorted_[task][custom_metric] = custom_metric

    # Check if sorting is requested.
    if sort:
        # Ensure that sorting and returning the best model are not both requested simultaneously.
        if return_best_model:
            raise MultiTrainError("You can only either sort or return a best model")
        
        # Proceed with sorting if the task and sort metric are valid.
        if task in sorted_ and sort in sorted_[task]:
            # Sort the DataFrame based on the specified metric. For regression tasks, sort in ascending order.
            results_df = results_df.sort_values(by=sorted_[task][sort], ascending=(task == "regression"))

            # Move the sorted column to the front of the DataFrame for better visibility.
            column_to_move = sorted_[task][sort]
            first_column = results_df.pop(column_to_move)
            results_df.insert(0, column_to_move, first_column)
        
        # Return the sorted DataFrame.
        return results_df
    else:
        # If sorting is not requested, check if returning the best model is requested.
        if return_best_model:
            # For classification tasks, sort in descending order to get the best model.
            if task == "classification":
                results_df = results_df.sort_values(
                    by=return_best_model, ascending=False
                ).head(1)
            # For regression tasks, determine the sorting order based on the metric.
            elif task == "regression":
                ascending_order = (
                    True
                    if return_best_model
                    in [
                        "mean squared error",
                        "mean absolute error",
                        "median absolute error",
                        "mean squared log error",
                    ]
                    else False
                )
                results_df = results_df.sort_values(
                    by=return_best_model, ascending=ascending_order
                ).head(1)

            # Return the DataFrame with the best model.
            return results_df
        else:
            # If neither sorting nor returning the best model is requested, return the original DataFrame.
            return results_df
