import inspect
import time
import logging
from typing import Dict, Optional
import warnings
import numpy as np
import sklearn
import pandas as pd
from MultiTrain.classification import classification_models 
from MultiTrain.regression import regression_models

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FunctionTransformer, make_pipeline

from sklearn.preprocessing import LabelEncoder
from MultiTrain.errors.errors import *


import dask.dataframe as gpd

from dask_ml.preprocessing import LabelEncoder as LabelEncoderGpu

from utils.model_metric_utils import _init_metrics, _models_classifier, _models_regressor

logger = logging.getLogger(__name__)

from sklearn.exceptions import ConvergenceWarning

# Suppress all sklearn warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from dataclasses import dataclass

@dataclass
class inMultiClassifier(classification_models.MultiClassifier):
    pass

@dataclass
class inMultiRegressor(regression_models.MultiRegressor):
    pass

gpu_multiclassifier = inMultiClassifier()
gpu_multiregressor = inMultiRegressor()

use_gpu_classifier = gpu_multiclassifier.use_gpu
use_gpu_regressor = gpu_multiregressor.use_gpu
    

if use_gpu_classifier or use_gpu_regressor:
    from sklearnex import patch_sklearn
    patch_sklearn()

def _cat_encoder(cat_data, auto_cat_encode):
    """
    Encodes categorical columns in the dataset using Label Encoding.

    Args:
        cat_data (pd.DataFrame): The dataset containing categorical data.
        auto_cat_encode (bool): If True, automatically encodes all categorical columns.

    Returns:
        pd.DataFrame: The dataset with encoded categorical columns.
    """
    cat_columns = list(cat_data.select_dtypes(include=["object", "category"]).columns)

    if auto_cat_encode is True:
        le = LabelEncoder()
        cat_data[cat_columns] = (
            cat_data[cat_columns].astype(str).apply(le.fit_transform)
        )
        return cat_data
    else:
        # Raise an error if columns are not encoded
        raise MultiTrainEncodingError(
            f"Ensure that all columns are encoded before splitting the dataset. Set "
            "auto_cat_encode to True or specify manual_encode"
        )


def _manual_encoder(manual_encode, dataset):
    """
    Manually encodes specified columns in the dataset using specified encoding types.

    Args:
        manual_encode (dict): Dictionary specifying encoding types and columns.
        dataset (pd.DataFrame): The dataset to encode.

    Returns:
        pd.DataFrame: The dataset with manually encoded columns.
    """
    encoder_types = ["label", "onehot"]

    for encode_type, encode_columns in manual_encode.items():
        if encode_type not in encoder_types:
            raise MultiTrainEncodingError(
                f"Encoding type {encode_type} not found. Use one of the following: {encoder_types}"
            )

    if "label" in manual_encode.keys():
        le = LabelEncoder()
        dataset[manual_encode["label"]] = dataset[manual_encode["label"]].apply(
            le.fit_transform
        )

    if "onehot" in manual_encode.keys():
        encode_columns = manual_encode["onehot"]
        for column in encode_columns:
            # Apply one-hot encoding
            dummies = pd.get_dummies(dataset[column], prefix=column)
            # Concatenate the new dummy columns with the original dataset
            dataset = pd.concat([dataset, dummies], axis=1)
            # Drop the original categorical column
            dataset.drop(column, axis=1, inplace=True)

    return dataset


def _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode):
    """
    Checks for non-encoded categorical columns and raises an error if found.

    Args:
        dataset (pd.DataFrame): The dataset to check.
        auto_cat_encode (bool): Indicates if automatic encoding is enabled.
        manual_encode (dict): Manual encoding instructions.

    Raises:
        MultiTrainEncodingError: If non-encoded columns are found.
    """
    for i in dataset.columns:
        if dataset[i].dtype == "object":
            if auto_cat_encode is False and manual_encode is None:
                raise MultiTrainEncodingError(
                    f"Ensure that all columns are encoded before splitting the dataset. Column '{i}' is not encoded.\nSet "
                    "auto_cat_encode to True or pass in a manual encoding dictionary."
                )


def _fill_missing_values(dataset, column):
    """
    Fills missing values in a specified column of the dataset.

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
    Handles missing values in the dataset using specified strategies.

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
    # Check for missing values in the dataset and raise an error if found
    if dataset.isna().values.any():
        if not fix_nan_custom:
            raise MultiTrainNaNError(
                f"Missing values found in the dataset. Please handle missing values before proceeding."
                "Pass a value to the fix_nan_custom parameter to handle missing values. i.e. "
                'fix_nan_custom={"column1": "ffill", "column2": "bfill"}'
            )

    if fix_nan_custom:
        if type(fix_nan_custom) != dict:
            raise MultiTrainTypeError(
                f"fix_nan_custom should be a dictionary of type {dict}. Got {type(fix_nan_custom)}. "
                '\nExample: {"column1": "ffill", "column2": "bfill", "column3": ["value"]}'
            )

        fill_list = ["ffill", "bfill", "interpolate"]

        for column, strategy in fix_nan_custom.items():
            if column not in dataset.columns:
                raise MultiTrainColumnMissingError(
                    f"Column {column} not found in list of columns. Please pass in a valid column."
                )

            if strategy in fill_list[:2]:
                dataset[column] = getattr(
                    dataset[column], strategy
                )()  # dataset[column].ffill()
                if dataset[column].isnull().any():
                    print(len(dataset[column]))
                    dataset[column] = _fill_missing_values(dataset, column)

            elif strategy == "interpolate":
                dataset[column] = getattr(dataset[column], strategy)(method="linear")
                if dataset[column].isnull().any():
                    dataset[column] = _fill_missing_values(dataset, column)

    return dataset


def _check_custom_models(custom_models, models):
    """
    Checks and retrieves custom models from the provided models dictionary.

    Args:
        custom_models (list): List of custom model names.
        models (dict): Dictionary of available models.

    Returns:
        tuple: A tuple containing a list of model names and a list of model instances.

    Raises:
        MultiTrainTypeError: If custom_models is not a list.
    """
    if custom_models is None:
        model_names = list(models.keys())
        model_list = list(models.values())
    elif custom_models:
        if type(custom_models) != list:
            raise MultiTrainTypeError(
                f"You must pass a list of models to the custom models parameter. Got type {type(custom_models)}"
            )

        model_names = custom_models
        model_list = [models[values] for values in models if values in custom_models]

    return model_names, model_list


def _prep_model_names_list(
    datasplits, custom_metric, random_state, n_jobs, custom_models, class_type, max_iter
):
    # Validate the datasplits parameter
    if type(datasplits) != tuple or len(datasplits) != 4:
        raise MultiTrainSplitError(
            'The "datasplits" parameter can only be of type tuple and must have 4 values. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.'
        )

    # Unpack the datasplits tuple
    X_train, X_test, y_train, y_test = (
        datasplits[0],
        datasplits[1],
        datasplits[2],
        datasplits[3],
    )

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

    # Check for custom models and get model names and list
    model_names, model_list = _check_custom_models(custom_models, models)

    return model_names, model_list, X_train, X_test, y_train, y_test


def _fit_pred(current_model, model_names, idx, X_train, y_train, X_test):
    """
    Fits a model and predicts on the test set, measuring the time taken.

    Args:
        current_model: The model to fit.
        model_names (list): List of model names.
        idx (int): Index of the current model.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.

    Returns:
        tuple: A tuple containing the fitted model, predictions, and time taken.
    """
    start = time.time()
    try:
        current_model.fit(X_train, y_train)
        current_prediction = current_model.predict(X_test)
    except ValueError:
        logger.error(f"{model_names[idx]} unable to fit properly")
        current_prediction = [np.nan for i in len(X_test)]
    end = time.time() - start

    return current_model, current_prediction, end


# Function to calculate metric
def _calculate_metric(metric_func, y_true, y_pred, average=None, task=None):
    try:
        if average:
            val = metric_func(y_true, y_pred, average=average)
        else:
            val = metric_func(y_true, y_pred)
    except Exception as e:
        val = np.nan
    return val


def _fit_pred_text(vectorizer, pipeline_dict, model, X_train, y_train, X_test):
    """
    Fits a text processing pipeline and predicts on the test set, measuring the time taken.

    Args:
        vectorizer (str): The type of vectorizer to use ('count' or 'tfidf').
        pipeline_dict (dict): Dictionary containing pipeline parameters.
        model: The model to fit.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.

    Returns:
        tuple: A tuple containing the fitted pipeline, predictions, and time taken.
    """
    vectorizer_map = {"count": CountVectorizer, "tfidf": TfidfVectorizer}

    if vectorizer not in vectorizer_map:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer}")

    VectorizerClass = vectorizer_map[vectorizer]

    try:
        start = time.time()
        pipeline = make_pipeline(
            VectorizerClass(
                ngram_range=pipeline_dict["ngram_range"],
                encoding=pipeline_dict["encoding"],
                max_features=pipeline_dict["max_features"],
                analyzer=pipeline_dict["analyzer"],
            ),
            model,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except Exception:
        start = time.time()
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
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

    end = time.time() - start
    return pipeline, predictions, end


def _display_table(
    results,
    sort=None,
    custom_metric=None,
    return_best_model: Optional[str] = None,
    task: str = None,
):
    """
    Displays a sorted table of results.

    Args:
        results (dict): The results to display.
        sort (str, optional): The metric to sort by.
        custom_metric (str, optional): A custom metric to include in sorting.
        return_best_model (str, optional): The metric to return the best model by, e.g., 'accuracy'.
    Returns:
        pd.DataFrame: A DataFrame of sorted results.
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
