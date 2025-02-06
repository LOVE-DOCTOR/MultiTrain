from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn import logger

from MultiTrain.utils.utils import (
    _calculate_metric,
    _cat_encoder,
    _check_custom_models,
    _display_table,
    _fit_pred,
    _handle_missing_values,
    _init_metrics,
    _manual_encoder,
    _models_regressor,
    _metrics,
    _non_auto_cat_encode_error,
    _fit_pred_text,
    _prep_model_names_list,
)

import time
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm.notebook import trange, tqdm
from IPython.display import display
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from matplotlib import pyplot as plt
from numpy.random import randint
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    make_scorer,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import (
    FunctionTransformer,
    Normalizer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)

from MultiTrain.errors.errors import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

@dataclass
class MultiRegressor:
    n_jobs: int = -1
    random_state: int = 42
    custom_models: list = None
    overfit_tolerance: float = 0.2
    max_iter: int = 1000

    logger.warn('Version 1.0.0 introduces new syntax and you might experience errors if using old syntax, visit the documentation in the GitHub Repo.')
    
    @staticmethod
    def split(
        data: pd.DataFrame,
        target: str,  # Target column name
        random_state: int = 42,  # Default random state for reproducibility
        test_size: float = 0.2,  # Default test size for train-test split (80/20 split)
        auto_cat_encode: bool = False,  # If True, automatically encode all categorical columns
        manual_encode: dict = None,  # Manual encoding dictionary, e.g., {'label': ['column1'], 'onehot': ['column2']}
        fix_nan_custom: Optional[
            Dict
        ] = False,  # Custom NaN handling, e.g., {'column1': 'ffill'}
        drop: list = None,
    ):  # List of columns to drop, e.g., ['column1', 'column2']
        """
        Splits the dataset into training and testing sets after performing optional preprocessing steps.

        Parameters:
        - data (pd.DataFrame): The input dataset.
        - target (str): The name of the target column.
        - random_state (int, optional): Random state for reproducibility. Default is 42.
        - test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        - auto_cat_encode (bool, optional): If True, automatically encode all categorical columns. Default is False.
        - manual_encode (dict, optional): Dictionary specifying manual encoding for columns. Default is None.
        - fix_nan_custom (Optional[Dict], optional): Custom NaN handling instructions. Default is False.
        - drop (list, optional): List of columns to drop from the dataset. Default is None.

        Returns:
        - tuple: A tuple containing the training and testing data splits (X_train, X_test, y_train, y_test).
        """

        # Create a copy of the dataset to avoid modifying the original data
        dataset = data.copy()
        if manual_encode:
            keys = list(manual_encode.keys())
            if 1 < len(keys) < 3:
                if len(keys) != len(set(keys)):
                    raise MultiTrainError('You cannot have duplicates of either "label" or "onehot" in your dictionary.')
                if any(item in manual_encode[keys[0]] for item in manual_encode[keys[1]]):
                    raise MultiTrainError('You cannot not have a column specified for different types of encoding i.e column1 present for label and column2 present for onehot')
                if fix_nan_custom:
                    fix_keys = list(fix_nan_custom.keys())
                    if len(fix_keys) != len(set(fix_keys)):
                            raise MultiTrainError('You cannot specify a column as a key more than once')
            if list(keys) > 2:
                raise MultiTrainError('You cannot have more than two keys i.e label, onehot')
        
        # Drop specified columns if 'drop' parameter is provided
        if drop:
            if type(drop) != list:
                raise MultiTrainTypeError(
                    f"You need to pass in a list of columns to drop. Got {type(drop)}"
                )
            dataset.drop(drop, axis=1, inplace=True)

        # Ensure the dataset is a pandas DataFrame
        if type(dataset) != pd.DataFrame:
            raise MultiTrainDatasetTypeError(
                f"You need to pass in a Dataset of type {pd.DataFrame}. Got {type(dataset)}"
            )

        # Check if the target column exists in the dataset
        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(
                f"Target column {target} not found in list of columns. Please pass in a target column."
            )

        # Check for categorical columns and raise an error if necessary
        _non_auto_cat_encode_error(
            dataset=dataset,
            auto_cat_encode=auto_cat_encode,
            manual_encode=manual_encode,
        )

        # Handle missing values in the dataset using custom instructions
        filled_dataset = _handle_missing_values(
            dataset=dataset, fix_nan_custom=fix_nan_custom
        )

        # Encode categorical columns if 'auto_cat_encode' is True
        if auto_cat_encode:
            cat_encoded_dataset = _cat_encoder(filled_dataset, auto_cat_encode)
            if manual_encode:
                raise MultiTrainEncodingError(
                    f"You cannot pass in a manual encoding dictionary if auto_cat_encode is set to True."
                )
            complete_dataset = cat_encoded_dataset.copy()

        # Encode columns as specified in 'manual_encode' dictionary
        if manual_encode:
            manual_encode_dataset = _manual_encoder(manual_encode, filled_dataset)
            if auto_cat_encode:
                raise MultiTrainEncodingError(
                    f"You cannot pass in a auto_cat_encode if a manual encoding dictionary is passed in."
                )
            complete_dataset = manual_encode_dataset.copy()

        # Separate features and target from the complete dataset
        data_features = complete_dataset.drop(target, axis=1)
        data_target = complete_dataset[target]

        # Split the dataset into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                data_features,
                data_target,
                test_size=test_size,
                random_state=random_state,
            )
        except ValueError as e:
            raise MultiTrainEncodingError(
                f"Ensure that the target column is encoded before splitting the dataset. \nOriginal error: {e}"
            )

        return (X_train, X_test, y_train, y_test)

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,  # must be a valid sklearn metric i.e mean_squared_error.
        show_train_score: bool = False,
        sort: str = None,
        return_best_model: Optional[str] = None,
    ):  # example 'mean_squared_error', 'r2_score', 'mean_absolute_error'
        """
        Fits multiple models to the provided training data and evaluates them using specified metrics.

        Parameters:
        - datasplits (tuple): A tuple containing four elements: X_train, X_test, y_train, y_test.
        - custom_metric (str, optional): A custom metric to evaluate the models. Must be a valid sklearn metric.
        - show_train_score (bool, optional): If True, also calculates and displays training scores.
        - sort (str, optional): Metric name to sort the final results. Examples include 'mean_squared_error', 'r2_score', etc.
        - return_best_model (str, optional): The metric to return the best model by, e.g., 'mean_squared_error'.

        Returns:
        - final_dataframe: A DataFrame containing the evaluation results of the models.
        """

        model_names, model_list, X_train, X_test, y_train, y_test = (
            _prep_model_names_list(
                datasplits,
                custom_metric,
                self.random_state,
                self.n_jobs,
                self.custom_models,
                "regression",
                self.max_iter,
            )
        )

        # Initialize progress bar for model training
        bar = trange(
            len(model_list),
            desc="Training Models",
            leave=False,
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        results = {}
        for idx in bar:
            # Update the postfix with the current model's name
            bar.set_postfix_str(f"Model: {model_names[idx]}")
            current_model = model_list[idx]

            # Fit the model and make predictions
            current_model, current_prediction, end = _fit_pred(
                current_model, model_names, idx, X_train, y_train, X_test
            )

            metric_results = {}
            # Wrap metrics in tqdm for additional progress tracking
            for metric_name, metric_func in tqdm(
                _metrics(custom_metric, 'regression').items(),
                desc=f"Evaluating {model_names[idx]}",
                leave=False,
            ):
                try:
                    if show_train_score:
                        # Calculate and store training metric
                        metric_results[f"{metric_name}_train"] = _calculate_metric(
                            metric_func,
                            y_train,
                            current_model.predict(X_train),
                        )

                    # Calculate and store test metric
                    metric_results[metric_name] = _calculate_metric(
                        metric_func,
                        y_test,
                        current_prediction,
                    )

                except ValueError as e:
                    logger.error(
                        f"Error calculating {metric_name} for {model_names[idx]}: {e}"
                    )

            # Store results for the current model
            results[model_names[idx]] = metric_results
            results[model_names[idx]].update({"Time(s)": end})

        # Display the results in a sorted DataFrame
        if custom_metric:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                custom_metric=custom_metric,
                return_best_model=return_best_model,
            )
        else:
            final_dataframe = _display_table(
                results=results, sort=sort, return_best_model=return_best_model
            )
        return final_dataframe
