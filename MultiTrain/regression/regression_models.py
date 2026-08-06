from dataclasses import dataclass
from numbers import Real
import platform
from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from MultiTrain.utils.utils import (
    _cat_encoder,
    _metrics,
    _calculate_metric,
    _display_table,
    _handle_missing_values,
    _manual_encoder,
    _non_auto_cat_encode_error,
    _prep_model_names_list,
    _prepare_train_test,
    _validate_datasplits,
    _validate_supervised_dataset,
)
from MultiTrain.utils.execution import prepare_tabular_features, run_models

import pandas as pd
from sklearn.model_selection import train_test_split
from MultiTrain.errors.errors import *

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Keep the accepted scaler names in one place so validation and pipeline setup agree.
SUPPORTED_SCALERS = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'MaxAbsScaler': MaxAbsScaler(), 
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer(),
    'QuantileTransformer': QuantileTransformer(),
    'PowerTransformer': PowerTransformer()
}

@dataclass
class MultiRegressor:
    n_jobs: int = 1
    random_state: int = 42
    custom_models: Optional[list] = None
    max_iter: int = 1000
    use_gpu: bool = False
    device: str = '0'
    model_workers: Optional[int] = None
    
    def __post_init__(self):
        type_validations = {
            'n_jobs': (self.n_jobs, int),
            'random_state': (self.random_state, int),
            'max_iter': (self.max_iter, int),
            'use_gpu': (self.use_gpu, bool),
            'device': (self.device, str)
        }
        
        for param_name, (param_value, expected_type) in type_validations.items():
            invalid_int = expected_type is int and (
                not isinstance(param_value, int) or isinstance(param_value, bool)
            )
            if invalid_int or (
                expected_type is not int and not isinstance(param_value, expected_type)
            ):
                raise MultiTrainTypeError(
                    f'Invalid type for {param_name}: expected {expected_type.__name__}, '
                    f'got {type(param_value).__name__}. Please provide a {expected_type.__name__} value.'
                )
                
        if not isinstance(self.custom_models, (list, type(None))):
            raise MultiTrainTypeError(
                f'Invalid type for custom_models: expected a list of custom models or None, '
                f'got {type(self.custom_models).__name__}. Please provide a list or None.'
            )
        if self.custom_models is not None and not all(
            isinstance(model, str) for model in self.custom_models
        ):
            raise MultiTrainTypeError("Every custom model name must be a string")
        if self.n_jobs == 0:
            raise MultiTrainTypeError("n_jobs cannot be zero")
        if self.model_workers is not None and (
            not isinstance(self.model_workers, int)
            or isinstance(self.model_workers, bool)
            or self.model_workers == 0
            or self.model_workers < -1
        ):
            raise MultiTrainTypeError(
                "model_workers must be None, -1, or a positive integer"
            )
        if self.max_iter <= 0:
            raise MultiTrainTypeError("max_iter must be positive")
        if not self.device:
            raise MultiTrainTypeError("device cannot be empty")

        logger.debug('MultiTrain regressor initialized')
            
    def split(
        self,
        data: Union[pd.DataFrame, str],
        target: str,  # Name of the target column
        random_state: int = 42,  # Random state for reproducibility
        test_size: float = 0.2,  # Proportion of the dataset for the test split
        auto_cat_encode: bool = False,  # Automatically encode all categorical columns if True
        manual_encode: dict = None,  # Manual encoding dictionary, e.g., {'label': ['column1'], 'onehot': ['column2']}
        fix_nan_custom: Optional[
            Dict
        ] = False,  # Custom NaN handling, e.g., {'column1': 'ffill'}
        drop: list = None, # List of columns to drop, e.g., ['column1', 'column2']
    ):  
        """
        Splits the dataset into training and testing sets after performing optional preprocessing steps.

        Parameters:
        - data (Union[pd.DataFrame, str]): The input dataset or a filepath to a dataset.
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

        if not isinstance(target, str) or not target:
            raise MultiTrainTypeError("target must be a non-empty string")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise MultiTrainTypeError("random_state must be an integer")
        if not isinstance(test_size, Real) or isinstance(test_size, bool):
            raise MultiTrainTypeError("test_size must be numeric")
        if not 0 < test_size < 1:
            raise MultiTrainSplitError("test_size must be between 0 and 1")
        if not isinstance(auto_cat_encode, bool):
            raise MultiTrainTypeError("auto_cat_encode must be a boolean")

        # Normalize file paths and dataframes into the same in-memory representation.
        if isinstance(data, pd.DataFrame):
            dataset = data.copy()
        elif isinstance(data, str):
            dataset = pd.read_csv(data)
        else:
            raise MultiTrainDatasetTypeError('You must either pass in a dataframe or a filepath')

        # Fail before modifying the dataset when preprocessing instructions are malformed.
        if manual_encode is not None and not isinstance(manual_encode, dict):
            raise MultiTrainTypeError(
                f"manual_encode must be a dictionary or None. Got {type(manual_encode)}"
            )
        if (
            fix_nan_custom is not False
            and fix_nan_custom is not None
            and not isinstance(fix_nan_custom, dict)
        ):
            raise MultiTrainTypeError(
                f"fix_nan_custom must be a dictionary. Got {type(fix_nan_custom)}"
            )

        # A column needs one unambiguous encoding strategy.
        if manual_encode:
            invalid_keys = set(manual_encode) - {"label", "onehot"}
            if invalid_keys:
                raise MultiTrainEncodingError(
                    f"Unsupported encoding types: {sorted(invalid_keys)}"
                )
            for encoding_type, columns in manual_encode.items():
                if not isinstance(columns, (list, tuple)):
                    raise MultiTrainTypeError(
                        f"Columns for {encoding_type} must be a list or tuple"
                    )
            overlap = set(manual_encode.get("label", [])) & set(
                manual_encode.get("onehot", [])
            )
            if overlap:
                raise MultiTrainEncodingError(
                    f"Columns cannot use multiple encodings: {sorted(overlap)}"
                )

        if auto_cat_encode and manual_encode:
            raise MultiTrainEncodingError("Cannot use both auto_cat_encode and manual_encode")
        if manual_encode and target in manual_encode.get("onehot", []):
            raise MultiTrainEncodingError("The target column cannot be one-hot encoded")

        # Remove ignored features before checking the columns used for training.
        if drop is not None and not isinstance(drop, list):
            raise MultiTrainTypeError(f"Drop parameter must be a list. Got {type(drop)}")
        if drop:
            missing_drop_columns = [column for column in drop if column not in dataset]
            if missing_drop_columns:
                raise MultiTrainColumnMissingError(
                    f"Columns to drop were not found: {missing_drop_columns}"
                )
            dataset.drop(drop, axis=1, inplace=True)

        # The target must still exist after optional columns have been dropped.
        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(f"Target column {target} not found in columns")

        _validate_supervised_dataset(dataset, target, "regression")

        _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode)
        # Split first so encoders and missing-value rules cannot learn from held-out rows.
        try:
            train_dataset, test_dataset = train_test_split(
                dataset,
                test_size=test_size,
                random_state=random_state
            )
        except ValueError as e:
            raise MultiTrainSplitError(f"Unable to split the dataset: {e}") from e

        train_dataset, test_dataset = _prepare_train_test(
            train_dataset,
            test_dataset,
            auto_cat_encode=auto_cat_encode,
            manual_encode=manual_encode,
            fix_nan_custom=fix_nan_custom,
        )
        X_train = train_dataset.drop(target, axis=1)
        X_test = test_dataset.drop(target, axis=1)
        y_train = train_dataset[target]
        y_test = test_dataset[target]

        return (np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)) if self.use_gpu else (X_train, X_test, y_train, y_test)

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,  # must be a valid sklearn metric i.e mean_squared_error.
        show_train_score: bool = False,
        sort: str = None,
        pca: Union[bool, str] = False,
        return_best_model: Optional[str] = None, # example 'mean_squared_error', 'r2_score', 'mean_absolute_error'
        n_components: Optional[Union[int, float]] = None,
    ):  
        """
        Fits multiple models to the provided training data and evaluates them using specified metrics.

        Parameters:
        - datasplits (tuple): A tuple containing four elements: X_train, X_test, y_train, y_test.
        - custom_metric (str, optional): A custom metric to evaluate the models. Must be a valid sklearn metric.
        - show_train_score (bool, optional): If True, also calculates and displays training scores.
        - sort (str, optional): Metric name to sort the final results. Examples include 'mean_squared_error', 'r2_score', etc.
        - pca (bool or str, optional): Scaler to apply before the shared PCA transformation.
        - return_best_model (str, optional): The metric to return the best model by, e.g., 'mean_squared_error'.
        - n_components (int or float, optional): Component count or explained-variance target for PCA.

        Returns:
        - final_dataframe: A DataFrame containing measurements for every selected model.
        """
        
        if custom_metric is not None and not isinstance(custom_metric, str):
            raise MultiTrainTypeError("custom_metric must be a string or None")
        if not isinstance(show_train_score, bool):
            raise MultiTrainTypeError("show_train_score must be a boolean")
        if sort is not None and not isinstance(sort, str):
            raise MultiTrainTypeError("sort must be a string or None")
        if pca is not False and not isinstance(pca, str):
            raise MultiTrainPCAError(
                "pca must be False or the name of a supported scaler"
            )
        if return_best_model is not None and not isinstance(return_best_model, str):
            raise MultiTrainTypeError("return_best_model must be a string or None")

        _validate_datasplits(datasplits, "regression")

        # The historical pca argument selects the scaler used before PCA.
        if pca:
            if pca not in SUPPORTED_SCALERS:
                raise MultiTrainPCAError(f'Supported scalers are {list(SUPPORTED_SCALERS.keys())}, got {pca}')
            pca_scaler = SUPPORTED_SCALERS[pca]
        else:
            pca_scaler = False
        
        gpu_enabled = self.use_gpu and platform.system() != "Darwin"
        model_names, model_list, X_train, X_test, y_train, y_test = _prep_model_names_list(
            datasplits, custom_metric, self.random_state, self.n_jobs,
            self.custom_models, "regression", self.max_iter,
            gpu_enabled, self.device,
        )

        prepared_train, prepared_test = prepare_tabular_features(
            X_train, X_test, pca_scaler, n_components
        )
        completed = run_models(
            model_names,
            model_list,
            prepared_train,
            y_train,
            prepared_test,
            y_test,
            show_train_score,
            "regression",
            self.model_workers,
            self.n_jobs,
            use_gpu=gpu_enabled,
        )

        results = {}
        for completed_model in completed:
            metric_results = {}
            for metric_name, metric_func in _metrics(
                custom_metric, "regression"
            ).items():
                if show_train_score:
                    metric_results[f"{metric_name}_train"] = _calculate_metric(
                        metric_func,
                        y_train,
                        completed_model.train_prediction,
                    )
                metric_results[metric_name] = _calculate_metric(
                    metric_func,
                    y_test,
                    completed_model.test_prediction,
                )

            if show_train_score:
                metric_results["root_mean_squared_error_train"] = np.sqrt(
                    metric_results["mean_squared_error_train"]
                )
            metric_results["root_mean_squared_error"] = np.sqrt(
                metric_results["mean_squared_error"]
            )
            results[completed_model.name] = {
                **metric_results,
                "Time": completed_model.elapsed,
            }
    
        # Format and rank the accumulated model results only after every fit finishes.
        if custom_metric:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                custom_metric=custom_metric,
                return_best_model=return_best_model,
                task='regression'
            )
        else:
            final_dataframe = _display_table(
                results=results, 
                sort=sort, 
                return_best_model=return_best_model,
                task='regression'
            )
        return final_dataframe
    

@dataclass
class subMultiRegressor(MultiRegressor):
    def __init__(self, n_jobs: int = 1, random_state: int = 42, custom_models: Optional[list] = None, max_iter: int = 1000, use_gpu: bool = False, device: str = '0', model_workers: Optional[int] = None):
        super().__init__(
            n_jobs=n_jobs,
            random_state=random_state,
            custom_models=custom_models,
            max_iter=max_iter,
            use_gpu=use_gpu,
            device=device,
            model_workers=model_workers,
        )
        
    def __post_init__(self):
        if not isinstance(self.use_gpu, bool):
            raise MultiTrainTypeError(f'Invalid type for use_gpu: expected bool, got {type(self.use_gpu).__name__}. Please provide a boolean value (True or False).')
        
        if not isinstance(self.device, str):
            raise MultiTrainTypeError(f'Invalid type for device: expected str, got {type(self.device).__name__}. Please provide a string value.')
        
        super().__post_init__()
