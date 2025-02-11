from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from MultiTrain.utils.utils import (
    _cat_encoder,
    _metrics,
    _calculate_metric,
    _display_table,
    _fit_pred,
    _handle_missing_values,
    _manual_encoder,
    _non_auto_cat_encode_error,
    _prep_model_names_list,
)

import pandas as pd
from tqdm.notebook import trange, tqdm
from sklearn.model_selection import train_test_split
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

# Suppress all warnings at start
warnings.filterwarnings("ignore", category=Warning)

# Cache supported scalers
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
    n_jobs: int = -1
    random_state: int = 42
    custom_models: Optional[list] = None
    max_iter: int = 1000
    use_gpu: bool = False
    device: str = '0'
    
    def __post_init__(self):
        type_validations = {
            'n_jobs': (self.n_jobs, int),
            'random_state': (self.random_state, int),
            'max_iter': (self.max_iter, int),
            'use_gpu': (self.use_gpu, bool),
            'device': (self.device, str)
        }
        
        for param_name, (param_value, expected_type) in type_validations.items():
            if not isinstance(param_value, expected_type):
                raise MultiTrainTypeError(
                    f'Invalid type for {param_name}: expected {expected_type.__name__}, '
                    f'got {type(param_value).__name__}. Please provide a {expected_type.__name__} value.'
                )
                
        if not isinstance(self.custom_models, (list, type(None))):
            raise MultiTrainTypeError(
                f'Invalid type for custom_models: expected a list of custom models or None, '
                f'got {type(self.custom_models).__name__}. Please provide a list or None.'
            )

        if self.use_gpu:
            from sklearnex import patch_sklearn
            patch_sklearn(global_patch=True)
            logger.info('Device acceleration enabled')

        logger.warning('Version 1.1.1 introduces new syntax and you might experience errors if using old syntax, visit the documentation in the GitHub Repo.')
            
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

        # Load dataset
        if isinstance(data, pd.DataFrame):
            dataset = data.copy()
        elif isinstance(data, str):
            dataset = pd.read_csv(data)
        else:
            raise MultiTrainDatasetTypeError('You must either pass in a dataframe or a filepath')

        # Validate manual encoding
        if manual_encode:
            keys = list(manual_encode.keys())
            if 1 < len(keys) < 3:
                if len(keys) != len(set(keys)):
                    raise MultiTrainError('You cannot have duplicates of either "label" or "onehot" in your dictionary.')
                if any(item in manual_encode[keys[0]] for item in manual_encode[keys[1]]):
                    raise MultiTrainError('You cannot specify a column for different types of encoding')
                if fix_nan_custom and len(fix_nan_custom.keys()) != len(set(fix_nan_custom.keys())):
                    raise MultiTrainError('You cannot specify a column as a key more than once')
            if len(keys) > 2:
                raise MultiTrainError('You cannot have more than two keys, i.e., label, onehot')

        # Validate fix_nan_custom for duplicate keys
        if fix_nan_custom:
            if len(fix_nan_custom.keys()) != len(set(fix_nan_custom.keys())):
                raise MultiTrainError('You cannot specify a column as a key more than once in fix_nan_custom')

        # Handle drops
        if drop:
            if not isinstance(drop, list):
                raise MultiTrainTypeError(f"Drop parameter must be a list. Got {type(drop)}")
            dataset.drop(drop, axis=1, inplace=True)

        # Validate dataset and target
        if not isinstance(dataset, pd.DataFrame):
            raise MultiTrainDatasetTypeError(f"Dataset must be a pandas DataFrame. Got {type(dataset)}")
        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(f"Target column {target} not found in columns")

        # Process dataset
        _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode)
        filled_dataset = _handle_missing_values(dataset, fix_nan_custom)
        complete_dataset = filled_dataset.copy()

        # Handle encoding
        if auto_cat_encode and manual_encode:
            raise MultiTrainEncodingError("Cannot use both auto_cat_encode and manual_encode")
        elif auto_cat_encode:
            complete_dataset = _cat_encoder(filled_dataset, auto_cat_encode)
        elif manual_encode:
            complete_dataset = _manual_encoder(manual_encode, filled_dataset)

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                complete_dataset.drop(target, axis=1),
                complete_dataset[target],
                test_size=test_size,
                random_state=random_state
            )
        except ValueError as e:
            raise MultiTrainEncodingError(f"Target column must be encoded before splitting. Error: {e}")

        return (np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)) if self.use_gpu else (X_train, X_test, y_train, y_test)

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,  # must be a valid sklearn metric i.e mean_squared_error.
        show_train_score: bool = False,
        sort: str = None,
        pca: Union[bool, str] = False,
        return_best_model: Optional[str] = None, # example 'mean_squared_error', 'r2_score', 'mean_absolute_error'
    ):  
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
        
        # Handle PCA scaler
        if pca:
            if pca not in SUPPORTED_SCALERS:
                raise MultiTrainPCAError(f'Supported scalers are {list(SUPPORTED_SCALERS.keys())}, got {pca}')
            pca_scaler = SUPPORTED_SCALERS[pca]
        else:
            pca_scaler = False
        
        model_names, model_list, X_train, X_test, y_train, y_test = _prep_model_names_list(
            datasplits, custom_metric, self.random_state, self.n_jobs,
            self.custom_models, "regression", self.max_iter
        )

        # Initialize progress bar for model training
        bar = trange(
            len(model_list),
            desc="Training Models",
            leave=False,
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
        )

        results = {}
        for idx in bar:
            # Update the postfix with the current model's name
            bar.set_postfix_str(f"Model: {model_names[idx]}")
            current_model = model_list[idx]

            # Fit the model and make predictions
            current_model, current_prediction, end = _fit_pred(
                current_model, model_names, idx, X_train, y_train, X_test, pca_scaler
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

            
            metric_results['root_mean_squared_error'] = np.sqrt(metric_results['mean_squared_error'])
            results[model_names[idx]] = {**metric_results, "Time": end}
    
        # Display the results in a sorted DataFrame
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
    def __init__(self, n_jobs: int = -1, random_state: int = 42, custom_models: Optional[list] = None, max_iter: int = 1000, use_gpu: bool = False, device: str = '0'):
        super().__init__(n_jobs, random_state, custom_models, max_iter, use_gpu, device)
        
    def __post_init__(self):
        if not isinstance(self.use_gpu, bool):
            raise MultiTrainTypeError(f'Invalid type for use_gpu: expected bool, got {type(self.use_gpu).__name__}. Please provide a boolean value (True or False).')
        
        if not isinstance(self.device, str):
            raise MultiTrainTypeError(f'Invalid type for device: expected str, got {type(self.device).__name__}. Please provide a string value.')
        
        logging.disable()  # Disable logger warnings
        
        super().__post_init__()
