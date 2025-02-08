from dataclasses import dataclass
from typing import Dict, Optional
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

# Suppress all sklearn warnings
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
    _fit_pred_text,
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

warnings.filterwarnings("ignore", category=Warning)


@dataclass
class MultiClassifier:
    n_jobs: int = -1
    random_state: int = 42
    custom_models: Optional[list] = None
    max_iter: int = 1000
    use_gpu: bool = False
    device: str = '0'
    
    def __post_init__(self):
        if not isinstance(self.n_jobs, int):
            raise MultiTrainTypeError(f'Invalid type for n_jobs: expected int, got {type(self.n_jobs).__name__}. Please provide an integer value.')
        
        if not isinstance(self.random_state, int):
            raise MultiTrainTypeError(f'Invalid type for random_state: expected int, got {type(self.random_state).__name__}. Please provide an integer value.')
        
        if not isinstance(self.custom_models, (list, type(None))):
            raise MultiTrainTypeError(f'Invalid type for custom_models: expected a list of custom models (check sklearn for the valid model names) or None, got {type(self.custom_models).__name__}. Please provide a list or None.')
        
        if not isinstance(self.max_iter, int):
            raise MultiTrainTypeError(f'Invalid type for max_iter: expected int, got {type(self.max_iter).__name__}. Please provide an integer value.')
        
        if not isinstance(self.use_gpu, bool):
            raise MultiTrainTypeError(f'Invalid type for use_gpu: expected bool, got {type(self.use_gpu).__name__}. Please provide a boolean value (True or False).')
        
        if not isinstance(self.device, str):
            raise MultiTrainTypeError(f'Invalid type for device: expected str, got {type(self.device).__name__}. Please provide a string value.')
        
        if self.use_gpu:
            from sklearnex import patch_sklearn
            patch_sklearn(global_patch=True)
            logger.info('Device acceleration enabled')

        logger.warning('Version 1.1.1 introduces new syntax and you might experience errors if using old syntax, visit the documentation in the GitHub Repo.')
            
    def split(
        self,
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
            if len(keys) > 2:
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

        return (np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)) if self.use_gpu else (X_train, X_test, y_train, y_test)

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,  # must be a valid sklearn metric i.e accuracy_score.
        show_train_score: bool = False,
        imbalanced: bool = False,
        sort: str = None,
        text: bool = False,
        vectorizer: str = None,  # example: count or tfidf
        pipeline_dict: dict = None,  # example: {'ngram_range': (1, 2), 'encoding': 'utf-8', 'max_features': 5000, 'analyzer': 'word'}
        return_best_model: Optional[str] = None,
    ):  # example 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy'
        """
        Fits multiple models to the provided training data and evaluates them using specified metrics.

        Parameters:
        - datasplits (tuple): A tuple containing four elements: X_train, X_test, y_train, y_test.
        - custom_metric (str, optional): A custom metric to evaluate the models. Must be a valid sklearn metric.
        - show_train_score (bool, optional): If True, also calculates and displays training scores.
        - imbalanced (bool, optional): If True, uses 'micro' average for precision, recall, and f1 metrics.
        - sort (str, optional): Metric name to sort the final results. Examples include 'accuracy', 'precision', etc.
        - return_best_model (str, optional): The metric to return the best model by, e.g., 'accuracy'.

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
                "classification",
                self.max_iter
            )
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

            if text is False:
                if pipeline_dict:
                    raise MultiTrainTextError(
                        "You cannot use pipeline_dict when the text parameter is False"
                    )
                # Fit the model and make predictions
                current_model, current_prediction, end = _fit_pred(
                    current_model, model_names, idx, X_train, y_train, X_test
                )
            elif text is True:
                if not pipeline_dict:
                    raise MultiTrainTextError(
                        "You must pass a dictionary with the following keys if you set text to True\n"
                        "ngram_range, encoding, max_features, analyzer. These keys are found in CountVectorizer or TfidfVectorizer"
                    )
                if not vectorizer:
                    raise MultiTrainTextError(
                        'You must pass one of "count" or "tfidf" to the vectorizer argument when using text=True'
                    )

                current_model, current_prediction, end = _fit_pred_text(
                    vectorizer, pipeline_dict, current_model, X_train, y_train, X_test
                )

            metric_results = {}
            avg_metrics = ["precision", "recall", "f1"]
            # Wrap metrics in tqdm for additional progress tracking
            for metric_name, metric_func in tqdm(
                _metrics(custom_metric, "classification").items(),
                desc=f"Evaluating {model_names[idx]}",
                leave=False,
            ):
                try:
                    # Determine the average type based on imbalanced flag and metric type
                    average_type = (
                        "micro"
                        if imbalanced and metric_name in avg_metrics
                        else "binary"
                    )

                    if show_train_score:
                        # Calculate and store training metric
                        metric_results[f"{metric_name}_train"] = _calculate_metric(
                            metric_func,
                            y_train,
                            current_model.predict(X_train),
                            average_type if metric_name in avg_metrics else None,
                        )

                    # Calculate and store test metric
                    metric_results[metric_name] = _calculate_metric(
                        metric_func,
                        y_test,
                        current_prediction,
                        average_type if metric_name in avg_metrics else None,
                    )

                except ValueError as e:
                    # Handle multiclass target error by using 'weighted' average
                    if "Target is multiclass but average='binary'" in str(e):
                        metric_results[metric_name] = metric_func(
                            y_test, current_prediction, average="weighted"
                        )

            # Store results for the current model
            results[model_names[idx]] = metric_results
            results[model_names[idx]].update({"Time": end})
            
        # Display the results in a sorted DataFrame
        if custom_metric:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                custom_metric=custom_metric,
                return_best_model=return_best_model,
                task="classification",
            )
        else:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                return_best_model=return_best_model,
                task="classification",
            )
        return final_dataframe

@dataclass
class subMultiClassifier(MultiClassifier):
    def __init__(self, n_jobs: int = -1, random_state: int = 42, custom_models: Optional[list] = None, max_iter: int = 1000, use_gpu: bool = False, device: str = '0'):
        super().__init__(n_jobs, random_state, custom_models, max_iter, use_gpu, device)
        
    def __post_init__(self):
        if not isinstance(self.use_gpu, bool):
            raise MultiTrainTypeError(f'Invalid type for use_gpu: expected bool, got {type(self.use_gpu).__name__}. Please provide a boolean value (True or False).')
        
        if not isinstance(self.device, str):
            raise MultiTrainTypeError(f'Invalid type for device: expected str, got {type(self.device).__name__}. Please provide a string value.')
        
        logging.disable()  # Disable logger warnings
        
        return super().__post_init__()