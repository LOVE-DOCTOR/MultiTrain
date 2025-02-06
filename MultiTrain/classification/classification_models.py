from dataclasses import dataclass
import random
from typing import Dict, List, Optional

from sklearn import logger

from MultiTrain.utils.utils import _calculate_metric, _cat_encoder, _check_custom_models, _display_table, _fit_pred, _handle_missing_values, _init_metrics, _manual_encoder, _models, _metrics, _non_auto_cat_encode_error

import time
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm.notebook import trange, tqdm
from IPython.display import display
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.combine import SMOTEENN, SMOTETomek

from matplotlib import pyplot as plt
from numpy.random import randint
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
    SMOTEN,
    ADASYN,
    BorderlineSMOTE,
    KMeansSMOTE,
    SVMSMOTE,
)
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)


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

from skopt import BayesSearchCV
from MultiTrain.errors.errors import *


@dataclass
class MultiClassifier:
        
    n_jobs: int = -1
    random_state: int = 42
    custom_models: list = None
    overfit_tolerance: float = 0.2

    
    
    @staticmethod
    def split(data: pd.DataFrame,
              target: str,  # Target column name
              random_state: int = 42,  # Default random state for reproducibility
              test_size: float = 0.2,  # Default test size for train-test split (80/20 split)
              auto_cat_encode: bool = False,  # If True, automatically encode all categorical columns
              manual_encode: dict = None,  # Manual encoding dictionary, e.g., {'label': ['column1'], 'onehot': ['column2']}
              fix_nan_custom: Optional[Dict] = False,  # Custom NaN handling, e.g., {'column1': 'ffill'}
              drop: list = None):  # List of columns to drop, e.g., ['column1', 'column2']
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

        # Drop specified columns if 'drop' parameter is provided
        if drop:
            if type(drop) != list:
                raise MultiTrainTypeError(f'You need to pass in a list of columns to drop. Got {type(drop)}')
            dataset.drop(drop, axis=1, inplace=True)

        # Ensure the dataset is a pandas DataFrame
        if type(dataset) != pd.DataFrame:
            raise MultiTrainDatasetTypeError(f'You need to pass in a Dataset of type {pd.DataFrame}. Got {type(dataset)}')

        # Check if the target column exists in the dataset
        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(f'Target column {target} not found in list of columns. Please pass in a target column.')

        # Check for categorical columns and raise an error if necessary
        _non_auto_cat_encode_error(dataset=dataset, auto_cat_encode=auto_cat_encode, manual_encode=manual_encode)

        # Handle missing values in the dataset using custom instructions
        filled_dataset = _handle_missing_values(dataset=dataset, fix_nan_custom=fix_nan_custom)

        # Encode categorical columns if 'auto_cat_encode' is True
        if auto_cat_encode:
            cat_encoded_dataset = _cat_encoder(filled_dataset, auto_cat_encode)
            if manual_encode:
                raise MultiTrainEncodingError(f'You cannot pass in a manual encoding dictionary if auto_cat_encode is set to True.')
            complete_dataset = cat_encoded_dataset.copy()

        # Encode columns as specified in 'manual_encode' dictionary
        if manual_encode:
            manual_encode_dataset = _manual_encoder(manual_encode, filled_dataset)
            if auto_cat_encode:
                raise MultiTrainEncodingError(f'You cannot pass in a auto_cat_encode if a manual encoding dictionary is passed in.')
            complete_dataset = manual_encode_dataset.copy()

        # Separate features and target from the complete dataset
        data_features = complete_dataset.drop(target, axis=1)
        data_target = complete_dataset[target]

        # Split the dataset into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(data_features, 
                                                                data_target, 
                                                                test_size=test_size, 
                                                                random_state=random_state)
        except ValueError as e:
            raise MultiTrainEncodingError(f'Ensure that the target column is encoded before splitting the dataset. \nOriginal error: {e}')
        
        return (X_train, X_test, y_train, y_test)

    

    def fit(self,
            datasplits: tuple,
            custom_metric: str = None, # must be a valid sklearn metric i.e accuracy_score. 
            show_train_score: bool = False,
            imbalanced=False,
            sort: str = None): # example 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy'
        """
        Fits multiple models to the provided training data and evaluates them using specified metrics.

        Parameters:
        - datasplits (tuple): A tuple containing four elements: X_train, X_test, y_train, y_test.
        - custom_metric (str, optional): A custom metric to evaluate the models. Must be a valid sklearn metric.
        - show_train_score (bool, optional): If True, also calculates and displays training scores.
        - imbalanced (bool, optional): If True, uses 'micro' average for precision, recall, and f1 metrics.
        - sort (str, optional): Metric name to sort the final results. Examples include 'accuracy', 'precision', etc.

        Returns:
        - final_dataframe: A DataFrame containing the evaluation results of the models.
        """
        
        # Validate the datasplits parameter
        if type(datasplits) != tuple or len(datasplits) != 4:
            raise MultiTrainSplitError('The "datasplits" parameter can only be of type tuple and must have 4 values. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.')

        # Unpack the datasplits tuple
        X_train, X_test, y_train, y_test = datasplits[0], datasplits[1], datasplits[2], datasplits[3]

        # Check if the custom metric is already in the default metrics
        if custom_metric in _init_metrics():
            raise MultiTrainMetricError(f'Metric already exists. Please pass in a different metric.'
                                        f'Default metrics are: {_init_metrics()}')
        
        # Initialize models
        models = _models(random_state=self.random_state, n_jobs=self.n_jobs)

        results = {}

        # Check for custom models and get model names and list
        model_names, model_list = _check_custom_models(self.custom_models, models)
        
        # Initialize progress bar for model training
        bar = trange(len(model_list),
             desc="Training Models",
             leave=False,
             bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        results = {}
        for idx in bar:
            # Update the postfix with the current model's name
            bar.set_postfix_str(f"Model: {model_names[idx]}")
            current_model = model_list[idx]
            
            # Fit the model and make predictions
            current_model, current_prediction, end = _fit_pred(current_model, 
                                                               model_names, 
                                                               idx, 
                                                               X_train, 
                                                               y_train,
                                                               X_test)
            
            metric_results = {}
            avg_metrics = ['precision', 'recall', 'f1']
            # Wrap metrics in tqdm for additional progress tracking
            for metric_name, metric_func in tqdm(_metrics(custom_metric).items(), desc=f"Evaluating {model_names[idx]}", leave=False):
                try:
                    # Determine the average type based on imbalanced flag and metric type
                    average_type = 'micro' if imbalanced and metric_name in avg_metrics else 'binary'
                    
                    if show_train_score:
                        # Calculate and store training metric
                        metric_results[f"{metric_name}_train"] = _calculate_metric(metric_func, y_train, current_model.predict(X_train), average_type if metric_name in avg_metrics else None)
                    
                    # Calculate and store test metric
                    metric_results[metric_name] = _calculate_metric(metric_func, y_test, current_prediction, average_type if metric_name in avg_metrics else None)
                    
                except ValueError as e:
                    # Handle multiclass target error by using 'weighted' average
                    if "Target is multiclass but average='binary'" in str(e):
                        metric_results[metric_name] = metric_func(y_test, current_prediction, average='weighted')
                
            # Store results for the current model
            results[model_names[idx]] = metric_results
            results[model_names[idx]].update({'Time(s)': end})

        # Display the results in a sorted DataFrame
        if custom_metric:
            final_dataframe = _display_table(results, sort, custom_metric)
        else:
            final_dataframe = _display_table(results, sort)
        return final_dataframe

