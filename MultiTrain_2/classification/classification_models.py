from dataclasses import dataclass
import random
from typing import Dict, List, Optional

from sklearn import logger

from MultiTrain_2.utils.utils import _cat_encoder, _check_custom_models, _display_table, _fit_pred, _handle_missing_values, _init_metrics, _manual_encoder, _models, _metrics, _non_auto_cat_encode_error

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
from MultiTrain_2.errors.errors import *


@dataclass
class MultiClassifier:
        
    n_jobs: int = -1
    random_state: int = 42
    custom_models: list = None

    
    def __call__(self):
        return self
    
    @staticmethod
    def split(data: pd.DataFrame,
              target: str, # target column
              random_state: int = 42, # default random state
              test_size: float = 0.2, # default 80/20 split
              auto_cat_encode: bool = False, # if True, will encode all categorical columns
              manual_encode: dict = None, # example {'label': ['column1', 'column2'], 'onehot': ['column3', 'column4']}
              encode_column_subset: Optional[List] = None,
              fix_nan_custom: Optional[Dict] = False, # example {'column1': 'ffill', 'column2': 'bfill', 'column3': ['value']}
              drop: list = None): # example ['column1', 'column2']
              
    

        dataset = data.copy()

        if drop:
            if type(drop) != list:
                raise MultiTrainTypeError(f'You need to pass in a list of columns to drop. Got {type(drop)}')
            
            dataset.drop(drop, axis=1, inplace=True)
        
        if type(dataset) != pd.DataFrame:
            raise MultiTrainDatasetTypeError(f'You need to pass in a Dataset of type {pd.DataFrame}. Got {type(dataset)}')
        

        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(f'Target column {target} not found in list of columns. Please pass in a target column.')
        

        
        # Check for the presence of categorical columns and raise error
        # if there are categorical columns and auto_cat_encode is set to False and 
        # no manual encoding dictionary is passed in
        _non_auto_cat_encode_error(dataset=dataset, auto_cat_encode=auto_cat_encode, manual_encode=manual_encode)

        # Handle missing values in the dataset
        filled_dataset = _handle_missing_values(dataset=dataset, fix_nan_custom=fix_nan_custom)

        # If auto_cat_encode is True, encode all categorical columns
        # Also checks if manual_encode is passed in simultaneously and raises an error
        if auto_cat_encode:
            cat_encoded_dataset = _cat_encoder(filled_dataset, auto_cat_encode)
            if manual_encode:
                raise MultiTrainEncodingError(f'You cannot pass in a manual encoding dictionary if auto_cat_encode is set to True.')

            complete_dataset = cat_encoded_dataset.copy()

        # If manual_encode is passed in, encode the columns as specified in the dictionary
        # Also checks if auto_cat_encode is passed in simultaneously and raises an error            
        if manual_encode:
            manual_encode_dataset = _manual_encoder(manual_encode, filled_dataset)
            if auto_cat_encode:
                raise MultiTrainEncodingError(f'You cannot pass in a auto_cat_encode if a manual encoding dictionary is passed in.')

            complete_dataset = manual_encode_dataset.copy()

        data_features = complete_dataset.drop(target, axis=1)

        data_target = complete_dataset[target]


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
        
        if type(datasplits) != tuple or len(datasplits) != 4:
            raise MultiTrainSplitError('The "datasplits" parameter can only be of type tuple and must have 4 values. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.')
        # if isinstance(datasplits[0]) != pd.DataFrame or isinstance(datasplits[1]) != pd.DataFrame:
        #     raise MultiTrainSplitError('The "datasplits" parameter can only contain pandas DataFrames and pandas Series. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.')
        # if isinstance(datasplits[2]) != pd.DataFrame or isinstance(datasplits[3]) != pd.DataFrame:
        #     raise MultiTrainSplitError('The "datasplits" parameter can only contain pandas DataFrames and pandas Series. Ensure that you are passing in the result of the split function into the datasplits parameter for it to function properly.')
        

        X_train, X_test, y_train, y_test = datasplits[0], datasplits[1], datasplits[2], datasplits[3]
        if custom_metric in _init_metrics():
            raise MultiTrainMetricError(f'Metric already exists. Please pass in a different metric.'
                                        f'Default metrics are: {_init_metrics()}')
            
        models = _models(random_state=self.random_state, n_jobs=self.n_jobs)

        results = {}

        model_names, model_list = _check_custom_models(self.custom_models, models)
            
        bar = trange(len(model_list),
             desc="Training in progress: ",
             leave=False,
             bar_format="{desc}{percentage:3.0f}% {bar}{remaining} [{n_fmt}/{total_fmt} {postfix}]")

        results = {}
        for idx in bar:
            # Update the postfix with the current model's name
            bar.set_postfix({"Model": model_names[idx]})
            current_model = model_list[idx]
            
            current_model, current_prediction, end = _fit_pred(current_model, 
                                                               model_names, 
                                                               idx, 
                                                               X_train, 
                                                               y_train,
                                                               X_test)
            
            metric_results = {}
            # Wrap metrics in tqdm for additional progress tracking
            for metric_name, metric_attr in tqdm(_metrics(custom_metric).items(), desc=f"Evaluating {model_names[idx]}", leave=False):
                current_metric = metric_attr
                try:
                    if imbalanced is False:
                        if show_train_score is True:
                            metric_results[metric_name + '_train'] = current_metric(y_train, current_model.predict(X_train))
                            metric_results[metric_name] = current_metric(y_test, current_prediction)
                    elif imbalanced is True and metric_name in ['precision', 'recall', 'f1']:
                        if show_train_score is True:
                            metric_results[metric_name + '_train'] = current_metric(y_train, current_model.predict(X_train), average='micro')
                            metric_results[metric_name] = current_metric(y_test, current_prediction, average='micro')
                        
                except ValueError as e:
                    if "Target is multiclass but average='binary'" in str(e):
                        metric_results[metric_name] = current_metric(y_test, current_prediction, average='weighted')
                
            results[model_names[idx]] = metric_results
            results[model_names[idx]].update({'Time(s)': end})

        if custom_metric:
            final_dataframe = _display_table(results, sort, custom_metric)
        else:
            final_dataframe = _display_table(results, sort)
        return final_dataframe


