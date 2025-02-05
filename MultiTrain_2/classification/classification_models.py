from typing import Dict, List, Optional
from MultiTrain_2.utils.utils import _cat_encoder, _handle_missing_values, _manual_encoder, _models, _metrics, _non_auto_cat_encode_error

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm.notebook import trange
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

class MultiClassifier:
    
    def __init__(self, 
                 n_jobs: Optional[int] = -1, 
                 random_state: int = 42,
                 ):
        
        self.n_jobs = n_jobs
        self.random_state = random_state

    
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
              fix_nan: Optional[List] = False, # example [bool, strategy] i.e [True, 'ffill'], [True, 'bfill'], [True, 'interpolate']
              fix_nan_custom: Optional[Dict] = False): # example {'column1': 'ffill', 'column2': 'bfill', 'column3': ['value']} 
    

        dataset = data.copy()
        
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
            cat_encoded_dataset = _cat_encoder(filled_dataset, auto_cat_encode, encode_column_subset)
            if manual_encode:
                raise MultiTrainEncodingError(f'You cannot pass in a manual encoding dictionary if auto_cat_encode is set to True.')

            complete_dataset = cat_encoded_dataset.copy()

        # If manual_encode is passed in, encode the columns as specified in the dictionary
        # Also checks if auto_cat_encode is passed in simultaneously and raises an error            
        if manual_encode:
            manual_encode_dataset = _manual_encoder(manual_encode, filled_dataset, encode_column_subset)
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
            X_train: pd.DataFrame, 
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series):
        
        models = _models(random_state=self.random_state, n_jobs=self.n_jobs)

        results = {}
        for model_name, model_attr in models.items():
            current_model = model_attr
            current_model.fit(X_train, y_train)
            current_prediction = current_model.predict(X_test)
            
            metric_results = {}
            for metric_name, metric_attr in _metrics().items():
                current_metric = metric_attr
                metric_results[metric_name] = current_metric(y_test, current_prediction)
                
            results[model_name] = metric_results

        print(results)

