import inspect
import time
from typing import Dict, List, Optional
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, Perceptron
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from MultiTrain_2.errors.errors import *
from sklearn.metrics import *

def _models(random_state, n_jobs):
    """
    Returns a dictionary of classifier models from various libraries.

    Each key is a string representing the name of the classifier, and the value is an instance of the classifier.

    Args:
        random_state (int): Seed used by the random number generator.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        dict: A dictionary of classifier models.
    """
    return {
        "LogisticRegression": LogisticRegression(random_state=random_state, n_jobs=n_jobs),
        "LogisticRegressionCV": LogisticRegressionCV(n_jobs=n_jobs),
        "SGDClassifier": SGDClassifier(n_jobs=n_jobs),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(n_jobs=n_jobs),
        "RidgeClassifier": RidgeClassifier(),
        "RidgeClassifierCV": RidgeClassifierCV(),
        "Perceptron": Perceptron(n_jobs=n_jobs),
        "LinearSVC": LinearSVC(random_state=random_state),
        "NuSVC": NuSVC(random_state=random_state),
        "SVC": SVC(random_state=random_state),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=n_jobs),
        "MLPClassifier": MLPClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
        "ExtraTreeClassifier": ExtraTreeClassifier(random_state=random_state),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs),
        "BaggingClassifier": BaggingClassifier(random_state=random_state, n_jobs=n_jobs),
        "CatBoostClassifier": CatBoostClassifier(random_state=random_state, thread_count=n_jobs, silent=True),
        "RandomForestClassifier": RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=random_state),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=random_state),
        "LGBMClassifier": LGBMClassifier(random_state=random_state, n_jobs=n_jobs, verbose=-1),
        "XGBClassifier": XGBClassifier(random_state=random_state, n_jobs=n_jobs, verbosity=0, verbose=False),
    }

def _init_metrics():
    """
    Initializes a list of default metric names.

    Returns:
        list: A list of metric names.
    """
    return ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score', 'balanced_accuracy_score']

def _metrics(custom_metric):
    """
    Returns a dictionary of metric functions from sklearn.

    Each key is a string representing the name of the metric, and the value is the metric function.

    Args:
        custom_metric (str): Name of a custom metric to include.

    Returns:
        dict: A dictionary of metric functions.
    """
    if custom_metric:
        # Check if the custom metric is a valid sklearn metric
        if custom_metric not in [name for name, obj in inspect.getmembers(sklearn.metrics, inspect.isfunction)]:
            raise MultiTrainMetricError(f'Custom metric ({custom_metric}) is not a valid metric. Please check the sklearn documentation for a valid list of metrics.')
        return {
            custom_metric: globals().get(custom_metric),
            "precision": precision_score,
            "recall": recall_score,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        }
    else:
        return {
            "precision": precision_score,
            "recall": recall_score,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        }

def _cat_encoder(cat_data, auto_cat_encode, encode_subset):
    """
    Encodes categorical columns in the dataset using Label Encoding.

    Args:
        cat_data (pd.DataFrame): The dataset containing categorical data.
        auto_cat_encode (bool): If True, automatically encodes all categorical columns.
        encode_subset (list): List of specific columns to encode if auto_cat_encode is False.

    Returns:
        pd.DataFrame: The dataset with encoded categorical columns.
    """
    dataset = cat_data.copy()
    cat_columns = list(dataset.select_dtypes(include=['object', 'category']).columns)

    if auto_cat_encode is True:
        le = LabelEncoder()
        dataset[cat_columns] = dataset[cat_columns].astype(str).apply(le.fit_transform)
        return dataset
    else:
        # Raise an error if columns are not encoded
        raise MultiTrainEncodingError(f"Ensure that all columns are encoded before splitting the dataset. Set " 
                                "encode to True or pass in a list of columns to encode with the encode_subset parameter.")
        
def _manual_encoder(manual_encode, dataset):
    """
    Manually encodes specified columns in the dataset using specified encoding types.

    Args:
        manual_encode (dict): Dictionary specifying encoding types and columns.
        dataset (pd.DataFrame): The dataset to encode.

    Returns:
        pd.DataFrame: The dataset with manually encoded columns.
    """
    encoder_types = ['label', 'onehot']
     
    for encode_type, encode_columns in manual_encode.items():
        if encode_type not in encoder_types:
            raise MultiTrainEncodingError(f'Encoding type {encode_type} not found. Use one of the following: {encoder_types}')

    if manual_encode['label']:
        le = LabelEncoder()
        dataset[manual_encode['label']] = dataset[manual_encode['label']].apply(le.fit_transform)
                
    if manual_encode['onehot']:
        encode_columns = manual_encode['onehot']
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
        if dataset[i].dtype == 'object':
            if auto_cat_encode is False and manual_encode is None:
                raise MultiTrainEncodingError(f"Ensure that all columns are encoded before splitting the dataset. Column '{i}' is not encoded.\nSet " 
                                        "auto_cat_encode to True or pass in a manual encoding dictionary.")

def _fill_missing_values(dataset, column):
    """
    Fills missing values in a specified column of the dataset.

    Args:
        dataset (pd.DataFrame): The dataset containing missing values.
        column (str): The column to fill missing values in.

    Returns:
        pd.Series: The column with missing values filled.
    """
    if dataset[column].dtype in ['object', 'category']:
        mode_val = dataset[column].mode()[0]
        dataset[column] = dataset[column].fillna(mode_val)  
        return dataset[column]
    else:
        dataset[column] = dataset[column].fillna(0)
        return dataset[column]

def _handle_missing_values(dataset: pd.DataFrame, 
                           fix_nan_custom: Optional[Dict] = False) -> pd.DataFrame: 
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
            raise MultiTrainNaNError(f'Missing values found in the dataset. Please handle missing values before proceeding.'
                                    'Pass a value to the fix_nan_custom parameter to handle missing values. i.e. '
                                    'fix_nan_custom={"column1": "ffill", "column2": "bfill"}')
            
    if fix_nan_custom:
        if type(fix_nan_custom) != dict:
            raise MultiTrainTypeError(
                f'fix_nan_custom should be a dictionary of type {dict}. Got {type(fix_nan_custom)}. '
                '\nExample: {"column1": "ffill", "column2": "bfill", "column3": ["value"]}'
                )

        fill_list = ['ffill', 'bfill', 'interpolate']

        for column, strategy in fix_nan_custom.items():
            if column not in dataset.columns:
                raise MultiTrainColumnMissingError(f'Column {column} not found in list of columns. Please pass in a valid column.')
            
            if strategy in fill_list[:2]:
                dataset[column] = getattr(dataset[column], strategy)()
                if dataset[column].isnull().any():
                    print(len(dataset[column]))
                    dataset[column] = _fill_missing_values(dataset, column)
            
            elif strategy == 'interpolate':
                dataset[column] = getattr(dataset[column], strategy)(method='linear')
                if dataset[column].isnull().any():
                    dataset[column] = _fill_missing_values(dataset, column)

        return dataset
    
def _display_table(results, sort, custom_metric=None):
    """
    Displays a sorted table of results.

    Args:
        results (dict): The results to display.
        sort (str): The metric to sort by.
        custom_metric (str, optional): A custom metric to include in sorting.

    Returns:
        pd.DataFrame: A DataFrame of sorted results.
    """
    results_df = pd.DataFrame(results).T
    sorted_ = {'accuracy': 'accuracy', 
               'precision': 'precision', 
               'recall': 'recall', 
               'f1': 'f1', 
               'roc_auc': 'roc_auc', 
               'balanced_accuracy': 'balanced_accuracy'}
    
    if custom_metric:
        sorted_[custom_metric] = custom_metric
         
    if sort:
        results_df = results_df.sort_values(by=sorted_[sort], ascending=False)
        # Move the sorted column to the front
        column_to_move = sorted_[sort]
        first_column = results_df.pop(column_to_move)
        results_df.insert(0, column_to_move, first_column)
        return results_df
    else:
        return results_df
    
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
            raise MultiTrainTypeError(f"You must pass a list of models to the custom models parameter. Got type {type(custom_models)}")
        
        model_names = custom_models
        model_list = [models[values] for values in models if values in custom_models]
    
    return model_names, model_list

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
    except ValueError:
        sklearn.logger.error(f"{model_names[idx]} unable to fit properly")
        pass
    current_prediction = current_model.predict(X_test)
    end = time.time() - start
    
    return current_model, current_prediction, end

# Function to calculate metric
def _calculate_metric(metric_func, y_true, y_pred, average=None):
    if average:
        return metric_func(y_true, y_pred, average=average)
    return metric_func(y_true, y_pred)