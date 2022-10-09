""" This module contains the Base Model"""
import logging
import time
import warnings
from collections import Counter
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm.notebook import trange
from IPython.display import display
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from lightgbm import LGBMClassifier
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

from MultiTrain.errors.fit_exceptions import (
    raise_text_error,
    raise_imbalanced_error,
    raise_kfold1_error,
    raise_split_data_error,
    raise_fold_type_error,
    raise_kfold2_error,
    raise_splitting_error,
)
from MultiTrain.errors.split_exceptions import (
    feature_label_type_error,
    strat_error,
    dimensionality_reduction_type_error,
    test_size_error,
    missing_values_error,
)

from MultiTrain.methods.multitrain_methods import (
    directory,
    show_best,
    img,
    img_plotly,
    write_to_excel,
    _check_target,
    _get_cat_num,
    _fill,
    _fill_columns,
    _dummy,
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    Perceptron,
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

from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    MultinomialNB,
    ComplementNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import (
    FunctionTransformer,
    Normalizer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    make_scorer,
)

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier

# os.environ['OMP_NUM_THREADS'] = "1"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class BaseModel:

    """Base Model"""

    __clasn_metrics = {
        "roc_auc": roc_auc_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "balanced_accuracy_score": balanced_accuracy_score,
    }

    __clasn_keys = [
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy_score",
    ]

    __reg_metrics = {
        "median_absolute_error": median_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_error": mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error,
        "r2": r2_score,
    }

    __reg_keys = [
        "median_absolute_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "r2",
    ]

    def __init__(self) -> None:
        pass

    def split(
        self,
        X: any,
        y: any,
        strat: bool = False,
        sizeOfTest: float = 0.2,
        randomState: int = None,
        shuffle_data: bool = True,
        dimensionality_reduction: bool = False,
        normalize: any = None,
        columns_to_scale: list = None,
        n_components: int = None,
        missing_values: dict = None,
        encode: Union[str, dict] = None,
    ):

        global the_y
        the_y = y
        """
        :param X: features
        :param y: labels
        :param n_components: This sets the number of components to keep
        :param columns_to_scale:
        :param normalize: Transforms input into the range [0,1] or any other range with one of MinMaxScaler, StandardScaler or RobustScaler.
        :param dimensionality_reduction: Utilizes PCA to reduce the dimension of the training and test features
        :param strat: used to initialize stratify = y in train_test_split if True
        :param sizeOfTest: define size of test data
        :param randomState: define random state
        :param shuffle_data: If set to True, it sets shuffle to True in train_test_split
        :param missing_values: Dictionary to fill missing values for categorical and numerical columns, e.g {'cat': 'most_frequent', 'num': 'mean'} where the key 'cat' represents categorical column and the corresponding value represents the strategy used to fill the missing value.

        Example
        df = pd.read_csv("nameOfFile.csv")
        X = df.drop("nameOfLabelColumn", axis=1)
        y = df["nameOfLabelColumn")
        split(X = features, y = labels, sizeOfTest=0.3, randomState=42, strat=True, shuffle_data=True)
        """

        try:
            # values for normalize
            norm = [
                "StandardScaler",
                "MinMaxScaler",
                "RobustScaler",
                "Normalizer",
            ]

            if missing_values:
                categorical_values, numerical_values = _get_cat_num(missing_values)
                cat, num = _fill(categorical_values, numerical_values)
                X = _fill_columns(cat, num, X)

            if encode is not None:
                X = _dummy(X, encode)

            if strat is True:

                if shuffle_data is False:
                    raise TypeError("shuffle_data can only be False if strat is False")

                elif shuffle_data is True:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=sizeOfTest,
                        train_size=1 - sizeOfTest,
                        stratify=y,
                        random_state=randomState,
                        shuffle=shuffle_data,
                    )
                    if dimensionality_reduction is False:
                        return X_train, X_test, y_train, y_test

                    if dimensionality_reduction is True:
                        if normalize is None:
                            raise ValueError(
                                'Pass one of ["StandardScaler", "MinMaxScaler", "RobustScaler" to '
                                "normalize if dimensionality_reduction is True"
                            )

                        if normalize is not None:
                            if columns_to_scale is None:
                                if isinstance(columns_to_scale, list) is False:
                                    raise ValueError(
                                        "Pass a list containing the columns to be scaled to the "
                                        "column_to_scale parameter when using normalize"
                                    )

                            if columns_to_scale is not None:
                                if isinstance(columns_to_scale, tuple):
                                    raise ValueError(
                                        "You can only pass a list to columns_to_scale"
                                    )
                                elif isinstance(columns_to_scale, list):
                                    if normalize in norm:
                                        if normalize == "StandardScaler":
                                            scale = StandardScaler()
                                        elif normalize == "MinMaxScaler":
                                            scale = MinMaxScaler()
                                        elif normalize == "RobustScaler":
                                            scale = RobustScaler()
                                        elif normalize == "Normalizer":
                                            scale = Normalizer()

                                        X_train[columns_to_scale] = scale.fit_transform(
                                            X_train[columns_to_scale]
                                        )
                                        X_test[columns_to_scale] = scale.transform(
                                            X_test[columns_to_scale]
                                        )

                                        pca = PCA(
                                            n_components=n_components,
                                            random_state=self.random_state,
                                        )
                                        X_train = pca.fit_transform(X_train)
                                        X_test = pca.transform(X_test)
                                        return X_train, X_test, y_train, y_test
                                    else:
                                        raise ValueError(f"{normalize} not in {norm}")

            else:
                norm = [
                    "StandardScaler",
                    "MinMaxScaler",
                    "RobustScaler",
                    "Normalizer",
                ]
                if normalize:
                    if columns_to_scale is None:
                        raise ValueError(
                            "Pass a list containing the columns to be scaled to the "
                            "column_to_scale parameter when using normalize"
                        )
                    if columns_to_scale:
                        if isinstance(columns_to_scale, tuple):
                            raise ValueError(
                                "You can only pass a list to columns_to_scale"
                            )

                        if isinstance(columns_to_scale, list):
                            if normalize in norm:
                                if normalize == "StandardScaler":
                                    scale = StandardScaler()
                                elif normalize == "MinMaxScaler":
                                    scale = MinMaxScaler()
                                elif normalize == "RobustScaler":
                                    scale = RobustScaler()
                                elif normalize == "Normalizer":
                                    scale = Normalizer()

                                (X_train, X_test, y_train, y_test,) = train_test_split(
                                    X,
                                    y,
                                    test_size=sizeOfTest,
                                    train_size=1 - sizeOfTest,
                                    random_state=randomState,
                                    shuffle=shuffle_data,
                                )

                                X_train[columns_to_scale] = scale.fit_transform(
                                    X_train[columns_to_scale]
                                )
                                X_test[columns_to_scale] = scale.transform(
                                    X_test[columns_to_scale]
                                )
                                return X_train, X_test, y_train, y_test

                            else:
                                raise ValueError(f"{normalize} not in {norm}")

                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=sizeOfTest,
                        train_size=1 - sizeOfTest,
                        random_state=randomState,
                        shuffle=shuffle_data,
                    )

                    return X_train, X_test, y_train, y_test

        except Exception:
            feature_label_type_error(X, y)
            strat_error(strat)
            dimensionality_reduction_type_error(dimensionality_reduction)
            test_size_error(sizeOfTest)
            missing_values_error(missing_values)

    def use_model(cls, df, model: str = None, best: str = None):
        """


        :param df: the dataframe object
        :param model: name of the classifier algorithm
        :param best: the evaluation metric used to find the best model

        :return:
        """

        if cls.select_models is None:
            name = cls._model_names()
            MODEL = cls._initialize()
        else:
            MODEL, name = cls._custom()

        if model is not None and best is not None:
            raise Exception("You can only use one of the two arguments.")

        if model:
            assert (
                model in name
            ), f"name {model} is not found, here is a list of the available models to work with: {name}"
            index_ = name.index(model)
            return MODEL[index_]

        elif best:
            instance = cls._get_index(df, best)
            print(df)
            return instance

    def tune_parameters(
        self,
        model: str = None,
        parameters: dict = None,
        tune: str = None,
        use_cpu: int = None,
        cv: int = 5,
        n_iter: any = 50,
        return_train_score: bool = False,
        refit: bool = True,
        random_state: int = None,
        factor: int = 3,
        verbose: int = 5,
        resource: any = "n_samples",
        max_resources: any = "auto",
        min_resources_grid: any = "exhaust",
        min_resources_rand: any = "smallest",
        aggressive_elimination: any = False,
        error_score: any = np.nan,
        pre_dispatch: any = "2*n_jobs",
        optimizer_kwargs: any = None,
        fit_params: any = None,
        n_points: any = 1,
        score=None,
    ):
        """
        :param score: if None Accuraccy for classification and r2 for regression
        :param n_points:
        :param fit_params:
        :param optimizer_kwargs:
        :param pre_dispatch:
        :param error_score:
        :param min_resources_grid:
        :param min_resources_rand:
        :param aggressive_elimination:
        :param max_resources:
        :param resource: Defines the resource that increases with each iteration.
        :param verbose:
        :param return_train_score:
        :param refit:
        :param random_state:
        :param n_iter:
        :param model: This is the instance of the model to be used
        :param factor: To be used with HalvingGridSearchCV, It is the ‘halving’ parameter, which determines the proportion of
        candidates that are selected for each subsequent iteration. For example, factor=3 means that only one third of the
        candidates are selected.
        :param parameters: the dictionary of the model parameters
        :param tune: the type of searching method to use, either grid for GridSearchCV
        or random for RandomSearchCV
        :param use_cpu : the value set determines the number of cores used for training,
        if set to -1 it uses all the available cores
        :param cv:This determines the cross validation splitting strategy, defaults to 5
        :return:
        """

        if isinstance(parameters, dict) is False:
            raise TypeError(
                "The 'parameters' argument only accepts a dictionary of the parameters for the "
                "model you want to train with."
            )

        if self.__class__.__name__ == "MultiClassifier":
            score = "accuracy"
        else:
            score = "r2 "
        if tune:
            # K and M

            if self.__class__.__name__ == "MultiClassifier":
                if score not in BaseModel.__clasn_keys:
                    raise ValueError(
                        f"expected one of {BaseModel.__clasn_keys}, received {score}"
                    )
                scorers = make_scorer(BaseModel.__clasn_metrics[score])
            else:
                if score not in BaseModel.__reg_keys:
                    raise ValueError(
                        f"expected one of {BaseModel.__reg_keys}, received {score}"
                    )
                scorers = make_scorer(BaseModel.__reg_metrics[score])

            if tune == "grid":
                tuned_model = GridSearchCV(
                    estimator=model,
                    param_grid=parameters,
                    n_jobs=use_cpu,
                    cv=cv,
                    verbose=verbose,
                    error_score=error_score,
                    pre_dispatch=pre_dispatch,
                    return_train_score=return_train_score,
                    scoring=scorers,
                    refit=refit,
                )
                return tuned_model

            elif tune == "random":
                tuned_model = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=parameters,
                    n_jobs=use_cpu,
                    cv=cv,
                    verbose=verbose,
                    random_state=random_state,
                    n_iter=n_iter,
                    return_train_score=return_train_score,
                    error_score=error_score,
                    scoring=scorers,
                    refit=refit,
                    pre_dispatch=pre_dispatch,
                )

                return tuned_model

            elif tune == "bayes":
                tuned_model = BayesSearchCV(
                    estimator=model,
                    search_spaces=parameters,
                    n_jobs=use_cpu,
                    return_train_score=return_train_score,
                    cv=cv,
                    verbose=verbose,
                    refit=refit,
                    random_state=random_state,
                    scoring=scorers,
                    error_score=error_score,
                    optimizer_kwargs=optimizer_kwargs,
                    n_points=n_points,
                    n_iter=n_iter,
                    fit_params=fit_params,
                    pre_dispatch=pre_dispatch,
                )

                return tuned_model

            elif tune == "half-grid":
                tuned_model = HalvingGridSearchCV(
                    estimator=model,
                    param_grid=parameters,
                    n_jobs=use_cpu,
                    cv=cv,
                    verbose=verbose,
                    random_state=42,
                    factor=factor,
                    refit=refit,
                    scoring=scorers,
                    resource=resource,
                    min_resources=min_resources_grid,
                    max_resources=max_resources,
                    error_score=error_score,
                    aggressive_elimination=aggressive_elimination,
                )

                return tuned_model

            elif tune == "half-random":
                tuned_model = HalvingRandomSearchCV(
                    estimator=model,
                    param_distributions=parameters,
                    n_jobs=use_cpu,
                    cv=cv,
                    verbose=verbose,
                    random_state=42,
                    factor=factor,
                    refit=refit,
                    scoring=scorers,
                    resource=resource,
                    error_score=error_score,
                    min_resources=min_resources_rand,
                    max_resources=max_resources,
                    aggressive_elimination=aggressive_elimination,
                )

                return tuned_model
