from collections import Counter
from operator import __setitem__
from typing import Union

import seaborn as sns
import plotly.express as px

# from hyperopt import tpe
# from hpsklearn import HyperoptEstimator, sklearn_ExtraTreesClassifier, random_forest
from IPython.display import display
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
    SMOTENC,
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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from catboost import CatBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from pandas import DataFrame
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import (
    LogisticRegressionCV,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    Perceptron,
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import NuSVC
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from MultiTrain.methods.multitrain_methods import (
    directory,
    img,
    img_plotly,
    kf_best_model,
    write_to_excel,
    _check_target, _get_cat_num, _fill, _fill_columns, _dummy,
)

from skopt import BayesSearchCV
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
)
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
from matplotlib import pyplot as plt
from numpy.random import randint
from imblearn.pipeline import Pipeline as imbpipe
import pandas as pd
import numpy as np
import warnings
import time

import logging
import os

# os.environ['OMP_NUM_THREADS'] = "1"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class MultiClassifier:
    def __init__(
        self,
        cores: int = -1,
        random_state: int = randint(1000),
        verbose: bool = False,
        imbalanced: bool = False,
        sampling: str = None,
        strategy: str or float = "auto"
    ) -> None:

        self.cores = cores
        self.random_state = random_state
        self.verbose = verbose
        self.imbalanced = imbalanced
        self.sampling = sampling
        self.strategy = strategy

        self.oversampling_list = [
            "SMOTE",
            "RandomOverSampler",
            "SMOTEN",
            "ADASYN",
            "BorderlineSMOTE",
            "KMeansSMOTE",
            "SVMSMOTE",
        ]
        self.oversampling_methods = [
            SMOTE(sampling_strategy=self.strategy, random_state=self.random_state),
            RandomOverSampler(
                sampling_strategy=self.strategy, random_state=self.random_state
            ),
            SMOTEN(sampling_strategy=self.strategy, random_state=self.random_state),
            ADASYN(sampling_strategy=self.strategy, random_state=self.random_state),
            BorderlineSMOTE(
                sampling_strategy=self.strategy, random_state=self.random_state
            ),
            KMeansSMOTE(
                sampling_strategy=self.strategy, random_state=self.random_state
            ),
            SVMSMOTE(sampling_strategy=self.strategy, random_state=self.random_state),
        ]

        self.undersampling_list = [
            "CondensedNearestNeighbour",
            "EditedNearestNeighbours",
            "RepeatedEditedNearestNeighbours",
            "AllKNN",
            "InstanceHardnessThreshold",
            "NearMiss",
            "NeighbourhoodCleaningRule",
            "OneSidedSelection",
            "RandomUnderSampler",
            "TomekLinks",
        ]
        self.undersampling_methods = [
            CondensedNearestNeighbour(
                sampling_strategy=self.strategy,
                random_state=self.random_state,
                n_jobs=self.cores,
            ),
            EditedNearestNeighbours(sampling_strategy=self.strategy, n_jobs=self.cores),
            RepeatedEditedNearestNeighbours(
                sampling_strategy=self.strategy, n_jobs=self.cores
            ),
            AllKNN(sampling_strategy=self.strategy, n_jobs=self.cores),
            InstanceHardnessThreshold(
                sampling_strategy=self.strategy,
                random_state=self.random_state,
                n_jobs=self.cores,
            ),
            NearMiss(sampling_strategy=self.strategy, n_jobs=self.cores),
            NeighbourhoodCleaningRule(
                sampling_strategy=self.strategy, n_jobs=self.cores
            ),
            OneSidedSelection(sampling_strategy=self.strategy, n_jobs=self.cores),
            RandomUnderSampler(
                sampling_strategy=self.strategy, random_state=self.random_state
            ),
            TomekLinks(sampling_strategy=self.strategy, n_jobs=self.cores),
        ]

        self.over_under_list = ["SMOTEENN", "SMOTETomek"]
        self.over_under_methods = [
            SMOTEENN(
                sampling_strategy=self.strategy,
                random_state=self.random_state,
                n_jobs=self.cores,
            ),
            SMOTETomek(
                sampling_strategy=self.strategy,
                random_state=self.random_state,
                n_jobs=self.cores,
            ),
        ]

        self.kf_binary_columns_train = [
            "Overfitting",
            "Accuracy(Train)",
            "Accuracy",
            "Balanced Accuracy(train)",
            "Balanced Accuracy",
            "Precision(Train)",
            "Precision",
            "Recall(Train)",
            "Recall",
            "f1(Train)",
            "f1",
            "r2(Train)",
            "r2",
            "Standard Deviation of Accuracy(Train)",
            "Standard Deviation of Accuracy",
            "Time Taken(s)",
        ]

        self.kf_binary_columns_test = [
            "Overfitting",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "f1",
            "r2",
            "Standard Deviation of Accuracy",
            "Time Taken(s)",
        ]

        self.kf_multiclass_columns_train = [
            "Precision Macro(Train)",
            "Precision Macro",
            "Recall Macro(Train)",
            "Recall Macro",
            "f1 Macro(Train)",
            "f1 Macro",
            "Time Taken(s)",
        ]

        self.kf_multiclass_columns_test = [
            "Precision Macro",
            "Recall Macro",
            "f1 Macro",
        ]

        self.t_split_binary_columns_train = [
            "Overfitting",
            "Accuracy(Train)",
            "Accuracy",
            "Balanced Accuracy(Train)",
            "Balanced Accuracy",
            "r2 score(Train)",
            "r2 score",
            "ROC AUC(Train)",
            "ROC AUC",
            "f1 score(Train)",
            "f1 score",
            "Precision(Train)",
            "Precision",
            "Recall(Train)",
            "Recall",
            "execution time(seconds)",
        ]

        self.t_split_binary_columns_test = [
            "Overfitting",
            "Accuracy",
            "Balanced Accuracy",
            "r2 score",
            "ROC AUC",
            "f1 score",
            "Precision",
            "Recall",
            "execution time(seconds)",
        ]

        self.t_split_multiclass_columns_train = [
            "Overfitting",
            "Accuracy(Train)",
            "Accuracy",
            "Balanced Accuracy(Train)",
            "Balanced Accuracy",
            "r2 score(Train)",
            "r2 score",
            "f1 score(Train)",
            "f1 score",
            "Precision(Train)",
            "Precision",
            "Recall(Train)",
            "Recall",
            "execution time(seconds)",
        ]

        self.t_split_multiclass_columns_test = [
            "Overfitting",
            "Accuracy",
            "Balanced Accuracy",
            "r2 score",
            "f1 score",
            "Precision",
            "Recall",
            "execution time(seconds)",
        ]

    def strategies(self) -> None:
        print(f"Over-Sampling Methods = {self.oversampling_list}")
        print("\n")
        print(f"Under-Sampling Methods = {self.undersampling_list}")
        print("\n")
        print(
            f"Combination of over and under-sampling methods = {self.over_under_list}"
        )

    def _get_sample_index_method(self):

        if self.sampling in self.oversampling_list:
            index_ = self.oversampling_list.index(self.sampling)
            method = self.oversampling_methods[index_]
            return method

        elif self.sampling in self.undersampling_list:
            index_ = self.undersampling_list.index(self.sampling)
            method = self.undersampling_methods[index_]
            return method

        elif self.sampling in self.over_under_list:
            index_ = self.over_under_list.index(self.sampling)
            method = self.over_under_methods[index_]
            return method


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
        encode: Union[str, dict] = None
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
        if isinstance(X, int or bool) or isinstance(y, int or bool):
            raise ValueError(
                f"{X} and {y} are not valid arguments for 'split'."
                f"Try using the standard variable names e.g split(X, y) instead of split({X}, {y})"
            )
        elif isinstance(strat, bool) is False:
            raise TypeError(
                "argument of type int or str is not valid. Parameters for strat is either False or True"
            )

        elif sizeOfTest < 0 or sizeOfTest > 1:
            raise ValueError("value of sizeOfTest should be between 0 and 1")

        else:
            # values for normalize
            norm = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']

            if missing_values:
                if isinstance(missing_values, dict):
                    if missing_values['cat'] != 'most_frequent':
                        raise ValueError(
                            f"Received value '{missing_values['cat']}', you can only use 'most_frequent' for "
                            f"categorical columns")
                    elif missing_values['num'] not in ['mean', 'median', 'most_frequent']:
                        raise ValueError(
                            f"Received value '{missing_values['num']}', you can only use one of ['mean', 'median', "
                            f"'most_frequent'] for numerical columns")
                    categorical_values, numerical_values = _get_cat_num(missing_values)
                    cat, num = _fill(categorical_values, numerical_values)
                    X = _fill_columns(cat, num, X)

                else:
                    raise TypeError(
                        f'missing_values parameter can only be of type dict, type {type(missing_values)} received')

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
                norm = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
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

                                X_train, X_test, y_train, y_test = train_test_split(
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

                                X_train, X_test = (
                                    X_train.reset_index(),
                                    X_test.reset_index(),
                                )
                                X_train, X_test = X_train.drop(
                                    "index", axis=1
                                ), X_test.drop("index", axis=1)

                                return X_train, X_test, y_train, y_test

                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=sizeOfTest,
                        train_size=1 - sizeOfTest,
                        random_state=randomState,
                        shuffle=shuffle_data,
                    )
                    X_train, X_test = X_train.reset_index(), X_test.reset_index()
                    X_train, X_test = X_train.drop("index", axis=1), X_test.drop(
                        "index", axis=1
                    )

                    return X_train, X_test, y_train, y_test

    def classifier_model_names(self) -> list:
        model_names = [
            "Logistic Regression",
            "LogisticRegressionCV",
            "SGDClassifier",
            "PassiveAggressiveClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "AdaBoostClassifier",
            "CatBoostClassifier",
            "XGBClassifier",
            "GaussianNB",
            "LinearDiscriminantAnalysis",
            "KNeighborsClassifier",
            "MLPClassifier",
            "SVC",
            "DecisionTreeClassifier",
            "BernoulliNB",
            "MultinomialNB",
            "ComplementNB",
            "ExtraTreesClassifier",
            "RidgeClassifier",
            "RidgeClassifierCV",
            "ExtraTreeClassifier",
            "QuadraticDiscriminantAnalysis",
            "LinearSVC",
            "BaggingClassifier",
            "BalancedBaggingClassifier",
            "Perceptron",
            "NuSVC",
            "LGBMClassifier",
        ]
        return model_names

    def _initialize_(self):
        """
        It initializes all the models that we will be using in our ensemble
        """
        # if self.param_tuning is True:
        #    if self.timeout is False:
        #        logger.info('It is recommended to set a timeout to avoid longer training times')

        lr = LogisticRegression(n_jobs=self.cores, random_state=self.random_state)
        lrcv = LogisticRegressionCV(n_jobs=self.cores, refit=True)
        sgdc = SGDClassifier(n_jobs=self.cores, random_state=self.random_state)
        pagg = PassiveAggressiveClassifier(
            n_jobs=self.cores, random_state=self.random_state
        )
        rfc = RandomForestClassifier(n_jobs=self.cores, random_state=self.random_state)
        gbc = GradientBoostingClassifier(random_state=self.random_state)
        hgbc = HistGradientBoostingClassifier(random_state=self.random_state)
        abc = AdaBoostClassifier(random_state=self.random_state)
        cat = CatBoostClassifier(
            thread_count=self.cores, verbose=False, random_state=self.random_state
        )
        xgb = XGBClassifier(
            eval_metric="mlogloss",
            n_jobs=self.cores,
            refit=True,
            random_state=self.random_state,
        )
        gnb = GaussianNB()
        lda = LinearDiscriminantAnalysis()
        knc = KNeighborsClassifier(n_jobs=self.cores)
        mlp = MLPClassifier(random_state=self.random_state)
        svc = SVC(random_state=self.random_state)
        dtc = DecisionTreeClassifier(random_state=self.random_state)
        bnb = BernoulliNB()
        mnb = MultinomialNB()
        conb = ComplementNB()
        etcs = ExtraTreesClassifier(n_jobs=self.cores, random_state=self.random_state)
        rcl = RidgeClassifier(random_state=self.random_state)
        rclv = RidgeClassifierCV()
        etc = ExtraTreeClassifier(random_state=self.random_state)
        # self.gpc = GaussianProcessClassifier(warm_start=True, random_state=42, n_jobs=-1)
        qda = QuadraticDiscriminantAnalysis()
        lsvc = LinearSVC(random_state=self.random_state)
        bc = BaggingClassifier(n_jobs=self.cores, random_state=self.random_state)
        bbc = BalancedBaggingClassifier(
            n_jobs=self.cores, random_state=self.random_state
        )
        per = Perceptron(n_jobs=self.cores, random_state=self.random_state)
        nu = NuSVC(random_state=self.random_state)
        lgbm = LGBMClassifier(random_state=self.random_state)

        return (
            lr,
            lrcv,
            sgdc,
            pagg,
            rfc,
            gbc,
            hgbc,
            abc,
            cat,
            xgb,
            gnb,
            lda,
            knc,
            mlp,
            svc,
            dtc,
            bnb,
            mnb,
            conb,
            etcs,
            rcl,
            rclv,
            etc,
            qda,
            lsvc,
            bc,
            bbc,
            per,
            nu,
            lgbm,
        )

    def _get_index(self, df, the_best):
        name = list(self.classifier_model_names())
        MODEL = self._initialize_()
        df["model_names"] = name
        high = [
            "accuracy",
            "balanced accuracy",
            "f1 score",
            "r2 score",
            "ROC AUC",
            "Test Acc",
            "Test Precision",
            "Test Recall",
            "Test f1",
            "Test r2",
            "Test Precision Macro",
            "Test Recall Macro",
            "Test f1 Macro",
        ]
        low = ["mean absolute error", "mean squared error", "Test std"]

        if the_best in high:
            best_model_details = df[df[the_best] == df[the_best].max()]

        elif the_best in low:
            best_model_details = df[df[the_best] == df[the_best].min()]

        else:
            raise Exception(f"metric {the_best} not found")

        best_model_details = best_model_details.reset_index()
        best_model_name = best_model_details.iloc[0]["model_names"]
        index_ = name.index(best_model_name)
        return MODEL[index_]

    def _startKFold_(self, param, param_X, param_y, param_cv, train_score):
        names = self.classifier_model_names()
        target_class = _check_target(param_y)
        if self.imbalanced is True:
            logger.info("You are receiving this message because you set imbalanced to True. All resampling techniques "
                        "e.g SMOTE has been disabled in this new version till a permanent fix is implemented, "
                        "use the split method instead if you're dealing with imbalanced data")
        if target_class == "binary":
            dataframe = {}
            for i in range(len(param)):

                if self.verbose is True:
                    print(param[i])

                score = (
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "r2",
                )

                start = time.time()
                try:
                    scores = cross_validate(
                        estimator=param[i],
                        X=param_X,
                        y=param_y,
                        scoring=score,
                        cv=param_cv,
                        n_jobs=self.cores,
                        return_train_score=True,
                    )
                    end = time.time()
                    seconds = end - start

                    mean_train_acc = scores["train_accuracy"].mean()
                    mean_test_acc = scores["test_accuracy"].mean()
                    mean_train_bacc = scores["train_balanced_accuracy"].mean()
                    mean_test_bacc = scores["test_balanced_accuracy"].mean()
                    mean_train_precision = scores["train_precision"].mean()
                    mean_test_precision = scores["test_precision"].mean()
                    mean_train_f1 = scores["train_f1"].mean()
                    mean_test_f1 = scores["test_f1"].mean()
                    mean_train_r2 = scores["train_r2"].mean()
                    mean_test_r2 = scores["test_r2"].mean()
                    mean_train_recall = scores["train_recall"].mean()
                    mean_test_recall = scores["test_recall"].mean()
                    train_stdev = scores["train_accuracy"].std()
                    test_stdev = scores["test_accuracy"].std()
                    overfitting = True if (mean_train_acc - mean_test_acc) > 0.1 else False
                except Exception:
                    logger.error(f"{param[i]} unable to fit properly")
                    seconds, mean_train_acc, mean_test_acc = np.nan, np.nan, np.nan
                    mean_train_bacc, mean_test_bacc = np.nan, np.nan
                    mean_train_precision, mean_test_precision = np.nan, np.nan
                    mean_train_f1, mean_test_f1, mean_train_r2, mean_test_r2 = np.nan, np.nan, np.nan, np.nan
                    mean_train_recall, mean_test_recall, train_stdev, test_stdev, overfitting = np.nan, np.nan, np.nan, np.nan, False

                # USING RESAMPLING TECHNIQUES HAS BEEN TEMPORARILY DISABLED ON cross_validate till a more permanent
                # fix is implemented

                # elif self.imbalanced is True:
                #    start = time.time()
                #    method = self._get_sample_index_method()
                #    pipeline = imbpipe(
                #        steps=[
                #            ("over", method), ("model", param[i])
                #        ]
                #    )
                #    try:
                #        scores = cross_validate(
                #            estimator=pipeline,
                #            X=param_X,
                #            y=param_y,
                #            scoring=score,
                #            cv=param_cv,
                #            n_jobs=self.cores,
                #            return_train_score=True
                #        )
                #        end = time.time()
                #        seconds = end - start

                #        mean_train_acc = scores["train_accuracy"].mean()
                #        mean_test_acc = scores["test_accuracy"].mean()
                #        mean_train_bacc = scores["train_balanced_accuracy"].mean()
                #        mean_test_bacc = scores["test_balanced_accuracy"].mean()
                #        mean_train_precision = scores["train_precision"].mean()
                #        mean_test_precision = scores["test_precision"].mean()
                #        mean_train_f1 = scores["train_f1"].mean()
                #        mean_test_f1 = scores["test_f1"].mean()
                #        mean_train_r2 = scores["train_r2"].mean()
                #        mean_test_r2 = scores["test_r2"].mean()
                #        mean_train_recall = scores["train_recall"].mean()
                #        mean_test_recall = scores["test_recall"].mean()
                #        train_stdev = scores["train_accuracy"].std()
                #        test_stdev = scores["test_accuracy"].std()
                #        overfitting = True if (mean_train_acc - mean_test_acc) > 0.1 else False
                #    except Exception:
                #        logger.error(f'{param[i]} unable to fit properly')
                #        seconds, mean_train_acc, mean_test_acc = np.nan, np.nan, np.nan
                #        mean_train_bacc, mean_test_bacc = np.nan, np.nan
                #        mean_train_precision, mean_test_precision = np.nan, np.nan
                #        mean_train_f1, mean_test_f1, mean_train_r2, mean_test_r2 = np.nan, np.nan, np.nan, np.nan
                #        mean_train_recall, mean_test_recall, train_stdev, test_stdev, overfitting = np.nan, np.nan, np.nan, np.nan, False

                # scores = scores.tolist()
                if train_score is True:
                    scores_df = [
                        overfitting,
                        mean_train_acc,
                        mean_test_acc,
                        mean_train_bacc,
                        mean_test_bacc,
                        mean_train_precision,
                        mean_test_precision,
                        mean_train_f1,
                        mean_test_f1,
                        mean_train_r2,
                        mean_test_r2,
                        mean_train_recall,
                        mean_test_recall,
                        train_stdev,
                        test_stdev,
                        seconds,
                    ]
                    dataframe.update({names[i]: scores_df})

                elif train_score is False:
                    scores_df = [
                        overfitting,
                        mean_test_acc,
                        mean_test_bacc,
                        mean_test_precision,
                        mean_test_f1,
                        mean_test_r2,
                        mean_test_recall,
                        test_stdev,
                        seconds,
                    ]
                    dataframe.update({names[i]: scores_df})
            return dataframe

        elif target_class == "multiclass":
            dataframe = {}
            for j in range(len(param)):
                start = time.time()
                score = ("precision_macro", "recall_macro", "f1_macro")
                scores = cross_validate(
                    estimator=param[j],
                    X=param_X,
                    y=param_y,
                    scoring=score,
                    cv=param_cv,
                    n_jobs=-1,
                    return_train_score=True,
                )
                end = time.time()
                seconds = end - start

                if train_score is True:

                    mean_train_precision = scores["train_precision_macro"].mean()
                    mean_test_precision = scores["test_precision_macro"].mean()
                    mean_train_f1 = scores["train_f1_macro"].mean()
                    mean_test_f1 = scores["test_f1_macro"].mean()
                    mean_train_recall = scores["train_recall_macro"].mean()
                    mean_test_recall = scores["test_recall_macro"].mean()

                    scores_df = [
                        mean_train_precision,
                        mean_test_precision,
                        mean_train_f1,
                        mean_test_f1,
                        mean_train_recall,
                        mean_test_recall,
                        seconds,
                    ]

                    dataframe.update({names[j]: scores_df})

                elif train_score is False:

                    mean_test_precision = scores["test_precision_macro"].mean()
                    mean_test_f1 = scores["test_f1_macro"].mean()
                    mean_test_recall = scores["test_recall_macro"].mean()

                    scores_df = [
                        mean_test_precision,
                        mean_test_f1,
                        mean_test_recall,
                        seconds,
                    ]

                    dataframe.update({names[j]: scores_df})

            return dataframe

    def fit(
        self,
        X: str = None,
        y: str = None,
        split_self: bool = False,
        X_train: str = None,
        X_test: str = None,
        y_train: str = None,
        y_test: str = None,
        split_data=None,
        splitting: bool = False,
        kf: bool = False,
        fold: int = 5,
        excel: bool = False,
        return_best_model: bool = None,
        show_train_score: bool = False,
        text: bool = False,
        vectorizer: str = None,
        ngrams: tuple = None,
    ) -> DataFrame:
        # If splitting is False, then do nothing. If splitting is True, then assign the values of split_data to the
        # variables X_train, X_test, y_train, and y_test

        # :param ngrams: It can be used when text is set to True.
        # :param vectorizer: It can only be used when text is set to True.
        # :param text: Set this parameter to True only if you’re working on an NLP problem.
        # :param show_train_score: Set this parameter to True to show the train scores together with the test scores
        # :param return_fastest_model: defaults to False, set to True when you want the method to only return a dataframe
        # of the fastest model

        # :param return_best_model: defaults to False, set to True when you want the method to only return a dataframe of
        # the best model

        # :param split_self: defaults to False, set to True when you split the data yourself

        # :param excel: defaults to False, set to True when you want the dataframe to save to an excel file in your
        # current working directory

        # :param y: labels

        # :param X: features

        # :type fold: object
        # :param fold: arguments for KFold where 10 is the n_splits, 1 is the random_state and True is to allow shuffling
        # :param kf: defaults to False, set to True when you want to use KFold cross validation as your splitting method
        # :param X_train: The training data
        # :param X_test: The test data
        # :param y_train: The training set labels
        # :param y_test: The test set labels
        # :param split_data: str = None, splitting: bool = False
        # :type split_data: str
        # :param splitting: bool = False, defaults to False
        # :type splitting: bool (optional)
        # :param target: defaults to binary, this is used to specify if the target is binary or multiclass
        # If using splitting = True
        # df = pd.read_csv("nameOfFile.csv")
        # X = df.drop("nameOfLabelColumn", axis=1)
        # y = df["nameOfLabelColumn")
        # the_split_data = split(X = features, y = labels, sizeOfTest=0.3, randomState=42, strat=True, shuffle_data=True)
        # fit_eval_models(splitting = True, split_data = the_split_data)

        # If using kf = True

        # fit(X = features, y = labels, kf = True, fold = (10, 42, True))

        global y_te

        target_class = _check_target(y) if y is not None else _check_target(split_data[3])
        if text:
            if isinstance(text, bool) is False:
                raise TypeError(
                    "parameter text is of type bool only. set to true or false"
                )

            if text is False:
                if vectorizer is not None:
                    raise Exception(
                        "parameter vectorizer can only be accepted when parameter text is True"
                    )

                if ngrams is not None:
                    raise Exception(
                        "parameter ngrams can only be accepted when parameter text is True"
                    )

        if self.imbalanced is False:
            if self.sampling:
                raise Exception(
                    'this parameter can only be used if "imbalanced" is set to True'
                )

        if isinstance(splitting, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in splitting. Try splitting = False or splitting = True "
                f"instead of splitting = {splitting}"
            )

        if isinstance(kf, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in kf. Try kf = False or kf = True "
                f"instead of kf = {kf}"
            )

        if isinstance(fold, int) is False:
            raise TypeError(
                "param fold is of type int, pass a integer to fold e.g fold = 5, where 5 is number of "
                "splits you want to use for the cross validation procedure"
            )

        if kf:
            if split_self is True:
                raise Exception(
                    "split_self should only be set to True when you split with train_test_split from "
                    "sklearn.model_selection"
                )

            if splitting:
                raise ValueError(
                    "KFold cross validation cannot be true if splitting is true and splitting cannot be "
                    "true if KFold is true"
                )

            if split_data:
                raise ValueError(
                    "split_data cannot be used with kf, set splitting to True to use param "
                    "split_data"
                )

        if kf is True and (X is None or y is None or (X is None and y is None)):
            raise ValueError("Set the values of features X and target y")

        if splitting is True or split_self is True:
            if splitting and split_data:
                X_tr, X_te, y_tr, y_te = (
                    split_data[0],
                    split_data[1],
                    split_data[2],
                    split_data[3],
                )
            elif (
                X_train is not None
                and X_test is not None
                and y_train is not None
                and y_test is not None
            ):
                X_tr, X_te, y_tr, y_te = X_train, X_test, y_train, y_test
            model = self._initialize_()
            names = self.classifier_model_names()
            dataframe = {}
            for i in range(len(model)):
                if self.verbose is True:
                    print(model[i])
                start = time.time()

                if text is False:

                    if self.imbalanced is False:
                        try:
                            model[i].fit(X_tr, y_tr)
                        except ValueError:
                            logger.error(f"{model[i]} unable to fit properly")
                            pass

                    elif self.imbalanced is True:
                        method = self._get_sample_index_method()

                        if self.verbose is True:
                            print(f"Before resampling: {Counter(y_tr)}")
                        X_tr_, y_tr_ = method.fit_resample(X_tr, y_tr)
                        if self.verbose is True:
                            print(f"After resampling: {Counter(y_tr_)}")
                            print("\n")
                        try:
                            model[i].fit(X_tr_, y_tr_)
                        except ValueError:
                            logger.error(f'{model[i]} unable to fit properly')
                            pass

                    end = time.time()

                    try:

                        pred = model[i].predict(X_te)

                        pred_train = model[i].predict(X_tr)
                    except AttributeError:
                        pass

                elif text is True:
                    if vectorizer == "count":
                        try:
                            try:
                                pipeline = make_pipeline(
                                    CountVectorizer(ngram_range=ngrams), model[i]
                                )

                                pipeline.fit(X_tr, y_tr)
                                pred = pipeline.predict(X_te)

                                pred_train = pipeline.predict(X_tr)

                            except TypeError:
                                # This is a fix for the error below when using gradient boosting classifier or
                                # HistGradientBoostingClassifier TypeError: A sparse matrix was passed,
                                # but dense data is required. Use X.toarray() to convert to a dense numpy array.
                                pipeline = make_pipeline(
                                    CountVectorizer(ngram_range=ngrams),
                                    FunctionTransformer(
                                        lambda x: x.todense(), accept_sparse=True
                                    ),
                                    model[i],
                                )
                                pipeline.fit(X_tr, y_tr)
                                pred = pipeline.predict(X_te)

                                pred_train = pipeline.predict(X_tr)

                        except Exception:
                            logger.error(f'{model[i]} unable to fit properly')
                            pass

                    elif vectorizer == "tfidf":
                        try:
                            try:
                                pipeline = make_pipeline(
                                    TfidfVectorizer(ngram_range=ngrams), model[i]
                                )

                                pipeline.fit(X_tr, y_tr)
                                pred = pipeline.predict(X_te)

                                pred_train = pipeline.predict(X_tr)

                            except TypeError:
                                # This is a fix for the error below when using gradient boosting classifier or
                                # HistGradientBoostingClassifier TypeError: A sparse matrix was passed,
                                # but dense data is required. Use X.toarray() to convert to a dense numpy array.
                                pipeline = make_pipeline(
                                    TfidfVectorizer(ngram_range=ngrams),
                                    FunctionTransformer(
                                        lambda x: x.todense(), accept_sparse=True
                                    ),
                                    model[i],
                                )
                                pipeline.fit(X_tr, y_tr)

                        except Exception:
                            logger.error(f'{model[i]} unable to fit properly')
                            pass

                    end = time.time()

                true = y_te
                true_train = y_tr

                acc = accuracy_score(true, pred)
                bacc = balanced_accuracy_score(true, pred)
                r2 = r2_score(true, pred)
                try:
                    roc = roc_auc_score(true, pred)
                except ValueError:
                    roc = None

                if target_class == "binary":
                    f1 = f1_score(true, pred)
                    pre = precision_score(true, pred)
                    rec = recall_score(true, pred)

                elif target_class == "multiclass":
                    if self.imbalanced is True:
                        f1 = f1_score(true, pred, average="micro")
                        pre = precision_score(true, pred, average="micro")
                        rec = recall_score(true, pred, average="micro")
                    elif self.imbalanced is False:
                        f1 = f1_score(true, pred, average="macro")
                        pre = precision_score(true, pred, average="macro")
                        rec = recall_score(true, pred, average="macro")

                tacc = accuracy_score(true_train, pred_train)
                tbacc = balanced_accuracy_score(true_train, pred_train)
                tr2 = r2_score(true_train, pred_train)
                try:
                    troc = roc_auc_score(true_train, pred_train)
                except ValueError:

                    troc = None

                if target_class == "binary":
                    tf1 = f1_score(true_train, pred_train)
                    tpre = precision_score(true_train, pred_train)
                    trec = recall_score(true_train, pred_train)

                elif target_class == "multiclass":
                    if self.imbalanced is True:
                        tf1 = f1_score(true_train, pred_train, average="micro")
                        tpre = precision_score(true_train, pred_train, average="micro")
                        trec = recall_score(true_train, pred_train, average="micro")

                    elif self.imbalanced is False:
                        tf1 = f1_score(true_train, pred_train, average="macro")
                        tpre = precision_score(true_train, pred_train, average="macro")
                        trec = recall_score(true_train, pred_train, average="macro")

                overfit = True if (tacc - acc) > 0.1 else False
                time_taken = round(end - start, 2)

                if show_train_score is False:
                    eval_bin = [overfit, acc, bacc, r2, roc, f1, pre, rec, time_taken]
                    eval_mul = [overfit, acc, bacc, r2, f1, pre, rec, time_taken]

                elif show_train_score is True:
                    eval_bin = [
                        overfit,
                        tacc,
                        acc,
                        tbacc,
                        bacc,
                        tr2,
                        r2,
                        troc,
                        roc,
                        tf1,
                        f1,
                        tpre,
                        pre,
                        trec,
                        rec,
                        time_taken,
                    ]
                    eval_mul = [
                        overfit,
                        tacc,
                        acc,
                        tbacc,
                        bacc,
                        tr2,
                        r2,
                        tf1,
                        f1,
                        tpre,
                        pre,
                        trec,
                        rec,
                        time_taken,
                    ]

                if target_class == "binary":
                    dataframe.update({names[i]: eval_bin})
                elif target_class == "multiclass":
                    dataframe.update({names[i]: eval_mul})

            if show_train_score is False:
                if target_class == "binary":
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.t_split_binary_columns_test,
                    )

                elif target_class == "multiclass":
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.t_split_multiclass_columns_test,
                    )

            elif show_train_score is True:
                if target_class == "binary":
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.t_split_binary_columns_train,
                    )

                elif target_class == "multiclass":
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.t_split_multiclass_columns_train,
                    )

            if return_best_model is not None:
                logger.info(f"BEST MODEL BASED ON {return_best_model}")
                retrieve_df = df.reset_index()
                logger.info(
                    f'The best model based on the {return_best_model} metric is {retrieve_df["index"][0]}'
                )
                display(df.sort_values(by=return_best_model, ascending=False))

            elif return_best_model is None:
                display(df.style.highlight_max(color="yellow"))

            write_to_excel(excel, df)
            return df

        elif kf is True:

            # Fitting the models and predicting the values of the test set.
            KFoldModel = self._initialize_()
            names = self.classifier_model_names()

            if target_class == "binary":
                logger.info("Training started")
                dataframe = self._startKFold_(
                    param=KFoldModel,
                    param_X=X,
                    param_y=y,
                    param_cv=fold,
                    train_score=show_train_score,
                )

                if show_train_score is True:
                    df = pd.DataFrame.from_dict(
                        dataframe, orient="index", columns=self.kf_binary_columns_train
                    )

                elif show_train_score is False:
                    df = pd.DataFrame.from_dict(
                        dataframe, orient="index", columns=self.kf_binary_columns_test
                    )

                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_

            elif target_class == "multiclass":
                logger.info("Training started")
                dataframe = self._startKFold_(
                    param=KFoldModel,
                    param_X=X,
                    param_y=y,
                    param_cv=fold,
                    train_score=show_train_score,
                )

                if show_train_score is True:
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.kf_multiclass_columns_train,
                    )
                elif show_train_score is False:
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=self.kf_multiclass_columns_test,
                    )

                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_

    def use_model(self, df, model: str = None, best: str = None):
        """


        :param df: the dataframe object
        :param model: name of the classifier algorithm
        :param best: the evaluation metric used to find the best model

        :return:
        """

        name = self.classifier_model_names()
        MODEL = self._initialize_()

        if model is not None and best is not None:
            raise Exception("You can only use one of the two arguments.")

        if model:
            if model not in name:
                raise Exception(
                    f"name {model} is not found, "
                    f"here is a list of the available models to work with: {name}"
                )
            elif model in name:
                index_ = name.index(model)
                return MODEL[index_]

        elif best:
            instance = self._get_index(df, best)
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
        score="accuracy",
    ):
        """
        :param score:
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
        if tune:
            scorers = make_scorer(accuracy_score)

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
                    scoring=score,
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
                    scoring=score,
                    resource=resource,
                    error_score=error_score,
                    min_resources=min_resources_rand,
                    max_resources=max_resources,
                    aggressive_elimination=aggressive_elimination,
                )

                return tuned_model

    def visualize(
        self,
        param: {__setitem__},
        y: any = None,
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        size=(15, 8),
        save: str = None,
        save_name="dir1",
    ):

        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

        :param target:
        :param file_path:
        :param param: {__setitem__}
        :type param: {__setitem__}
        :param kf: set to True if you used KFold, defaults to False
        :type kf: bool (optional)
        :param t_split: True if you used the split method, defaults to False
        :type t_split: bool (optional)
        :param size: This is the size of the plot
        :param save: This is the format you want to save the plot in
        :type save: str
        :param save_name: The name of the file you want to save the visualization as, defaults to dir1 (optional)
        """

        names = self.classifier_model_names()
        sns.set()
        target_class = _check_target(the_y)
        param["model_names"] = names
        FILE_FORMATS = ["pdf", "png"]
        if save not in FILE_FORMATS:
            raise Exception("set save to either 'pdf' or 'png' ")

        if save in FILE_FORMATS:
            if isinstance(save_name, str) is False:
                raise ValueError("You can only set a string to save_name")

            if save_name is None:
                raise Exception("Please set a value to save_name")

        if file_path:
            if save is None:
                raise Exception(
                    "set save to either 'pdf' or 'png' before defining a file path"
                )

        if save is None:
            if save_name:
                raise Exception(
                    "You can only use save_name after param save is defined"
                )

        if kf is True and t_split is True:
            raise Exception(
                "set kf to True if you used KFold or set t_split to True"
                "if you used the split method."
            )
        if kf is True:

            plt.figure(figsize=size)
            plot = sns.barplot(x="model_names", y="Accuracy", data=param)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
            plt.title("ACCURACY")

            plt.figure(figsize=size)
            plot1 = sns.barplot(x="model_names", y="Precision", data=param)
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("PRECISION")

            plt.figure(figsize=size)
            plot2 = sns.barplot(x="model_names", y="Recall", data=param)
            plot2.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("RECALL")

            plt.figure(figsize=size)
            plot3 = sns.barplot(x="model_names", y="f1", data=param)
            plot3.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("F1 SCORE")

            plt.figure(figsize=size)
            plot1 = sns.barplot(x="model_names", y="r2", data=param)
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("R2 SCORE")

            plt.figure(figsize=size)
            plot1 = sns.barplot(
                x="model_names", y="Standard Deviation of Accuracy", data=param
            )
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("STANDARD DEVIATION")

            if save == "pdf":
                name = save_name + ".pdf"
                img(name, FILE_PATH=file_path, type_="file")

            elif save == "png":
                name = save_name
                img(FILENAME=name, FILE_PATH=file_path, type_="picture")

            display(plot)
            display(plot1)

        elif t_split is True:
            if target_class == "binary":
                plt.figure(figsize=size)
                plot = sns.barplot(x="model_names", y="Accuracy", data=param)
                plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
                plt.title("ACCURACY")

                plt.figure(figsize=size)
                plot1 = sns.barplot(x="model_names", y="r2 score", data=param)
                plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
                plt.title("R2 SCORE")

                plt.figure(figsize=size)
                plot2 = sns.barplot(x="model_names", y="ROC AUC", data=param)
                plot2.set_xticklabels(plot2.get_xticklabels(), rotation=90)
                plt.title("ROC AUC")

                plt.figure(figsize=size)
                plot3 = sns.barplot(x="model_names", y="f1 score", data=param)
                plot3.set_xticklabels(plot3.get_xticklabels(), rotation=90)
                plt.title("F1 SCORE")

                plt.figure(figsize=size)
                plot4 = sns.barplot(x="model_names", y="Precision", data=param)
                plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
                plt.title("PRECISION")

                plt.figure(figsize=size)
                plot5 = sns.barplot(x="model_names", y="Recall", data=param)
                plot5.set_xticklabels(plot5.get_xticklabels(), rotation=90)
                plt.title("RECALL")

                display(plot)
                display(plot1)
                display(plot2)
                display(plot3)
                display(plot4)
                display(plot5)

                if save == "pdf":
                    name = save_name + ".pdf"
                    img(name, FILE_PATH=file_path, type_="file")
                elif save == "png":
                    name = save_name
                    img(FILENAME=name, FILE_PATH=file_path, type_="picture")

            elif target_class == "multiclass":
                plt.figure(figsize=size)
                plot = sns.barplot(x="model_names", y="Accuracy", data=param)
                plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
                plt.title("ACCURACY")

                plt.figure(figsize=size)
                plot1 = sns.barplot(x="model_names", y="r2 score", data=param)
                plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
                plt.title("R2 SCORE")

                plt.figure(figsize=size)
                plot2 = sns.barplot(x="model_names", y="f1 score", data=param)
                plot2.set_xticklabels(plot2.get_xticklabels(), rotation=90)
                plt.title("F1 SCORE")

                plt.figure(figsize=size)
                plot3 = sns.barplot(x="model_names", y="Precision", data=param)
                plot3.set_xticklabels(plot3.get_xticklabels(), rotation=90)
                plt.title("PRECISION")

                plt.figure(figsize=size)
                plot4 = sns.barplot(x="model_names", y="Recall", data=param)
                plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
                plt.title("RECALL")

                display(plot)
                display(plot1)
                display(plot2)
                display(plot3)
                display(plot4)

                if save == "pdf":
                    name = save_name + ".pdf"
                    img(name, FILE_PATH=file_path, type_="file")
                elif save == "png":
                    name = save_name
                    img(FILENAME=name, FILE_PATH=file_path, type_="picture")

    def show(
        self,
        param: {__setitem__},
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        save: bool = False,
        save_name=None,
    ):
        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

        :param save:
        :param target:
        :param file_path:
        :param param: {__setitem__}
        :type param: {__setitem__}
        :param kf: set to True if you used KFold, defaults to False
        :type kf: bool (optional)
        :param t_split: True if you used the split method, defaults to False
        :type t_split: bool (optional)
        :param size: This is the size of the plot
        :param save_name: The name of the file you want to save the visualization as.
        """

        names = self.classifier_model_names()
        param["model_names"] = names
        target_class = _check_target(the_y)
        if kf is True:
            if t_split is True:
                raise Exception(
                    "set kf to True if you used KFold or set t_split to True"
                    "if you used the split method."
                )
            if target_class == "binary":
                IMAGE_COLUMNS = []
                for i in range(len(self.kf_binary_columns_train)):
                    IMAGE_COLUMNS.append(self.kf_binary_columns_train[i] + ".png")

                if save is True:
                    dire = directory(save_name)
                for j in range(len(IMAGE_COLUMNS)):

                    fig = px.bar(
                        data_frame=param,
                        x="model_names",
                        y=self.kf_binary_columns_train[j],
                        hover_data=[self.kf_binary_columns_train[j], "model_names"],
                        color="Time Taken(s)",
                    )
                    display(fig)
                    if save is True:
                        if save_name is None:
                            raise Exception("set save to True before using save_name")

                        else:
                            img_plotly(
                                name=IMAGE_COLUMNS[j],
                                figure=fig,
                                label=target_class,
                                FILENAME=dire,
                                FILE_PATH=file_path,
                            )

            elif target_class == "multiclass":
                IMAGE_COLUMNS = []
                for i in range(len(self.kf_multiclass_columns_train)):
                    IMAGE_COLUMNS.append(self.kf_multiclass_columns_train[i] + ".png")

                if save is True:
                    dire = directory(save_name)

                for j in range(len(self.kf_multiclass_columns_train)):
                    fig = px.bar(
                        data_frame=param,
                        x="model_names",
                        y=self.kf_multiclass_columns_train[j],
                        hover_data=[self.kf_multiclass_columns_train[j], "model_names"],
                        color="Time Taken(s)",
                    )
                    display(fig)
                    if save is True:
                        if save_name is None:
                            raise Exception("set save to True before using save_name")

                        elif save_name:
                            img_plotly(
                                name=IMAGE_COLUMNS[j],
                                figure=fig,
                                label=target_class,
                                FILENAME=dire,
                                FILE_PATH=file_path,
                            )

        if t_split is True:
            if kf is True:
                raise Exception(
                    "set kf to True if you used KFold or set t_split to True"
                    "if you used the split method."
                )

            if target_class == "binary":
                IMAGE_COLUMNS = []
                for i in range(len(self.t_split_binary_columns_test)):
                    IMAGE_COLUMNS.append(self.t_split_binary_columns_test[i] + ".png")

                if save is True:
                    dire = directory(save_name)
                for j in range(len(IMAGE_COLUMNS)):

                    fig = px.bar(
                        data_frame=param,
                        x="model_names",
                        y=self.t_split_binary_columns_test[j],
                        hover_data=[self.t_split_binary_columns_test[j], "model_names"],
                        color="execution time(seconds)",
                    )
                    display(fig)
                    if save is True:
                        if save_name is None:
                            raise Exception("set save to True before using save_name")

                        else:
                            img_plotly(
                                name=IMAGE_COLUMNS[j],
                                figure=fig,
                                label=target_class,
                                FILENAME=dire,
                                FILE_PATH=file_path,
                            )

            elif target_class == "multiclass":
                IMAGE_COLUMNS = []
                for i in range(len(self.t_split_multiclass_columns_test)):
                    IMAGE_COLUMNS.append(
                        self.t_split_multiclass_columns_test[i] + ".png"
                    )

                if save is True:
                    dire = directory(save_name)

                for j in range(len(self.t_split_multiclass_columns_test)):
                    fig = px.bar(
                        data_frame=param,
                        x="model_names",
                        y=self.t_split_multiclass_columns_test[j],
                        hover_data=[
                            self.t_split_multiclass_columns_test[j],
                            "model_names",
                        ],
                        color="execution time(seconds)",
                    )
                    display(fig)
                    if save is True:
                        if save_name is None:
                            raise Exception("set save to True before using save_name")

                        elif save_name:
                            img_plotly(
                                name=IMAGE_COLUMNS[j],
                                figure=fig,
                                label=target_class,
                                FILENAME=dire,
                                FILE_PATH=file_path,
                            )
