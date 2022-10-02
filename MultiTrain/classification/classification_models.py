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

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier

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
    img,
    img_plotly,
    kf_best_model,
    write_to_excel,
    _check_target,
    _get_cat_num,
    _fill,
    _fill_columns,
    _dummy,
)

# os.environ['OMP_NUM_THREADS'] = "1"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class MultiClassifier:
    def __init__(
        self,
        cores: Optional[int] = None,
        random_state: int = randint(1000),
        verbose: bool = False,
        imbalanced: bool = False,
        sampling: str = None,
        strategy: str or float = "auto",
        select_models: Union[list, tuple] = None,
    ) -> None:

        self.cores = cores
        self.random_state = random_state
        self.verbose = verbose
        self.imbalanced = imbalanced
        self.sampling = sampling
        self.strategy = strategy
        self.select_models = select_models
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

    def _select_few_models(self):

        model_dict = {
            "LogisticRegression": LogisticRegression(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "LogisticRegressionCV": LogisticRegressionCV(n_jobs=self.cores, refit=True),
            "SGDClassifier": SGDClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "PassiveAggressiveClassifier": PassiveAggressiveClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "GradientBoostingClassifier": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
                random_state=self.random_state
            ),
            "AdaBoostClassifier": AdaBoostClassifier(random_state=self.random_state),
            "CatBoostClassifier": CatBoostClassifier(
                thread_count=self.cores,
                verbose=False,
                random_state=self.random_state,
            ),
            "XGBClassifier": XGBClassifier(
                eval_metric="mlogloss",
                n_jobs=self.cores,
                refit=True,
                random_state=self.random_state,
            ),
            "GuassianNB": GaussianNB(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "KNeighborsClassifier": KNeighborsClassifier(n_jobs=self.cores),
            "MLPClassifier": MLPClassifier(random_state=self.random_state),
            "SVC": SVC(random_state=self.random_state),
            "DecisionTreeClassifier": DecisionTreeClassifier(
                random_state=self.random_state
            ),
            "BernoulliNB": BernoulliNB(),
            "MultinomialNB": MultinomialNB(),
            "ComplementNB": ComplementNB(),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "RidgeClassifier": RidgeClassifier(random_state=self.random_state),
            "ExtraTreeClassifier": ExtraTreeClassifier(random_state=self.random_state),
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
            "LinearSVC": LinearSVC(random_state=self.random_state),
            "BaggingClassifier": BaggingClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "BalancedBaggingClassifier": BalancedBaggingClassifier(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "Perceptron": Perceptron(n_jobs=self.cores, random_state=self.random_state),
            "NuSVC": NuSVC(random_state=self.random_state),
            "LGBMClassifier": LGBMClassifier(random_state=self.random_state),
        }
        return model_dict

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

        else:
            raise ValueError(
                f"{self.sampling} is not a valid sampler. Call the 'strategies' method to view the lists "
                f" of all valid samplers"
            )

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

    def classifier_model_names(self) -> list:
        model_names = [
            "LogisticRegression",
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
            thread_count=self.cores,
            verbose=False,
            random_state=self.random_state,
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

    def _custom(self):
        if type(self.select_models) not in [tuple, list]:
            raise TypeError(
                f"received type {type(self.select_models)} for select_models parameter, expected list or "
                f"tuple"
            )

        custom_models = []
        name = self.select_models
        for i in self.select_models:
            for key, value in self._select_few_models().items():
                if i == key:
                    custom_models.append(value)
                elif i not in self.classifier_model_names():
                    raise ValueError(
                        f'{i} unknown, use the "classifier_model_names" method to view the classifier '
                        f"algorithms available"
                    )
        return custom_models, name

    def _get_index(self, df, the_best):
        if self.select_models is None:
            name = list(self.classifier_model_names())
            MODEL = self._initialize_()
        else:
            MODEL, name = self._custom()
        df["model_names"] = name
        high = [
            "Accuracy",
            "Balanced Accuracy",
            "Test Acc",
            "f1 score",
            "f1",
            "f1 Macro",
            "Test f1",
            "Test f1 Macro",
            "Precision",
            "Precision Macro",
            "Test Precision",
            "Test Precision Macro",
            "Recall",
            "Recall Macro",
            "Test Recall",
            "Test Recall Macro",
            "ROC AUC",
        ]
        low = ["mean absolute error", "mean squared error", "Test std"]

        if the_best in high:
            best_model_details = df[df[the_best] == df[the_best].max()]

        elif the_best in low:
            best_model_details = df[df[the_best] == df[the_best].min()]

        else:
            raise Exception(
                f"metric {the_best} not found, refer to the resulting dataframe from fit to select a metric"
            )

        best_model_details = best_model_details.reset_index()
        best_model_name = best_model_details.iloc[0]["model_names"]
        index_ = name.index(best_model_name)
        return MODEL[index_]

    def _startKFold_(self, param, param_X, param_y, param_cv, train_score):
        if self.select_models is None:
            names = self.classifier_model_names()
        else:
            names = self.select_models
        target_class = _check_target(param_y)
        if self.imbalanced is True:
            logger.info(
                "You are receiving this message because you set imbalanced to True. All resampling techniques "
                "e.g SMOTE has been disabled in this new version till a permanent fix is implemented, "
                "use the split method instead if you're dealing with imbalanced data"
            )
        if target_class == "binary":
            dataframe = {}
            bar = trange(
                len(param),
                desc="Training in progress: ",
                bar_format="{desc}{percentage:3.0f}% {bar}{remaining} [{n_fmt}/{total_fmt} {postfix}]",
            )
            for i in bar:
                bar.set_postfix({"Model ": names[i]})

                if self.verbose is True:
                    print(names[i])

                score = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")

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
                    mean_train_recall = scores["train_recall"].mean()
                    mean_test_recall = scores["test_recall"].mean()
                    train_stdev = scores["train_accuracy"].std()
                    test_stdev = scores["test_accuracy"].std()
                    overfitting = (
                        True if (mean_train_acc - mean_test_acc) > 0.1 else False
                    )
                except Exception:
                    logger.error(f"{names[i]} unable to fit properly")
                    seconds, mean_train_acc, mean_test_acc = (
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                    mean_train_bacc, mean_test_bacc = np.nan, np.nan
                    mean_train_precision, mean_test_precision = np.nan, np.nan
                    (mean_train_f1, mean_test_f1) = np.nan, np.nan
                    (
                        mean_train_recall,
                        mean_test_recall,
                        train_stdev,
                        test_stdev,
                        overfitting,
                    ) = (np.nan, np.nan, np.nan, np.nan, False)

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
                #        mean_train_f1, mean_test_f1 = np.nan, np.nan
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
                        mean_test_recall,
                        test_stdev,
                        seconds,
                    ]
                    dataframe.update({names[i]: scores_df})
            return dataframe

        elif target_class == "multiclass":
            dataframe = {}
            bar = trange(
                len(param),
                desc="Training in progress: ",
                bar_format="{desc}{percentage:3.0f}% {bar}{remaining} [{n_fmt}/{total_fmt} {postfix}]",
            )
            for i in bar:
                bar.set_postfix({"Model ": names[i]})
                start = time.time()
                score = ("precision_macro", "recall_macro", "f1_macro")
                scores = cross_validate(
                    estimator=param[i],
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

                    dataframe.update({names[i]: scores_df})

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

                    dataframe.update({names[i]: scores_df})

            return dataframe

    def fit(
        self,
        X=None,
        y=None,
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

        global y_te, pred
        if self.cores is None:
            logger.info(
                "It is advisable to set cores in the MultiClassifier object to -1 to use all cores in the "
                "cpu, this reduces training time significantly"
            )
        try:
            if splitting is True:
                target_class = (
                    _check_target(y) if y is not None else _check_target(split_data[3])
                )
                if splitting and split_data:
                    X_tr, X_te, y_tr, y_te = (
                        split_data[0],
                        split_data[1],
                        split_data[2],
                        split_data[3],
                    )

                if self.select_models is None:
                    model = self._initialize_()
                    names = self.classifier_model_names()
                else:
                    model, names = self._custom()
                dataframe = {}
                bar = trange(
                    len(model),
                    desc="Training in progress: ",
                    bar_format="{desc}{percentage:3.0f}% {bar}{remaining} [{n_fmt}/{total_fmt} {postfix}]",
                )
                for index in bar:
                    bar.set_postfix({"Model ": names[index]})
                    start = time.time()

                    if text is False:
                        if self.imbalanced is False:
                            try:
                                model[index].fit(X_tr, y_tr)
                            except ValueError:
                                logger.error(f"{names[index]} unable to fit properly")
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
                                model[index].fit(X_tr_, y_tr_)
                            except ValueError:
                                logger.error(f"{names[index]} unable to fit properly")
                                pass

                        end = time.time()

                        try:

                            pred = model[index].predict(X_te)

                            pred_train = model[index].predict(X_tr)
                        except AttributeError:
                            pass

                    elif text is True:

                        if vectorizer == "count":
                            try:
                                try:
                                    pipeline = make_pipeline(
                                        CountVectorizer(ngram_range=ngrams),
                                        model[index],
                                    )

                                    pipeline.fit(X_tr, y_tr)
                                    pred = pipeline.predict(X_te)
                                    pred_train = pipeline.predict(X_tr)

                                except Exception:
                                    # This is a fix for the error below when using gradient boosting classifier or
                                    # HistGradientBoostingClassifier TypeError: A sparse matrix was passed,
                                    # but dense data is required. Use X.toarray() to convert to a dense numpy array.
                                    pipeline = make_pipeline(
                                        CountVectorizer(ngram_range=ngrams),
                                        FunctionTransformer(
                                            lambda x: x.todense(),
                                            accept_sparse=True,
                                        ),
                                        model[index],
                                    )
                                    pipeline.fit(X_tr, y_tr)
                                    pred = pipeline.predict(X_te)

                                    pred_train = pipeline.predict(X_tr)

                            except Exception:
                                logger.error(f"{names[index]} unable to fit properly")

                        elif vectorizer == "tfidf":
                            try:
                                try:
                                    pipeline = make_pipeline(
                                        TfidfVectorizer(ngram_range=ngrams),
                                        model[index],
                                    )

                                    pipeline.fit(X_tr, y_tr)
                                    pred = pipeline.predict(X_te)

                                    pred_train = pipeline.predict(X_tr)

                                except Exception:
                                    # This is a fix for the error below when using gradient boosting classifier or
                                    # HistGradientBoostingClassifier TypeError: A sparse matrix was passed,
                                    # but dense data is required. Use X.toarray() to convert to a dense numpy array.
                                    pipeline = make_pipeline(
                                        TfidfVectorizer(ngram_range=ngrams),
                                        FunctionTransformer(
                                            lambda x: x.todense(),
                                            accept_sparse=True,
                                        ),
                                        model[index],
                                    )
                                    pipeline.fit(X_tr, y_tr)

                            except Exception:
                                logger.error(f"{names[index]} unable to fit properly")

                        end = time.time()

                    true = y_te
                    true_train = y_tr

                    acc = accuracy_score(true, pred)
                    bacc = balanced_accuracy_score(true, pred)
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
                            tpre = precision_score(
                                true_train, pred_train, average="micro"
                            )
                            trec = recall_score(true_train, pred_train, average="micro")

                        elif self.imbalanced is False:
                            tf1 = f1_score(true_train, pred_train, average="macro")
                            tpre = precision_score(
                                true_train, pred_train, average="macro"
                            )
                            trec = recall_score(true_train, pred_train, average="macro")

                    overfit = True if (tacc - acc) > 0.1 else False
                    time_taken = round(end - start, 2)

                    if show_train_score is False:
                        eval_bin = [
                            overfit,
                            acc,
                            bacc,
                            roc,
                            f1,
                            pre,
                            rec,
                            time_taken,
                        ]
                        eval_mul = [
                            overfit,
                            acc,
                            bacc,
                            f1,
                            pre,
                            rec,
                            time_taken,
                        ]

                    elif show_train_score is True:
                        eval_bin = [
                            overfit,
                            tacc,
                            acc,
                            tbacc,
                            bacc,
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
                            tf1,
                            f1,
                            tpre,
                            pre,
                            trec,
                            rec,
                            time_taken,
                        ]

                    if target_class == "binary":
                        dataframe.update({names[index]: eval_bin})
                    elif target_class == "multiclass":
                        dataframe.update({names[index]: eval_mul})

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
                target_class = (
                    _check_target(y) if y is not None else _check_target(split_data[3])
                )
                # Fitting the models and predicting the values of the test set.
                if self.select_models is None:
                    KFoldModel = self._initialize_()
                    names = self.classifier_model_names()
                else:
                    KFoldModel, names = self._custom()

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
                            dataframe,
                            orient="index",
                            columns=self.kf_binary_columns_train,
                        )

                    elif show_train_score is False:
                        df = pd.DataFrame.from_dict(
                            dataframe,
                            orient="index",
                            columns=self.kf_binary_columns_test,
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

        except Exception:
            raise_text_error(text, vectorizer, ngrams)
            raise_imbalanced_error(self.imbalanced, self.sampling)
            raise_kfold1_error(kf, splitting, split_data)
            raise_split_data_error(split_data, splitting)
            raise_fold_type_error(fold)
            raise_kfold2_error(kf, X, y)
            raise_splitting_error(splitting, split_data)

    def use_model(self, df, model: str = None, best: str = None):
        """


        :param df: the dataframe object
        :param model: name of the classifier algorithm
        :param best: the evaluation metric used to find the best model

        :return:
        """

        if self.select_models is None:
            name = self.classifier_model_names()
            MODEL = self._initialize_()
        else:
            MODEL, name = self._custom()

        if model is not None and best is not None:
            raise Exception("You can only use one of the two arguments.")

        if model:
            assert (
                model in name
            ), f"name {model} is not found, here is a list of the available models to work with: {name}"
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
        param: pd.DataFrame,
        y: any = None,
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        size=(15, 8),
        save: str = None,
        save_name: str = None,
    ):

        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

        :param y:
        :param file_path:
        :param param: pd.DataFrame
        :type param: pd.DataFrame
        :param kf: set to True if you used KFold, defaults to False
        :type kf: bool (optional)
        :param t_split: True if you used the split method, defaults to False
        :type t_split: bool (optional)
        :param size: This is the size of the plot
        :param save: This is the format you want to save the plot in
        :type save: str
        :param save_name: The name of the file you want to save the visualization as, defaults to dir1 (optional)
        """

        if self.select_models is None:
            names = self.classifier_model_names()
        else:
            names = self.select_models
        sns.set()
        target_class = _check_target(y)
        param["model_names"] = names
        FILE_FORMATS = ["pdf", "png"]

        if save is not None:
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
            plot4 = sns.barplot(
                x="model_names", y="Standard Deviation of Accuracy", data=param
            )
            plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
            plt.title("STANDARD DEVIATION")

            if save == "pdf":
                name = save_name + ".pdf"
                img(name, FILE_PATH=file_path, type_="file")

            elif save == "png":
                name = save_name
                img(FILENAME=name, FILE_PATH=file_path, type_="picture")

            display(plot)
            display(plot1)
            display(plot2)
            display(plot3)
            display(plot4)

        elif t_split is True:
            if target_class == "binary":
                plt.figure(figsize=size)
                plot = sns.barplot(x="model_names", y="Accuracy", data=param)
                plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
                plt.title("ACCURACY")

                plt.figure(figsize=size)
                plot1 = sns.barplot(x="model_names", y="ROC AUC", data=param)
                plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
                plt.title("ROC AUC")

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

            elif target_class == "multiclass":
                plt.figure(figsize=size)
                plot = sns.barplot(x="model_names", y="Accuracy", data=param)
                plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
                plt.title("ACCURACY")

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
        param: pd.DataFrame,
        y: any = None,
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        save: bool = False,
        save_name: str = None,
    ):
        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

        :param y:
        :param save:
        :param target:
        :param file_path:
        :param param: pd.DataFrame
        :type param: pd.DataFrame
        :param kf: set to True if you used KFold, defaults to False
        :type kf: bool (optional)
        :param t_split: True if you used the split method, defaults to False
        :type t_split: bool (optional)
        :param size: This is the size of the plot
        :param save_name: The name of the file you want to save the visualization as.
        """
        if self.select_models is None:
            names = self.classifier_model_names()
        else:
            names = self.select_models
        param["model_names"] = names
        target_class = _check_target(y)
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
                        hover_data=[
                            self.kf_binary_columns_train[j],
                            "model_names",
                        ],
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
                        hover_data=[
                            self.kf_multiclass_columns_train[j],
                            "model_names",
                        ],
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
                        hover_data=[
                            self.t_split_binary_columns_test[j],
                            "model_names",
                        ],
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
