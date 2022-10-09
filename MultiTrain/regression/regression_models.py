import time
from operator import __setitem__
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from numpy.random import randint
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    PoissonRegressor,
    GammaRegressor,
    HuberRegressor,
    RidgeCV,
    BayesianRidge,
    ElasticNetCV,
    LassoCV,
    LassoLarsIC,
    LassoLarsCV,
    Lars,
    LarsCV,
    SGDRegressor,
    TweedieRegressor,
    RANSACRegressor,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    OrthogonalMatchingPursuit,
    LassoLars,
    ARDRegression,
    TheilSenRegressor,
    Ridge,
    ElasticNet,
    Lasso,
    LinearRegression,
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
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    HalvingRandomSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
)
from sklearn.svm import LinearSVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from skopt import BayesSearchCV
from skopt.learning import (
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    RandomForestRegressor,
)
from tqdm.notebook import trange
from xgboost import XGBRegressor

from MultiTrain.errors.fit_exceptions import (
    raise_kfold1_error,
    raise_fold_type_error,
    raise_kfold2_error,
    raise_splitting_error,
    raise_split_data_error,
)

from MultiTrain.errors.split_exceptions import (
    feature_label_type_error,
    strat_error,
    dimensionality_reduction_type_error,
    test_size_error,
    missing_values_error,
)

from MultiTrain.methods.multitrain_methods import (
    write_to_excel,
    show_best,
    img,
    directory,
    img_plotly,
    _fill_columns,
    _fill,
    _get_cat_num,
    _dummy,
)

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
from MultiTrain.base_model import BaseModel


class MultiRegressor(BaseModel):
    def __init__(
        self,
        cores: Optional[int] = None,
        random_state: int = randint(1000),
        verbose: bool = False,
        select_models: Union[list, tuple, None] = None,
    ):
        self.select_models = select_models
        self.cores = cores
        self.random_state = random_state
        self.verbose = verbose
        self.high = [
            "Neg Mean Absolute Error",
            "Neg Root Mean Squared Error",
            "r2 score",
            "Neg Root Mean Squared Log Error",
            "Neg Median Absolute Error",
            "Neg Median Absolute Percentage Error",
        ]

        self.low = [
            "Mean Absolute Error",
            "Root Mean Squared Error",
            "Root Mean Squared Log Error",
            "Median Absolute Error",
            "Mean Absolute Percentage Error",
        ]

    def _model_names(self):
        """Gives the all the regression model names in sklearn

        Returns:
            list: list of the regression names in sklearn package
        """
        model_names = [
            "Linear Regression",
            "Random Forest Regressor",
            "XGBRegressor",
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "SVR",
            "BaggingRegressor",
            "NuSVR",
            "ExtraTreeRegressor",
            "ExtraTreesRegressor",
            "AdaBoostRegressor",
            "PoissonRegressor",
            "LGBMRegressor",
            "KNeighborsRegressor",
            "DecisionTreeRegressor",
            "MLPRegressor",
            "HuberRegressor",
            "GammaRegressor",
            "LinearSVR",
            "RidgeCV",
            "Ridge",
            "BayesianRidge",
            "TransformedTargetRegressor",
            "ElasticNetCV",
            "ElasticNet",
            "LassoCV",
            "LassoLarsIC",
            "LassoLarsCV",
            "Lars",
            "LarsCV",
            "SGDRegressor",
            "TweedieRegressor",
            "Lasso",
            "RANSACRegressor",
            "OrthogonalMatchingPursuitCV",
            "PassiveAggressiveRegressor",
            "GaussianProcessRegressor",
            "OrthogonalMatchingPursuit",
            "DummyRegressor",
            "LassoLars",
            "KernelRidge",
            "ARDRegression",
            "TheilSenRegressor",
        ]
        return model_names

    def _select_few_models(self):
        model_dict = {
            "LinearRegression": LinearRegression(n_jobs=self.cores),
            "RandomForestRegressor": RandomForestRegressor(
                random_state=self.random_state
            ),
            "XGBRegressor": XGBRegressor(random_state=self.random_state),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                random_state=self.random_state
            ),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                random_state=self.random_state
            ),
            "SVR": SVR(),
            "BaggingRegressor": BaggingRegressor(random_state=self.random_state),
            "NuSVR": NuSVR(),
            "ExtraTreeRegressor": ExtraTreeRegressor(random_state=self.random_state),
            "ExtraTreesRegressor": ExtraTreesRegressor(random_state=self.random_state),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=self.random_state),
            "PoissonRegressor": PoissonRegressor(),
            "LGBMRegressor": LGBMRegressor(random_state=self.random_state),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(
                random_state=self.random_state
            ),
            "MLPRegressor": MLPRegressor(random_state=self.random_state),
            "HuberRegressor": HuberRegressor(),
            "GammaRegressor": GammaRegressor(),
            "LinearSVR": LinearSVR(random_state=self.random_state),
            "RidgeCV": RidgeCV(),
            "Ridge": Ridge(random_state=self.random_state),
            "BayesianRidge": BayesianRidge(),
            "TransformedTargetRegressor": TransformedTargetRegressor(),
            "ElasticNetCV": ElasticNetCV(
                n_jobs=self.cores, random_state=self.random_state
            ),
            "ElasticNet": ElasticNet(random_state=self.random_state),
            "LassoCV": LassoCV(n_jobs=self.cores, random_state=self.random_state),
            "LassoLarsIC": LassoLarsIC(),
            "LassoLarsCV": LassoLarsCV(),
            "Lars": Lars(random_state=self.random_state),
            "LarsCV": LarsCV(n_jobs=self.cores),
            "SGDRegressor": SGDRegressor(random_state=self.random_state),
            "TweedieRegressor": TweedieRegressor(),
            "Lasso": Lasso(random_state=self.random_state),
            "RANSACRegressor": RANSACRegressor(random_state=self.random_state),
            "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV(
                n_jobs=self.cores
            ),
            "PassiveAggressiveRegressor": PassiveAggressiveRegressor(
                random_state=self.random_state
            ),
            "GaussianProcessRegressor": GaussianProcessRegressor(
                random_state=self.random_state
            ),
            "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
            "DummyRegressor": DummyRegressor(),
            "LassoLars": LassoLars(random_state=self.random_state),
            "KernelRidge": KernelRidge(),
            "ARDRegression": ARDRegression(),
            "TheilSenRegressor": TheilSenRegressor(
                n_jobs=self.cores, random_state=self.random_state
            ),
        }
        return model_dict

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
                elif i not in self._model_names():
                    raise ValueError(
                        f'{i} unknown, use the "_model_names" method to view the regression algorithms '
                        f"available "
                    )
        return custom_models, name

    ### Split was here

    def _initialize(self):
        """
        It _initializes all the models that we will be using in our ensemble
        """
        # __n_neighbors = round(np.sqrt(self.__shape[0]))

        lr = LinearRegression(n_jobs=self.cores)
        rfr = RandomForestRegressor(random_state=self.random_state)
        xgb = XGBRegressor(random_state=self.random_state)
        gbr = GradientBoostingRegressor(random_state=self.random_state)
        hgbr = HistGradientBoostingRegressor(random_state=self.random_state)
        svr = SVR()
        br = BaggingRegressor(random_state=self.random_state)
        nsvr = NuSVR()
        etr = ExtraTreeRegressor(random_state=self.random_state)
        etrs = ExtraTreesRegressor(random_state=self.random_state)
        ada = AdaBoostRegressor(random_state=self.random_state)
        pr = PoissonRegressor()
        lgbm = LGBMRegressor(random_state=self.random_state)
        knr = KNeighborsRegressor()
        dtr = DecisionTreeRegressor(random_state=self.random_state)
        mlp = MLPRegressor(random_state=self.random_state)
        hub = HuberRegressor()
        gmr = GammaRegressor()
        lsvr = LinearSVR(random_state=self.random_state)
        ridg = RidgeCV()
        rid = Ridge(random_state=self.random_state)
        byr = BayesianRidge()
        ttr = TransformedTargetRegressor()
        eltcv = ElasticNetCV(n_jobs=self.cores, random_state=self.random_state)
        elt = ElasticNet(random_state=self.random_state)
        lcv = LassoCV(n_jobs=self.cores, random_state=self.random_state)
        llic = LassoLarsIC()
        llcv = LassoLarsCV()
        lars = Lars(random_state=self.random_state)
        lrcv = LarsCV(n_jobs=self.cores)
        sgd = SGDRegressor(random_state=self.random_state)
        twr = TweedieRegressor()
        lass = Lasso(random_state=self.random_state)
        ranr = RANSACRegressor(random_state=self.random_state)
        ompc = OrthogonalMatchingPursuitCV(n_jobs=self.cores)
        par = PassiveAggressiveRegressor(random_state=self.random_state)
        gpr = GaussianProcessRegressor(random_state=self.random_state)
        ompu = OrthogonalMatchingPursuit()
        dr = DummyRegressor()
        lassla = LassoLars(random_state=self.random_state)
        krid = KernelRidge()
        ard = ARDRegression()
        theil = TheilSenRegressor(n_jobs=self.cores, random_state=self.random_state)

        return (
            lr,
            rfr,
            xgb,
            gbr,
            hgbr,
            svr,
            br,
            nsvr,
            etr,
            etrs,
            ada,
            pr,
            lgbm,
            knr,
            dtr,
            mlp,
            hub,
            gmr,
            lsvr,
            ridg,
            rid,
            byr,
            ttr,
            eltcv,
            elt,
            lcv,
            llic,
            llcv,
            lars,
            lrcv,
            sgd,
            twr,
            lass,
            ranr,
            ompc,
            par,
            gpr,
            ompu,
            dr,
            lassla,
            krid,
            ard,
            theil,
        )

    def _get_index(self, df, the_best):
        if self.select_models is None:
            name = list(self._model_names())
            MODEL = self._initialize()
        else:
            MODEL, name = self._custom()
        df["model_names"] = name

        if the_best in self.high:
            best_model_details = df[df[the_best] == df[the_best].max()]

        elif the_best in self.low:
            best_model_details = df[df[the_best] == df[the_best].min()]

        else:
            raise Exception(f"metric {the_best} not found")

        best_model_details = best_model_details.reset_index()
        best_model_name = best_model_details.iloc[0]["model_names"]
        index_ = name.index(best_model_name)
        return MODEL[index_]

    def startKFold(self, param, param_X, param_y, param_cv, train_score):

        """_summary_

        Args:
            param (any): _description_
            param_X (any): _description_
            param_y (any): _description_
            param_cv (any): _description_
            train_score (int or float): _description_

        Returns:
            df: dataframe
        """

        if self.select_models is None:
            names = self._model_names()
        else:
            names = self.select_models

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
            start = time.time()
            score = (
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "neg_mean_squared_error",
                "r2",
                "neg_median_absolute_error",
                "neg_mean_squared_log_error",
                "neg_mean_absolute_percentage_error",
            )

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
                mean_train_mae = scores["train_neg_mean_absolute_error"].mean()
                mean_test_mae = scores["test_neg_mean_absolute_error"].mean()
                mean_train_rmse = scores["train_neg_root_mean_squared_error"].mean()
                mean_test_rmse = scores["test_neg_root_mean_squared_error"].mean()
                mean_train_r2 = scores["train_r2"].mean()
                mean_test_r2 = scores["test_r2"].mean()
                mean_train_rmsle = np.sqrt(
                    scores["train_neg_mean_squared_log_error"].mean()
                )
                mean_test_rmsle = np.sqrt(
                    scores["test_neg_mean_squared_log_error"].mean()
                )
                mean_train_meae = scores["train_neg_median_absolute_error"].mean()
                mean_test_meae = scores["test_neg_median_absolute_error"].mean()
                mean_train_mape = scores[
                    "train_neg_mean_absolute_percentage_error"
                ].mean()
                mean_test_mape = scores[
                    "test_neg_mean_absolute_percentage_error"
                ].mean()

                # scores = scores.tolist()
                scores_df = [
                    mean_train_mae,
                    mean_test_mae,
                    mean_train_rmse,
                    mean_test_rmse,
                    mean_train_r2,
                    mean_test_r2,
                    mean_train_rmsle,
                    mean_test_rmsle,
                    mean_train_meae,
                    mean_test_meae,
                    mean_train_mape,
                    mean_test_mape,
                    seconds,
                ]
                dataframe.update({names[i]: scores_df})

            elif train_score is False:
                mean_test_mae = scores["test_neg_mean_absolute_error"].mean()
                mean_test_rmse = scores["test_neg_root_mean_squared_error"].mean()
                mean_test_r2 = scores["test_r2"].mean()
                mean_test_rmsle = np.sqrt(
                    scores["test_neg_mean_squared_log_error"].mean()
                )
                mean_test_meae = scores["test_neg_median_absolute_error"].mean()
                mean_test_mape = scores[
                    "test_neg_mean_absolute_percentage_error"
                ].mean()

                scores_df = [
                    mean_test_mae,
                    mean_test_rmse,
                    mean_test_r2,
                    mean_test_rmsle,
                    mean_test_meae,
                    mean_test_mape,
                ]
                dataframe.update({names[i]: scores_df})

            return dataframe

    def fit(
        self,
        X: str = None,
        y: str = None,
        split_data: str = None,
        splitting: bool = False,
        kf: bool = False,
        fold: int = 5,
        excel: bool = False,
        return_best_model: str = None,
        show_train_score: bool = False,
    ):
        """
        If splitting is False, then do nothing. If splitting is True, then assign the values of split_data to the
        variables X_train, X_test, y_train, and y_test

        :param show_train_score:

        :param return_best_model: sorts the resulting dataframe according to the evaluation metric set here

        :param split_self: defaults to False, set to True when you split the data yourself

        :param excel: defaults to False, set to True when you want the dataframe to save to an excel file in your
        current working directory

        :param y: labels

        :param X: features

        :type fold: object
        :param fold: arguments for KFold where 10 is the n_splits, 1 is the random_state and True is to allow shuffling
        :param kf: defaults to False, set to True when you want to use KFold cross validation as your splitting method
        :param X_train: The training data
        :param X_test: The test data
        :param y_train: The training set labels
        :param y_test: The test set labels
        :param split_data: str = None, splitting: bool = False
        :type split_data: str
        :param splitting: bool = False, defaults to False
        :type splitting: bool (optional)
        :param target: defaults to binary, this is used to specify if the target is binary or multiclass
        If using splitting = True
        df = pd.read_csv("nameOfFile.csv")
        X = df.drop("nameOfLabelColumn", axis=1)
        y = df["nameOfLabelColumn")
        the_split_data = split(X = features, y = labels, sizeOfTest=0.3, randomState=42, strat=True, shuffle_data=True)
        fit_eval_models(splitting = True, split_data = the_split_data)

        If using kf = True

        fit(X = features, y = labels, kf = True, fold = (10, 42, True))
        """
        if self.cores is None:
            logger.info(
                "It is advisable to set cores in the MultiClassifier object to -1 to use all cores in the "
                "cpu, this reduces training time significantly"
            )
        try:
            if splitting is True:
                if splitting and split_data:
                    X_tr, X_te, y_tr, y_te = (
                        split_data[0],
                        split_data[1],
                        split_data[2],
                        split_data[3],
                    )
                # self.__shape = X_tr.shape

                if self.select_models is None:
                    model = self._initialize()
                    names = self._model_names()
                else:
                    model, names = self._custom()

                dataframe = {}
                bar = trange(
                    len(model),
                    desc="Training in progress: ",
                    bar_format="{desc}{percentage:3.0f}% {bar}{remaining} [{n_fmt}/{total_fmt} {postfix}]",
                )
                for i in bar:
                    bar.set_postfix({"Model ": names[i]})
                    start = time.time()
                    if self.verbose is True:
                        print(names[i])
                    try:
                        model[i].fit(X_tr, y_tr)
                    except ValueError:
                        X_tr, X_te = X_tr.to_numpy(), X_te.to_numpy()
                        X_tr, X_te = X_tr.reshape(-1, 1), X_te.reshape(-1, 1)

                        y_tr, y_te = y_tr.to_numpy(), y_te.to_numpy()
                        y_tr, y_te = y_tr.reshape(-1, 1), y_te.reshape(-1, 1)

                        model[i].fit(X_tr, y_tr)

                    end = time.time()
                    pred = model[i].predict(X_te)
                    # X_tr is X_train, X_te is X_test, y_tr is y_train, y_te is y_test
                    true = y_te
                    mae = mean_absolute_error(true, pred)
                    rmse = np.sqrt(mean_squared_error(true, pred))
                    r2 = r2_score(true, pred)
                    try:
                        rmsle = np.sqrt(mean_squared_log_error(true, pred))
                    except ValueError:
                        rmsle = np.nan
                    meae = median_absolute_error(true, pred)
                    mape = mean_absolute_percentage_error(true, pred)

                    time_taken = round(end - start, 2)
                    eval_metrics = [mae, rmse, r2, rmsle, meae, mape, time_taken]
                    dataframe.update({names[i]: eval_metrics})

                dataframe_columns = [
                    "Mean Absolute Error",
                    "Root Mean Squared Error",
                    "r2 score",
                    "Root Mean Squared Log Error",
                    "Median Absolute Error",
                    "Mean Absolute Percentage Error",
                    "Time Taken(s)",
                ]
                df = pd.DataFrame.from_dict(
                    dataframe, orient="index", columns=dataframe_columns
                )

                if return_best_model is not None:
                    df = show_best(df, return_best_model, self.high, self.low)

                elif return_best_model is None:
                    display(df.style.highlight_max(color="yellow"))

                write_to_excel(excel, df)
                return df

            elif kf is True:

                # Fitting the models and predicting the values of the test set.
                if self.select_models is None:
                    KFoldModel = self._initialize()
                    names = self._model_names()
                else:
                    KFoldModel, names = self._custom()

                logger.info("Training started")
                dataframe = self.startKFold(
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
                        columns=[
                            "Neg Mean Absolute Error(Train)",
                            "Neg Mean Absolute Error",
                            "Neg Root Mean Squared Error(Train)",
                            "Neg Root Mean Squared Error",
                            "r2(Train)",
                            "r2",
                            "Neg Root Mean Squared Log Error(Train)",
                            "Neg Root Mean Squared Log Error",
                            "Neg Median Absolute Error(Train)",
                            "Neg Median Absolute Error",
                            "Neg Mean Absolute Percentage Error" "(Train)",
                            "Neg Mean Absolute Percentage Error",
                            "Time Taken(s)",
                        ],
                    )

                    if return_best_model is not None:
                        df = show_best(df, return_best_model, self.high, self.low)

                    elif return_best_model is None:
                        display(df.style.highlight_max(color="yellow"))

                    write_to_excel(excel, df)
                    return df

                if show_train_score is False:
                    df = pd.DataFrame.from_dict(
                        dataframe,
                        orient="index",
                        columns=[
                            "Neg Mean Absolute Error",
                            "Neg Root Mean Squared Error",
                            "r2",
                            "Neg Root Mean Squared Log Error",
                            "Neg Median Absolute Error",
                            "Neg Mean Absolute Percentage Error",
                            "Time Taken(s)",
                        ],
                    )
                    if return_best_model is not None:
                        df = show_best(df, return_best_model, self.high, self.low)
                    elif return_best_model is None:
                        display(df.style.highlight_max(color="yellow"))

                    write_to_excel(excel, df)
        except Exception:
            raise_kfold1_error(kf, splitting, split_data)
            raise_split_data_error(split_data, splitting)
            raise_fold_type_error(fold)
            raise_kfold2_error(kf, X, y)
            raise_splitting_error(splitting, split_data)

    ##use_model was here

    def visualize(
        self,
        param: pd.DataFrame,
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        size: tuple = (15, 8),
        save: str = None,
        save_name: str = None,
    ):
        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

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
            names = self._model_names()
        else:
            names = self.select_models
        sns.set()

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
            plot = sns.barplot(x="model_names", y="Neg Mean Absolute Error", data=param)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
            plt.title("Neg Mean Absolute Error")

            plt.figure(figsize=size)
            plot1 = sns.barplot(
                x="model_names", y="Neg Root Mean Squared Error", data=param
            )
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Root Mean Squared Error")

            plt.figure(figsize=size)
            plot2 = sns.barplot(
                x="model_names",
                y="Neg Root Mean Squared Log Error",
                data=param,
            )
            plot2.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Root Mean Squared Log Error")

            plt.figure(figsize=size)
            plot3 = sns.barplot(
                x="model_names", y="Neg Median Absolute Error", data=param
            )
            plot3.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Median Absolute Error")

            plt.figure(figsize=size)
            plot4 = sns.barplot(x="model_names", y="r2", data=param)
            plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
            plt.title("R2 SCORE")

            plt.figure(figsize=size)
            plot5 = sns.barplot(
                x="model_names",
                y="Neg Mean Absolute Percentage Error",
                data=param,
            )
            plot5.set_xticklabels(plot5.get_xticklabels(), rotation=90)
            plt.title("Neg Mean Absolute Percentage Error")

            plt.figure(figsize=size)
            plot6 = sns.barplot(x="model_names", y="Time Taken(s)", data=param)
            plot6.set_xticklabels(plot6.get_xticklabels(), rotation=90)
            plt.title("Time Taken(s)")

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
            display(plot5)
            display(plot6)

        elif t_split is True:
            plt.figure(figsize=size)
            plot = sns.barplot(x="model_names", y="Mean Absolute Error", data=param)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
            plt.title("Mean Absolute Error")

            plt.figure(figsize=size)
            plot1 = sns.barplot(
                x="model_names", y="Root Mean Squared Error", data=param
            )
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Root Mean Squared Error")

            plt.figure(figsize=size)
            plot2 = sns.barplot(
                x="model_names", y="Root Mean Squared Log Error", data=param
            )
            plot2.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Root Mean Squared Log Error")

            plt.figure(figsize=size)
            plot3 = sns.barplot(
                x="model_names", y="Neg Median Absolute Error", data=param
            )
            plot3.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Median Absolute Error")

            plt.figure(figsize=size)
            plot4 = sns.barplot(x="model_names", y="r2", data=param)
            plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
            plt.title("R2 SCORE")

            plt.figure(figsize=size)
            plot5 = sns.barplot(
                x="model_names", y="Mean Absolute Percentage Error", data=param
            )
            plot5.set_xticklabels(plot5.get_xticklabels(), rotation=90)
            plt.title("Mean Absolute Percentage Error")

            plt.figure(figsize=size)
            plot6 = sns.barplot(x="model_names", y="Time Taken(s)", data=param)
            plot6.set_xticklabels(plot6.get_xticklabels(), rotation=90)
            plt.title("Time Taken(s)")

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

    def show(
        self,
        param: pd.DataFrame,
        file_path: any = None,
        kf: bool = False,
        t_split: bool = False,
        save: bool = False,
        save_name=None,
        target="binary",
    ):
        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

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
            names = self._model_names()
        else:
            names = self.select_models

        param["model_names"] = names

        if kf is True:
            if t_split is True:
                raise Exception(
                    "set kf to True if you used KFold or set t_split to True"
                    "if you used the split method."
                )

            IMAGE_COLUMNS = []
            kfold_columns = [
                "Neg Mean Absolute Error",
                "Neg Root Mean Squared Error",
                "r2 score",
                "Neg Root Mean Squared Log Error",
                "Neg Median Absolute Error",
                "Neg Mean Absolute Percentage Error",
                "Time Taken(s)",
            ]
            for i in range(len(kfold_columns)):
                IMAGE_COLUMNS.append(kfold_columns[i] + ".png")

            if save is True:
                dire = directory(save_name)
            for j in range(len(IMAGE_COLUMNS)):

                fig = px.bar(
                    data_frame=param,
                    x="model_names",
                    y=kfold_columns[j],
                    hover_data=[kfold_columns[j], "model_names"],
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
                            label=target,
                            FILENAME=dire,
                            FILE_PATH=file_path,
                        )

        if t_split is True:
            if kf is True:
                raise Exception(
                    "set kf to True if you used KFold or set t_split to True"
                    "if you used the split method."
                )

            if target == "binary":
                IMAGE_COLUMNS = []
                t_split_columns = [
                    "Mean Absolute Error",
                    "Root Mean Squared Error",
                    "r2 score",
                    "Root Mean Squared Log Error",
                    "Median Absolute Error",
                    "Mean Absolute Percentage Error",
                    "Time Taken(s)",
                ]
                for i in range(len(t_split_columns)):
                    IMAGE_COLUMNS.append(t_split_columns[i] + ".png")

                if save is True:
                    dire = directory(save_name)
                for j in range(len(IMAGE_COLUMNS)):

                    fig = px.bar(
                        data_frame=param,
                        x="model_names",
                        y=t_split_columns[j],
                        hover_data=[t_split_columns[j], "model_names"],
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
                                label=target,
                                FILENAME=dire,
                                FILE_PATH=file_path,
                            )
