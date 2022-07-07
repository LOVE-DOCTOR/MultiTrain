import time
from operator import __setitem__

import numpy as np
import pandas as pd
import plotly.express as px
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from numpy import reshape
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, \
    AdaBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import PoissonRegressor, GammaRegressor, HuberRegressor, RidgeCV, BayesianRidge, ElasticNetCV, \
    LassoCV, LassoLarsIC, LassoLarsCV, Lars, LarsCV, SGDRegressor, TweedieRegressor, RANSACRegressor, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, LassoLars, ARDRegression, \
    QuantileRegressor, TheilSenRegressor, Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, \
    median_absolute_error, mean_absolute_percentage_error, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, HalvingRandomSearchCV, HalvingGridSearchCV, \
    RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from skopt import BayesSearchCV
from skopt.learning import ExtraTreesRegressor, GaussianProcessRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from MultiTrain.LOGGING import PrintLog
from MultiTrain.methods.multitrain_methods import write_to_excel, kf_best_model, t_best_model, img, directory, \
    img_plotly


class Regression:

    def __init__(self, lr=0, rfr=0, xgb=0, gbr=0, hgbr=0, svr=0, br=0, nsvr=0, etr=0, etrs=0, ada=0,
                 pr=0, lgbm=0, knr=0, dtr=0, mlp=0, hub=0, gmr=0, lsvr=0, ridg=0, rid=0, byr=0, ttr=0,
                 eltcv=0, elt=0, lcv=0, llic=0, llcv=0, l=0, lrcv=0, sgd=0, twr=0, glr=0, lass=0, ranr=0,
                 ompc=0, par=0, gpr=0, ompu=0, dr=0, lassla=0, krid=0, ard=0, theil=0, random_state=None):

        self.lr = lr
        self.rfr = rfr
        self.xgb = xgb
        self.gbr = gbr
        self.hgbr = hgbr
        self.svr = svr
        self.br = br
        self.nsvr = nsvr
        self.etr = etr
        self.etrs = etrs
        self.ada = ada
        self.pr = pr
        self.lgbm = lgbm
        self.knr = knr
        self.dtr = dtr
        self.mlp = mlp
        self.hub = hub
        self.gmr = gmr
        self.lsvr = lsvr
        self.ridg = ridg
        self.rid = rid
        self.byr = byr
        self.ttr = ttr
        self.eltcv = eltcv
        self.elt = elt
        self.lcv = lcv
        self.llic = llic
        self.llcv = llcv
        self.l = l
        self.lrcv = lrcv
        self.sgd = sgd
        self.twr = twr
        self.glr = glr
        self.lass = lass
        self.ranr = ranr
        self.ompc = ompc
        self.par = par
        self.gpr = gpr
        self.ompu = ompu
        self.dr = dr
        self.lassla = lassla
        self.krid = krid
        self.ard = ard
        # self.quant = quant
        self.theil = theil
        self.random_state = random_state

    def regression_model_names(self):
        model_names = ["Linear Regression", "Random Forest Regressor", "XGBRegressor", "GradientBoostingRegressor",
                       "HistGradientBoostingRegressor", "SVR", "BaggingRegressor", "NuSVR", "ExtraTreeRegressor",
                       "ExtraTreesRegressor", "AdaBoostRegressor", "PoissonRegressor", "LGBMRegressor",
                       "KNeighborsRegressor", "DecisionTreeRegressor", "MLPRegressor", "HuberRegressor",
                       "GammaRegressor", "LinearSVR", "RidgeCV", "Ridge", "BayesianRidge",
                       "TransformedTargetRegressor", "ElasticNetCV", "ElasticNet", "LassoCV", "LassoLarsIC",
                       "LassoLarsCV", "Lars", "LarsCV", "SGDRegressor", "TweedieRegressor", "Lasso",
                       "RANSACRegressor", "OrthogonalMatchingPursuitCV", "PassiveAggressiveRegressor",
                       "GaussianProcessRegressor", "OrthogonalMatchingPursuit", "DummyRegressor", "LassoLars",
                       "KernelRidge", "ARDRegression", "TheilSenRegressor"]
        return model_names

    def split(self,
              X: any,
              y: any,
              sizeOfTest: float = 0.2,
              randomState: int = None,
              shuffle_data: bool = True):
        """

        :param X: features
        :param y: labels
        :param strat: used to initialize stratify = y in train_test_split if True
        :param sizeOfTest: define size of test data
        :param randomState: define random state
        :param shuffle_data: If set to True, it sets shuffle to True in train_test_split

        Example
        df = pd.read_csv("nameOfFile.csv")
        X = df.drop("nameOfLabelColumn", axis=1)
        y = df["nameOfLabelColumn")
        split(X = features, y = labels, sizeOfTest=0.3, randomState=42, strat=True, shuffle_data=True)
        """
        if isinstance(X, int or bool) or isinstance(y, int or bool):
            raise ValueError(f"{X} and {y} are not valid arguments for 'split'."
                             f"Try using the standard variable names e.g split(X, y) instead of split({X}, {y})")

        elif sizeOfTest < 0 or sizeOfTest > 1:
            raise ValueError("value of sizeOfTest should be between 0 and 1")

        else:

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeOfTest,
                                                                train_size=1 - sizeOfTest, random_state=randomState,
                                                                shuffle=shuffle_data)
            return X_train, X_test, y_train, y_test

    def initialize(self):
        """
        It initializes all the models that we will be using in our ensemble
        """

        self.lr = LinearRegression(n_jobs=-1)
        self.rfr = RandomForestRegressor(random_state=self.random_state)
        self.xgb = XGBRegressor(random_state=self.random_state)
        self.gbr = GradientBoostingRegressor(random_state=self.random_state)
        self.hgbr = HistGradientBoostingRegressor(random_state=self.random_state)
        self.svr = SVR()
        self.br = BaggingRegressor(random_state=self.random_state)
        self.nsvr = NuSVR()
        self.etr = ExtraTreeRegressor(random_state=self.random_state)
        self.etrs = ExtraTreesRegressor(random_state=self.random_state)
        self.ada = AdaBoostRegressor(random_state=self.random_state)
        self.pr = PoissonRegressor()
        self.lgbm = LGBMRegressor(random_state=self.random_state)
        self.knr = KNeighborsRegressor()
        self.dtr = DecisionTreeRegressor(random_state=self.random_state)
        self.mlp = MLPRegressor(random_state=self.random_state)
        self.hub = HuberRegressor()
        self.gmr = GammaRegressor()
        self.lsvr = LinearSVR(random_state=self.random_state)
        self.ridg = RidgeCV()
        self.rid = Ridge(random_state=self.random_state)
        self.byr = BayesianRidge()
        self.ttr = TransformedTargetRegressor()
        self.eltcv = ElasticNetCV(n_jobs=-1, random_state=self.random_state)
        self.elt = ElasticNet(random_state=self.random_state)
        self.lcv = LassoCV(n_jobs=-1, random_state=self.random_state)
        self.llic = LassoLarsIC()
        self.llcv = LassoLarsCV()
        self.l = Lars(random_state=self.random_state)
        self.lrcv = LarsCV(n_jobs=-1)
        self.sgd = SGDRegressor(random_state=self.random_state)
        self.twr = TweedieRegressor()
        self.lass = Lasso(random_state=self.random_state)
        self.ranr = RANSACRegressor(random_state=self.random_state)
        self.ompc = OrthogonalMatchingPursuitCV(n_jobs=-1)
        self.par = PassiveAggressiveRegressor(random_state=self.random_state)
        self.gpr = GaussianProcessRegressor(random_state=self.random_state)
        self.ompu = OrthogonalMatchingPursuit()
        self.dr = DummyRegressor()
        self.lassla = LassoLars(random_state=self.random_state)
        self.krid = KernelRidge()
        self.ard = ARDRegression()
        # self.quant = QuantileRegressor()
        self.theil = TheilSenRegressor(n_jobs=-1, random_state=self.random_state)

        return self.lr, self.rfr, self.xgb, self.gbr, self.hgbr, self.svr, self.br, self.nsvr, self.etr, self.etrs, \
               self.ada, self.pr, self.lgbm, self.knr, self.dtr, self.mlp, self.hub, self.gmr, self.lsvr, self.ridg, \
               self.rid, self.byr, self.ttr, self.eltcv, self.elt, self.lcv, self.llic, self.llcv, self.l, self.lrcv, \
               self.sgd, self.twr, self.lass, self.ranr, self.ompc, self.par, self.gpr, self.ompu, self.dr, \
               self.lassla, self.krid, self.ard, self.theil

    def _get_index(self, df, the_best):
        name = list(self.regression_model_names())
        MODEL = self.initialize()
        df['model_names'] = name

        high = ["Neg Mean Absolute Error", "Neg Root Mean Squared Error", "r2 score",
                "Neg Root Mean Squared Log Error", "Neg Median Absolute Error",
                "Neg Median Absolute Percentage Error"]

        low = ["Mean Absolute Error", "Root Mean Squared Error",
               "Root Mean Squared Log Error", "Median Absolute Error",
               "Mean Absolute Percentage Error"]

        if the_best in high:
            best_model_details = df[df[the_best] == df[the_best].max()]

        elif the_best in low:
            best_model_details = df[df[the_best] == df[the_best].min()]

        else:
            raise Exception(f'metric {the_best} not found')

        best_model_details = best_model_details.reset_index()
        best_model_name = best_model_details.iloc[0]['model_names']
        index_ = name.index(best_model_name)
        return MODEL[index_]

    def startKFold(self, param, param_X, param_y, param_cv, train_score):
        names = self.regression_model_names()

        dataframe = {}
        for i in range(len(param)):
            start = time.time()
            score = ('neg_mean_absolute_error',
                     'neg_root_mean_squared_error',
                     'neg_mean_squared_error',
                     'r2',
                     'neg_median_absolute_error',
                     'neg_mean_squared_log_error',
                     'neg_mean_absolute_percentage_error')

            scores = cross_validate(estimator=param[i], X=param_X, y=param_y, scoring=score,
                                    cv=param_cv, n_jobs=-1, return_train_score=True)
            end = time.time()
            seconds = end - start

            if train_score is True:
                mean_train_mae = scores['train_neg_mean_absolute_error'].mean()
                mean_test_mae = scores['test_neg_mean_absolute_error'].mean()
                mean_train_rmse = scores['train_neg_root_mean_squared_error'].mean()
                mean_test_rmse = scores['test_neg_root_mean_squared_error'].mean()
                mean_train_r2 = scores['train_r2'].mean()
                mean_test_r2 = scores['test_r2'].mean()
                mean_train_rmsle = np.sqrt(scores['train_neg_mean_squared_log_error'].mean())
                mean_test_rmsle = np.sqrt(scores['test_neg_mean_squared_log_error'].mean())
                mean_train_meae = scores['train_neg_median_absolute_error'].mean()
                mean_test_meae = scores['test_neg_median_absolute_error'].mean()
                mean_train_mape = scores['train_neg_mean_absolute_percentage_error'].mean()
                mean_test_mape = scores['test_neg_mean_absolute_percentage_error'].mean()

                # scores = scores.tolist()
                scores_df = [mean_train_mae, mean_test_mae, mean_train_rmse, mean_test_rmse,
                             mean_train_r2, mean_test_r2, mean_train_rmsle, mean_test_rmsle,
                             mean_train_meae, mean_test_meae, mean_train_mape, mean_test_mape,
                             seconds]
                dataframe.update({names[i]: scores_df})

            elif train_score is False:
                mean_test_mae = scores['test_neg_mean_absolute_error'].mean()
                mean_test_rmse = scores['test_neg_root_mean_squared_error'].mean()
                mean_test_r2 = scores['test_r2'].mean()
                mean_test_rmsle = np.sqrt(scores['test_neg_mean_squared_log_error'].mean())
                mean_test_meae = scores['test_neg_median_absolute_error'].mean()
                mean_test_mape = scores['test_neg_mean_absolute_percentage_error'].mean()

                scores_df = [mean_test_mae, mean_test_rmse, mean_test_r2, mean_test_rmsle,
                             mean_test_meae, mean_test_mape]
                dataframe.update({names[i]: scores_df})

            return dataframe

    def fit(self,
            X: str = None,
            y: str = None,
            split_self: bool = False,
            X_train: str = None,
            X_test: str = None,
            y_train: str = None,
            y_test: str = None,
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
        :param return_fastest_model: defaults to False, set to True when you want the method to only return a dataframe
        of the fastest model

        :param return_best_model: defaults to False, set to True when you want the method to only return a dataframe of
        the best model

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

        if isinstance(splitting, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in splitting. Try splitting = False or splitting = True "
                f"instead of splitting = {splitting}")

        if isinstance(kf, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in kf. Try kf = False or kf = True "
                f"instead of kf = {kf}")

        if isinstance(fold, int) is False:
            raise TypeError(
                "param fold is of type int, pass a integer to fold e.g fold = 5, where 5 is number of "
                "splits you want to use for the cross validation procedure")

        if kf:
            if split_self is True:
                raise Exception(
                    "split_self should only be set to True when you split with train_test_split from "
                    "sklearn.model_selection")

            if splitting:
                raise ValueError("KFold cross validation cannot be true if splitting is true and splitting cannot be "
                                 "true if KFold is true")

            if split_data:
                raise ValueError("split_data cannot be used with kf, set splitting to True to use param "
                                 "split_data")

        if kf is True and (X is None or y is None or (X is None and y is None)):
            raise ValueError("Set the values of features X and target y")

        if splitting is True or split_self is True:
            if splitting and split_data:
                X_tr, X_te, y_tr, y_te = split_data[0], split_data[1], split_data[2], split_data[3]
            elif X_train is not None \
                    and X_test is not None \
                    and y_train is not None \
                    and y_test is not None:
                X_tr, X_te, y_tr, y_te = X_train, X_test, y_train, y_test
            model = self.initialize()
            names = self.regression_model_names()
            dataframe = {}
            for i in range(len(model)):
                start = time.time()
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
                r2 = r2_score(true, pred, force_finite=True)
                try:
                    rmsle = np.sqrt(mean_squared_log_error(true, pred))
                except ValueError:
                    rmsle = 'NaN'
                meae = median_absolute_error(true, pred)
                mape = mean_absolute_percentage_error(true, pred)

                time_taken = round(end - start, 2)
                eval_metrics = [mae, rmse, r2, rmsle, meae, mape, time_taken]
                dataframe.update({names[i]: eval_metrics})

            dataframe_columns = ["Mean Absolute Error", "Root Mean Squared Error", "r2 score",
                                 "Root Mean Squared Log Error", "Median Absolute Error",
                                 "Mean Absolute Percentage Error", "Time Taken(s)"]
            df = pd.DataFrame.from_dict(dataframe, orient='index', columns=dataframe_columns)

            t_split = t_best_model(df, return_best_model, excel)
            return t_split

        elif kf is True:

            # Fitting the models and predicting the values of the test set.
            KFoldModel = self.initialize()
            names = self.regression_model_names()

            PrintLog("Training started")
            dataframe = self.startKFold(param=KFoldModel, param_X=X, param_y=y, param_cv=fold,
                                        train_score=show_train_score)

            if show_train_score is True:
                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=["Neg Mean Absolute Error(Train)",
                                                                                "Neg Mean Absolute Error",
                                                                                "Neg Root Mean Squared Error(Train)",
                                                                                "Neg Root Mean Squared Error",
                                                                                "r2(Train)", "r2",
                                                                                "Neg Root Mean Squared Log Error(Train)",
                                                                                "Neg Root Mean Squared Log Error",
                                                                                "Neg Median Absolute Error(Train)",
                                                                                "Neg Median Absolute Error",
                                                                                "Neg Mean Absolute Percentage Error"
                                                                                "(Train)",
                                                                                "Neg Mean Absolute Percentage Error",
                                                                                "Time Taken(s)"])

                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_

            if show_train_score is False:
                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=["Neg Mean Absolute Error",
                                                                                "Neg Root Mean Squared Error",
                                                                                "r2",
                                                                                "Neg Root Mean Squared Log Error",
                                                                                "Neg Median Absolute Error",
                                                                                "Neg Mean Absolute Percentage Error",
                                                                                "Time Taken(s)"])
                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_

    def use_model(self, df, model: str = None, best: str = None):
        """


        :param df: the dataframe object
        :param model: name of the classifier algorithm
        :param best: the evaluation metric used to find the best model

        :return:
        """

        name = self.regression_model_names()
        MODEL = self.initialize()

        if model is not None and best is not None:
            raise Exception('You can only use one of the two arguments.')

        if model:
            if model not in name:
                raise Exception(f"name {model} is not found, "
                                f"here is a list of the available models to work with: {name}")
            elif model in name:
                index_ = name.index(model)
                return MODEL[index_]

        elif best:
            instance = self._get_index(df, best)
            return instance

    def tune_parameters(self,
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
                        verbose: int = 4,
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
                        score='accuracy'
                        ):
        """
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
        name = self.regression_model_names()
        MODEL = self.initialize()
        # index_ = name.index(model)
        # mod = MODEL[index_]

        if isinstance(parameters, dict) is False:
            raise TypeError("The 'parameters' argument only accepts a dictionary of the parameters for the "
                            "model you want to train with.")
        if tune:
            scorers = {
                'precision_score': make_scorer(precision_score),
                'recall_score': make_scorer(recall_score),
                'accuracy_score': make_scorer(accuracy_score)
            }

            if tune == 'grid':
                tuned_model = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=use_cpu, cv=cv,
                                           verbose=verbose, error_score=error_score, pre_dispatch=pre_dispatch,
                                           return_train_score=return_train_score, scoring=scorers, refit=refit)
                return tuned_model

            elif tune == 'random':
                tuned_model = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_jobs=use_cpu, cv=cv,
                                                 verbose=verbose, random_state=random_state, n_iter=n_iter,
                                                 return_train_score=return_train_score, error_score=error_score,
                                                 scoring=scorers, refit=refit, pre_dispatch=pre_dispatch)
                return tuned_model

            elif tune == 'bayes':
                tuned_model = BayesSearchCV(estimator=model, search_spaces=parameters, n_jobs=use_cpu,
                                            return_train_score=return_train_score, cv=cv, verbose=verbose,
                                            refit=refit, random_state=random_state, scoring=scorers,
                                            error_score=error_score, optimizer_kwargs=optimizer_kwargs,
                                            n_points=n_points, n_iter=n_iter, fit_params=fit_params,
                                            pre_dispatch=pre_dispatch)
                return tuned_model

            elif tune == 'half-grid':
                tuned_model = HalvingGridSearchCV(estimator=model, param_grid=parameters, n_jobs=use_cpu, cv=cv,
                                                  verbose=4, random_state=42, factor=factor, refit=refit,
                                                  scoring=score, resource=resource, min_resources=min_resources_grid,
                                                  max_resources=max_resources, error_score=error_score,
                                                  aggressive_elimination=aggressive_elimination)

                return tuned_model

            elif tune == 'half-random':
                tuned_model = HalvingRandomSearchCV(estimator=model, param_distributions=parameters, n_jobs=use_cpu,
                                                    cv=cv, verbose=4, random_state=42, factor=factor, refit=refit,
                                                    scoring=score, resource=resource, error_score=error_score,
                                                    min_resources=min_resources_rand, max_resources=max_resources,
                                                    aggressive_elimination=aggressive_elimination)
                return tuned_model

    def visualize(self,
                  param: {__setitem__},
                  file_path: any = None,
                  kf: bool = False,
                  t_split: bool = False,
                  size=(15, 8),
                  save: str = None,
                  save_name='dir1',
                  ):

        """
        The function takes in a dictionary of the model names and their scores, and plots them in a bar chart

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

        names = self.regression_model_names()
        sns.set()

        param['model_names'] = names
        FILE_FORMATS = ['pdf', 'png']
        if save not in FILE_FORMATS:
            raise Exception("set save to either 'pdf' or 'png' ")

        if save in FILE_FORMATS:
            if isinstance(save_name, str) is False:
                raise ValueError('You can only set a string to save_name')

            if save_name is None:
                raise Exception('Please set a value to save_name')

        if file_path:
            if save is None:
                raise Exception("set save to either 'pdf' or 'png' before defining a file path")

        if save is None:
            if save_name:
                raise Exception('You can only use save_name after param save is defined')

        if kf is True and t_split is True:
            raise Exception("set kf to True if you used KFold or set t_split to True"
                            "if you used the split method.")
        if kf is True:
            plt.figure(figsize=size)
            plot = sns.barplot(x="model_names", y="Neg Mean Absolute Error", data=param)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
            plt.title("Neg Mean Absolute Error")

            plt.figure(figsize=size)
            plot1 = sns.barplot(x="model_names", y="Neg Root Mean Squared Error", data=param)
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Root Mean Squared Error")

            plt.figure(figsize=size)
            plot2 = sns.barplot(x="model_names", y="Neg Root Mean Squared Log Error", data=param)
            plot2.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Root Mean Squared Log Error")

            plt.figure(figsize=size)
            plot3 = sns.barplot(x="model_names", y="Neg Median Absolute Error", data=param)
            plot3.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Neg Median Absolute Error")

            plt.figure(figsize=size)
            plot4 = sns.barplot(x="model_names", y="r2", data=param)
            plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
            plt.title("R2 SCORE")

            plt.figure(figsize=size)
            plot5 = sns.barplot(x="model_names", y="Neg Mean Absolute Percentage Error", data=param)
            plot5.set_xticklabels(plot5.get_xticklabels(), rotation=90)
            plt.title("Neg Mean Absolute Percentage Error")

            plt.figure(figsize=size)
            plot6 = sns.barplot(x="model_names", y="Time Taken(s)", data=param)
            plot6.set_xticklabels(plot6.get_xticklabels(), rotation=90)
            plt.title("Time Taken(s)")

            if save == 'pdf':
                name = save_name + ".pdf"
                img(name, FILE_PATH=file_path, type_='file')

            elif save == 'png':
                name = save_name
                img(FILENAME=name, FILE_PATH=file_path, type_='picture')

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
            plot1 = sns.barplot(x="model_names", y="Root Mean Squared Error", data=param)
            plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Root Mean Squared Error")

            plt.figure(figsize=size)
            plot2 = sns.barplot(x="model_names", y="Root Mean Squared Log Error", data=param)
            plot2.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Root Mean Squared Log Error")

            plt.figure(figsize=size)
            plot3 = sns.barplot(x="model_names", y="Neg Median Absolute Error", data=param)
            plot3.set_xticklabels(plot1.get_xticklabels(), rotation=90)
            plt.title("Median Absolute Error")

            plt.figure(figsize=size)
            plot4 = sns.barplot(x="model_names", y="r2", data=param)
            plot4.set_xticklabels(plot4.get_xticklabels(), rotation=90)
            plt.title("R2 SCORE")

            plt.figure(figsize=size)
            plot5 = sns.barplot(x="model_names", y="Mean Absolute Percentage Error", data=param)
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

            if save == 'pdf':
                name = save_name + ".pdf"
                img(name, FILE_PATH=file_path, type_='file')
            elif save == 'png':
                name = save_name
                img(FILENAME=name, FILE_PATH=file_path, type_='picture')

    def show(self,
             param: {__setitem__},
             file_path: any = None,
             kf: bool = False,
             t_split: bool = False,
             save: bool = False,
             save_name=None,
             target='binary',
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

        names = self.regression_model_names()

        param['model_names'] = names

        if kf is True:
            if t_split is True:
                raise Exception("set kf to True if you used KFold or set t_split to True"
                                "if you used the split method.")

            IMAGE_COLUMNS = []
            kfold_columns = ["Neg Mean Absolute Error", "Neg Root Mean Squared Error", "r2 score",
                             "Neg Root Mean Squared Log Error", "Neg Median Absolute Error",
                             "Neg Mean Absolute Percentage Error", "Time Taken(s)"]
            for i in range(len(kfold_columns)):
                IMAGE_COLUMNS.append(kfold_columns[i] + ".png")

            if save is True:
                dire = directory(save_name)
            for j in range(len(IMAGE_COLUMNS)):

                fig = px.bar(data_frame=param,
                             x="model_names",
                             y=kfold_columns[j],
                             hover_data=[kfold_columns[j], "model_names"],
                             color="Time Taken(s)")
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
                raise Exception("set kf to True if you used KFold or set t_split to True"
                                "if you used the split method.")

            if target == 'binary':
                IMAGE_COLUMNS = []
                t_split_columns = ["Mean Absolute Error", "Root Mean Squared Error", "r2 score",
                                   "Root Mean Squared Log Error", "Median Absolute Error",
                                   "Mean Absolute Percentage Error", "Time Taken(s)"]
                for i in range(len(t_split_columns)):
                    IMAGE_COLUMNS.append(t_split_columns[i] + ".png")

                if save is True:
                    dire = directory(save_name)
                for j in range(len(IMAGE_COLUMNS)):

                    fig = px.bar(data_frame=param,
                                 x="model_names",
                                 y=t_split_columns[j],
                                 hover_data=[t_split_columns[j], "model_names"],
                                 color="execution time(seconds)")
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

