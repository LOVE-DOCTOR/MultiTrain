import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas import DataFrame
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
    median_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from skopt.learning import ExtraTreesRegressor, GaussianProcessRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from MultiTrain.LOGGING import PrintLog
from MultiTrain.methods.multitrain_methods import write_to_excel, kf_best_model


class Regression:

    def __init__(self, lr=0, rfr=0, xgb=0, gbr=0, hgbr=0, svr=0, br=0, nsvr=0, etr=0, etrs=0, ada=0,
                 pr=0, lgbm=0, knr=0, dtr=0, mlp=0, hub=0, gmr=0, lsvr=0, ridg=0, rid=0, byr=0, ttr=0,
                 eltcv=0, elt=0, lcv=0, llic=0, llcv=0, l=0, lrcv=0, sgd=0, twr=0, glr=0, lass=0, ranr=0,
                 ompc=0, par=0, gpr=0, ompu=0, dr=0, lassla=0, krid=0, ard=0, quant=0, theil=0):

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
        self.quant = quant
        self.theil = theil

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
                       "KernelRidge", "ARDRegression", "QuantileRegressor", "TheilSenRegressor"]
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
        print("Regressors like HuberRegressor, QuantileRegressor, RANSACRegressor and TheilSen Regressor"
              " are robust to outliers")

        self.lr = LinearRegression()
        self.rfr = RandomForestRegressor()
        self.xgb = XGBRegressor()
        self.gbr = GradientBoostingRegressor()
        self.hgbr = HistGradientBoostingRegressor()
        self.svr = SVR()
        self.br = BaggingRegressor()
        self.nsvr = NuSVR()
        self.etr = ExtraTreeRegressor()
        self.etrs = ExtraTreesRegressor()
        self.ada = AdaBoostRegressor()
        self.pr = PoissonRegressor()
        self.lgbm = LGBMRegressor()
        self.knr = KNeighborsRegressor()
        self.dtr = DecisionTreeRegressor()
        self.mlp = MLPRegressor()
        self.hub = HuberRegressor()
        self.gmr = GammaRegressor()
        self.lsvr = LinearSVR()
        self.ridg = RidgeCV()
        self.rid = Ridge()
        self.byr = BayesianRidge()
        self.ttr = TransformedTargetRegressor()
        self.eltcv = ElasticNetCV()
        self.elt = ElasticNet()
        self.lcv = LassoCV()
        self.llic = LassoLarsIC()
        self.llcv = LassoLarsCV()
        self.l = Lars()
        self.lrcv = LarsCV()
        self.sgd = SGDRegressor()
        self.twr = TweedieRegressor()
        self.lass = Lasso()
        self.ranr = RANSACRegressor()
        self.ompc = OrthogonalMatchingPursuitCV()
        self.par = PassiveAggressiveRegressor()
        self.gpr = GaussianProcessRegressor()
        self.ompu = OrthogonalMatchingPursuit()
        self.dr = DummyRegressor()
        self.lassla = LassoLars()
        self.krid = KernelRidge()
        self.ard = ARDRegression()
        self.quant = QuantileRegressor()
        self.theil = TheilSenRegressor(n_jobs=-1)

        return self.lr, self.rfr, self.xgb, self.gbr, self.hgbr, self.svr, self.br, self.nsvr, self.etr, self.etrs, \
               self.ada, self.pr, self.lgbm, self.knr, self.dtr, self.mlp, self.hub, self.gmr, self.lsvr, self.ridg, \
               self.rid, self.byr, self.ttr, self.eltcv, self.elt, self.lcv, self.llic, self.llcv, self.l, self.lrcv, \
               self.sgd, self.twr, self.lass, self.ranr, self.ompc, self.par, self.gpr, self.ompu, self.dr, \
               self.lassla, self.krid, self.ard, self.quant, self.theil

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
                print(model[i])
                # X_tr is X_train, X_te is X_test, y_tr is y_train, y_te is y_test
                true = y_te
                mae = mean_absolute_error(true, pred)
                rmse = np.sqrt(mean_squared_error(true, pred))
                r2 = r2_score(true, pred)
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
                                 "Mean Absolute Percentage Error"]
            df = pd.DataFrame.from_dict(dataframe, orient='index', columns=dataframe_columns)

            if return_best_model is not None:
                display(f'BEST MODEL BASED ON {return_best_model}')

                if return_best_model == 'Mean Absolute Error':
                    display(df[df['Mean Absolute Error'] == df['Mean Absolute Error'].min()])
                elif return_best_model == 'RMSE':
                    display(df[df['RMSE'] == df['RMSE'].min()])
                elif return_best_model == 'r2 score':
                    display(df[df['r2 score'] == df['r2 score'].max()])

            elif return_best_model is None:
                display(df.style.highlight_max(color="yellow"))

            write_to_excel(excel, df)
            return df

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
