import time

from daal4py.sklearn.ensemble import RandomForestRegressor
from daal4py.sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from daal4py.sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from pandas import DataFrame
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, \
    AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import PoissonRegressor, GammaRegressor, HuberRegressor, RidgeCV, BayesianRidge, ElasticNetCV, \
    LassoCV, LassoLarsIC, LassoLarsCV, Lars, LarsCV, SGDRegressor, TweedieRegressor, RANSACRegressor, \
    OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, OrthogonalMatchingPursuit, LassoLars
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearnex.svm import SVR, NuSVR
from skopt.learning import ExtraTreesRegressor, GaussianProcessRegressor
from xgboost import XGBRegressor


class Regression:

    def __init__(self, lr=0, rfr=0, xgb=0, gbr=0, hgbr=0, svr=0, br=0, nsvr=0, etr=0, etrs=0, ada=0,
                 pr=0, lgbm=0, knr=0, dtr=0, mlp=0, hub=0, gmr=0, lsvr=0, ridg=0, rid=0, byr=0, ttr=0,
                 eltcv=0, elt=0, lcv=0, llic=0, llcv=0, l=0, lrcv=0, sgd=0, twr=0, glr=0, lass=0, ranr=0,
                 ompc=0, par=0, gpr=0, ompu=0, dr=0, lassla=0, krid=0):

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

    def regression_model_names(self):
        model_names = ["Linear Regression", "Random Forest Regressor", "XGBRegressor", "GradientBoostingRegressor",
                       "HistGradientBoostingRegressor", "SVR", "BaggingRegressor", "NuSVR", "ExtraTreeRegressor",
                       "ExtraTreesRegressor", "AdaBoostRegressor", "PoissonRegressor", "LGBMRegressor",
                       "KNeighborsRegressor", "DecisionTreeRegressor", "MLPRegressor", "HuberRegressor",
                       "GammaRegressor", "LinearSVR", "RidgeCV", "Ridge", "BayesianRidge",
                       "TransformedTargetRegressor", "ElasticNetCV", "ElasticNet"
                       ]

    def split(self,
              X: any,
              y: any,
              strat: bool = False,
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
        elif isinstance(strat, bool) is False:
            raise TypeError(
                "argument of type int or str is not valid. Parameters for strat is either False or True")

        elif sizeOfTest < 0 or sizeOfTest > 1:
            raise ValueError("value of sizeOfTest should be between 0 and 1")

        else:
            if strat is True:

                if shuffle_data is False:
                    raise TypeError("shuffle_data can only be False if strat is False")

                elif shuffle_data is True:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeOfTest,
                                                                        train_size=1 - sizeOfTest,
                                                                        stratify=y, random_state=randomState,
                                                                        shuffle=shuffle_data)

                    return X_train, X_test, y_train, y_test

            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeOfTest,
                                                                    train_size=1 - sizeOfTest)
                return X_train, X_test, y_train, y_test

    def initialize(self):
        """
        It initializes all the models that we will be using in our ensemble
        """
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
        self.glr = GeneralizedLinearRegressor()
        self.lass = Lasso()
        self.ranr = RANSACRegressor()
        self.ompc = OrthogonalMatchingPursuitCV()
        self.par = PassiveAggressiveRegressor()
        self.gpr = GaussianProcessRegressor()
        self.ompu = OrthogonalMatchingPursuit()
        self.dr = DummyRegressor()
        self.lassla = LassoLars()
        self.krid = KernelRidge()

        return (self.lr, self.rfr, self.xgb, self.gbr, self.hgbr, self.svr, self.br, self.nsvr, self.etr, self.etrs,
                self.ada, self.pr, self.lgbm, self.knr, self.dtr, self.mlp, self.hub, self.gmr, self.lsvr, self.ridg,
                self.rid, self.byr, self.ttr, self.eltcv, self.elt, self.lcv, self.llic, self.llcv, self.l, self.lrcv,
                self.sgd, self.twr, self.glr, self.lass, self.ranr, self.ompc, self.par, self.gpr, self.ompu, self.dr,
                self.lassla, self.krid)

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
            return_fastest_model: bool = False,
            target: str = 'binary'
            ) -> DataFrame:
        """
        If splitting is False, then do nothing. If splitting is True, then assign the values of split_data to the
        variables X_train, X_test, y_train, and y_test

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
            """
            if skf:
                raise TypeError("kf cannot be true if skf is true and skf cannot be true if kf is true. You can only "
                                "use one at the same time")
            """
            if split_data:
                raise ValueError("split_data cannot be used with kf, set splitting to True to use param "
                                 "split_data")

        if kf is True and (X is None or y is None or (X is None and y is None)):
            raise ValueError("Set the values of features X and target y")

        if target:
            accepted_targets = ['binary', 'multiclass']
            if target not in accepted_targets:
                raise Exception(f"target should be set to either binary or multiclass but target was set to {target}")

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
                    pass
                except MemoryError:
                    pass
                end = time.time()
                try:
                    pred = model[i].predict(X_te)
                except AttributeError:
                    pass
                true = y_te

                mae = mean_absolute_error(true, pred)
                mse = np.sqrt(mean_absolute_error(true, pred))
                r2 = r2_score(true, pred)
                try:
                    roc = roc_auc_score(true, pred)
                except ValueError:
                    roc = None
                try:
                    f1 = f1_score(true, pred)
                except ValueError:
                    f1 = None
                try:
                    pre = precision_score(true, pred)
                except ValueError:
                    pre = None
                try:
                    rec = recall_score(true, pred)
                except ValueError:
                    rec = None

                time_taken = round(end - start, 2)
                eval_bin = [acc, mae, mse, r2, roc, f1, pre, rec, time_taken]
                eval_mul = [acc, mae, mse, r2, time_taken]
                if target == 'binary':
                    dataframe.update({names[i]: eval_bin})
                elif target == 'multiclass':
                    dataframe.update({names[i]: eval_mul})

            if target == 'binary':
                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=self.t_split_binary_columns)
            elif target == 'multiclass':
                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=self.t_split_multiclass_columns)

            if return_best_model is not None:
                display(f'BEST MODEL BASED ON {return_best_model}')
                if return_best_model == 'accuracy':
                    display(df[df['accuracy'] == df['accuracy'].max()])
                elif return_best_model == 'mean absolute error':
                    display(df[df['mean absolute error'] == df['mean absolute error'].min()])
                elif return_best_model == 'mean squared error':
                    display(df[df['mean squared error'] == df['mean squared error'].min()])
                elif return_best_model == 'r2 score':
                    display(df[df['r2 score'] == df['r2 score'].max()])
                elif return_best_model == 'f1 score':
                    display(df[df['f1 score'] == df['f1 score'].max()])
                elif return_best_model == 'ROC AUC':
                    display(df[df['ROC AUC'] == df['ROC AUC'].max()])
            elif return_best_model is None:
                display(df.style.highlight_max(color="yellow"))

            if return_fastest_model is True:
                # df.drop(df[df['execution time(seconds)'] == 0.0].index, axis=0, inplace=True)
                display(f"FASTEST MODEL")
                display(df[df["execution time(seconds)"].max()])
            write_to_excel(excel, df)
            return df

        elif kf is True:

            # Fitting the models and predicting the values of the test set.
            KFoldModel = self.initialize()
            names = self.classifier_model_names()

            if target == 'binary':
                PrintLog("Training started")
                dataframe = self.startKFold(param=KFoldModel, param_X=X, param_y=y, param_cv=fold, fn_target=target)

                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=["Train Acc", "Test Acc",
                                                                                "Train Precision", "Test Precision",
                                                                                "Train Recall", "Test Recall",
                                                                                "Train f1", "Test f1", "Train r2",
                                                                                "Test r2",
                                                                                "Train std",
                                                                                "Test std", "Time Taken(s)"])
                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_

            elif target == 'multiclass':
                PrintLog("Training started")
                dataframe = self.startKFold(param=KFoldModel, param_X=X, param_y=y, param_cv=fold, fn_target=target)
                df = pd.DataFrame.from_dict(dataframe, orient='index', columns=["Train Precision Macro",
                                                                                "Test Precision Macro",
                                                                                "Train Recall Macro",
                                                                                "Test Recall Macro",
                                                                                "Train f1 Macro", "Test f1 Macro",
                                                                                "Time Taken(s)"])
                kf_ = kf_best_model(df, return_best_model, excel)
                return kf_
