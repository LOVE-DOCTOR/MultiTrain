# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42) In your
# code, you can adjust the names of X_train, X_test, y_train and y_test if you named them differently when splitting
# (line 58 - 60)
from numpy import mean
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearnex import patch_sklearn
from sklearnex.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearnex.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearnex.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from LOGGING.log_message import PrintLog
import warnings
import time

warnings.filterwarnings("ignore")

patch_sklearn()


def split(X: any, y: any, strat: bool = False, sizeOfTest: float = 0.2, randomState: int = None,
          shuffle_data: bool = True):
    if isinstance(X, int) or isinstance(y, int):
        raise ValueError(f"{X} and {y} are not valid arguments for 'split'."
                         f"Try using the standard variable names e.g split(X, y) instead of split({X}, {y})")
    elif isinstance(strat, bool) is False:
        raise TypeError("argument of type int or str is not valid. Parameters for strat is either False or True")

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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeOfTest, train_size=1 - sizeOfTest)
            return X_train, X_test, y_train, y_test


def classifier_model_names():
    model_names = ["Logistic Regression", "SGDClassifier", "PassiveAggressiveClassifier", "RandomForestClassifier",
                   "GradientBoostingClassifier", "HistGradientBoostingClassifier", "AdaBoostClassifier",
                   "CatBoostClassifier", "XGBClassifier", "GaussianNB", "LinearDiscriminantAnalysis",
                   "KNeighborsClassifier", "MLPClassifier", "SVC", "DecisionTreeClassifier", "BernoulliNB",
                   "MultinomialNB", "CategoricalNB", "ComplementNB", "ExtraTreesClassifier", "RidgeClassifier",
                   "ExtraTreeClassifier", "LinearSVC", "BaggingClassifier", "GaussianProcessClassifier",
                   "QuadraticDiscriminantAnalysis"]
    return model_names


class Models:

    def __init__(self, lr=0, sgdc=0, pagg=0, rfc=0, gbc=0, cat=0, xgb=0, gnb=0, lda=0, knc=0, mlp=0, svc=0,
                 dtc=0, bnb=0, mnb=0, cnb=0, conb=0, hgbc=0, abc=0, etcs=0, rcl=0, etc=0, lsvc=0, bc=0, gpc=0,
                 qda=0) -> None:
        """

        :param lr: Logistic Regression
        :param sgdc: Stochastic Gradient Descent Classifier
        :param pagg: Passive Aggressive Classifier
        :param rfc: Random Forest Classifier
        :param gbc: Gradient Boosting Classifier
        :param cat: CatBoostClassifier
        :param xgb: XGBoost Classifier
        :param gnb: GaussianNB Classifier
        :param lda: Linear Discriminant Analysis
        :param knc: K Neighbors Classifier
        :param mlp: MLP Classifier
        :param svc: Support Vector Classifier
        :param dtc: Decision Tree Classifier
        :param cnb: CategoricalNB
        :param conb: ComplementNB
        :param hgbc: Hist Gradient Boosting Classifier
        :param abc: Ada Boost Classifier
        :param etcs: Extra Trees Classifier
        :param rcl: Ridge Classifier
        :param etc: Extra TreesC lassifier
        :param lsvc: Linear Support Vector Classifier
        :param bc: Bagging Classifier
        :param gpc: Gaussian Process Classifier
        :param qda: Quadratic Discriminant Analysis
        """

        self.lr = lr
        self.sgdc = sgdc
        self.pagg = pagg
        self.rfc = rfc
        self.gbc = gbc
        self.cat = cat
        self.xgb = xgb
        self.gnb = gnb
        self.lda = lda
        self.knc = knc
        self.mlp = mlp
        self.svc = svc
        self.dtc = dtc
        self.bnb = bnb
        self.mnb = mnb
        self.cnb = cnb
        self.conb = conb
        self.hgbc = hgbc
        self.abc = abc
        self.etcs = etcs
        self.rcl = rcl
        self.etc = etc
        self.lsvc = lsvc
        self.bc = bc
        self.gpc = gpc
        self.qda = qda

    def initialize(self):
        """
        It initializes all the models that we will be using in our ensemble
        """
        self.lr = LogisticRegression(random_state=42, warm_start=True, max_iter=400)
        self.sgdc = SGDClassifier(random_state=42, early_stopping=True, validation_fraction=0.2,
                                  shuffle=True, n_iter_no_change=20)
        self.pagg = PassiveAggressiveClassifier(shuffle=True, fit_intercept=True, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=20, n_jobs=-1)
        self.rfc = RandomForestClassifier(random_state=42, warm_start=True)
        self.gbc = GradientBoostingClassifier(random_state=42, learning_rate=0.01, validation_fraction=0.1,
                                              n_iter_no_change=20)
        self.hgbc = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.2, random_state=42,
                                                   max_iter=300)
        self.abc = AdaBoostClassifier(random_state=42)
        self.cat = CatBoostClassifier(random_state=42, learning_rate=0.01)
        self.xgb = XGBClassifier(use_label_encoder=False)
        self.gnb = GaussianNB()
        self.lda = LinearDiscriminantAnalysis()
        self.knc = KNeighborsClassifier(n_jobs=-1)
        self.mlp = MLPClassifier(batch_size=10, shuffle=True, random_state=42, warm_start=True,
                                 early_stopping=True, n_iter_no_change=20)
        self.svc = SVC(random_state=42)
        self.dtc = DecisionTreeClassifier(random_state=42)
        self.bnb = BernoulliNB()
        self.mnb = MultinomialNB()
        self.cnb = CategoricalNB()
        self.conb = ComplementNB()
        self.etcs = ExtraTreesClassifier(warm_start=True, random_state=42, n_jobs=-1)
        self.rcl = RidgeClassifier(random_state=42, max_iter=300)
        self.etc = ExtraTreeClassifier(random_state=42)
        self.lsvc = LinearSVC(random_state=42, max_iter=300)
        self.bc = BaggingClassifier(warm_start=True, n_jobs=-1, random_state=42)
        self.gpc = GaussianProcessClassifier(warm_start=True, random_state=42, n_jobs=-1)
        self.qda = QuadraticDiscriminantAnalysis()
        return (self.lr, self.sgdc, self.pagg, self.rfc, self.gbc, self.hgbc, self.abc, self.cat, self.xgb, self.gnb,
                self.lda, self.knc, self.mlp, self.svc, self.dtc, self.bnb, self.mnb, self.cnb, self.conb, self.etcs,
                self.rcl, self.etc, self.lsvc, self.bc, self.gpc, self.qda)

    def fit_eval_models(self, X=None, y=None, X_train=0, X_test=0, y_train=0, y_test=0,
                        split_data: str = None, splitting: bool = False, kf: bool = False, fold: tuple = (10, 1, True)):
        """
        If splitting is False, then do nothing. If splitting is True, then assign the values of split_data to the
        variables X_train, X_test, y_train, and y_test

        :param y:
        :param X:
        :type fold: object
        :param fold:
        :param kf:
        :param X_train: The training data
        :param X_test: The test data
        :param y_train: The training set labels
        :param y_test: The test set labels
        :param split_data: str = None, splitting: bool = False
        :type split_data: str
        :param splitting: bool = False, defaults to False
        :type splitting: bool (optional)
        """

        if isinstance(splitting, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in splitting. Try splitting = False or splitting = True "
                f"instead of splitting = {splitting}")

        if isinstance(kf, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in kf. Try kf = False or kf = True "
                f"instead of kf = {kf}")

        if isinstance(fold, tuple) is False:
            raise TypeError(
                "param fold is of type tuple, pass a tuple to fold e.g fold = (10, 1, True), where 10 is number of "
                "splits you want to use for the cross validation procedure, where 1 is the random_state, where True "
                "is to allow shuffling.")

        if len(fold) != 3:
            raise ValueError(
                "all 3 values of fold have to be fulfilled e.g fold = (10, 1, True), where 10 is number of "
                "splits you want to use for the cross validation procedure, where 1 is the random_state, where True "
                "is to allow shuffling."
            )

        if isinstance(fold[2], bool) is False:
            raise TypeError("all 3 values of fold have to be fulfilled e.g fold = (10, 1, True), where 10 is number of "
                            "splits you want to use for the cross validation procedure, where 1 is the random_state, "
                            "where True is to allow shuffling. The third value of the list fold has to be a boolean "
                            "value, e.g True or False.")

        if kf:
            if splitting:
                raise ValueError("KFold cross validation cannot be true, if splitting is true and splitting cannot be "
                                 "true if KFold is true")

            elif split_data:
                raise ValueError("split_data cannot be used with kf, set splitting to True to use param "
                                 "split_data")

        if kf is True and len(fold) == 3 and (X is None or y is None or (X is None and y is None)):
            raise ValueError("Set the values of features X and target y")

        if splitting and split_data:
            X_train, X_test, y_train, y_test = split_data[0], split_data[1], split_data[2], split_data[3]
            start = time.time()
            model = self.initialize()
            names = classifier_model_names()
            for i in range(len(model)):
                model[i].fit(X_train, y_train)
                pred = model[i].predict(X_test)
                true = y_test

                acc = accuracy_score(true, pred)
                mae = mean_absolute_error(true, pred)
                mse = mean_squared_error(true, pred)
                clr = classification_report(true, pred)
                cfm = confusion_matrix(true, pred)
                r2 = r2_score(true, pred)

                print("The model used is ", names[i])
                print("The Accuracy of the Model is ", acc)
                print("The r2 score of the Model is ", r2)
                print("The Mean Absolute Error of the Model is", mae)
                print("The Mean Squared Error of the Model is", mse)
                print("\n")
                print("The Classification Report of the Model is")
                print(clr)
                print("\n")
                print("The Confusion Matrix of the Model is")
                print(cfm)
                print("\n")
            end = time.time()
            minutes = (end - start) / 60
            PrintLog(f"completed in {minutes} minutes")

        elif len(fold) == 3 and kf is True:
            start = time.time()
            cv = KFold(n_splits=fold[0], random_state=fold[1], shuffle=fold[2])

            # Fitting the models and predicting the values of the test set.
            KFoldModel = self.initialize()
            names = classifier_model_names()
            for i in range(len(KFoldModel)):
                scores = cross_val_score(KFoldModel[i], X, y, scoring='accuracy', cv=cv, n_jobs=-1)
                mean_ = mean(scores)
                print(f"{names[i]}:", mean_)
            end = time.time()
            minutes = (end-start)/60
            PrintLog(f"completed in {minutes} minutes")
