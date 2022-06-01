# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42) In your
# code, you can adjust the names of X_train, X_test, y_train and y_test if you named them differently when splitting
# (line 58 - 60)
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
from sklearnex.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearnex.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearnex.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

patch_sklearn()


def split(X: any, y: any, strat: bool = False, sizeOfTest: float = 0.2, randomState: int = None,
          shuffle_data: bool = True) -> tuple[any, any, any, any]:
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


class Models:

    def __init__(self, lr=0, sgdc=0, pagg=0, rfc=0, gbc=0, cat=0, xgb=0, gnb=0, lda=0, knc=0, mlp=0, svc=0,
                 dtc=0, bnb=0, mnb=0, cnb=0, conb=0, hgbc=0, abc=0, etc=0) -> None:
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
        :param knc: KNeighborsClassifier
        :param mlp: MLPClassifier
        :param svc: Support Vector Classifier
        :param dtc: DecisionTreeClassifier
        :param cnb: CategoricalNB
        :param conb: ComplementNB
        :param hgbc: HistGradientBoostingClassifier
        :param abc: AdaBoostClassifier
        :param etc: ExtraTreesClassifier
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
        self.etc = etc

    def initialize(self):
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
        self.etc = ExtraTreesClassifier(warm_start=True, random_state=42, n_jobs=-1)

        return (self.lr, self.sgdc, self.pagg, self.rfc, self.gbc, self.hgbc, self.abc, self.cat, self.xgb, self.gnb,
                self.lda, self.knc, self.mlp, self.svc, self.dtc, self.bnb, self.mnb, self.cnb, self.conb, self.etc)

    def fit_eval_models(self, X_train=None, X_test=None, y_train=None, y_test=None,
                        split_data: str = None, splitting: bool = False):

        if isinstance(splitting, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in splitting. Try splitting = False or splitting = True "
                f"instead of splitting = {splitting}")

        elif splitting is False:
            pass
        elif splitting and split_data:
            X_train, X_test, y_train, y_test = split_data[0], split_data[1], split_data[2], split_data[3]

        model = self.initialize()
        for i in range(len(model)):
            model[i].fit(X_train, y_train)
            pred = model[i].predict(X_test)
            true = y_test

            acc = accuracy_score(true, pred)
            # f1 = f1_score(true, pred)
            mae = mean_absolute_error(true, pred)
            mse = mean_squared_error(true, pred)
            clr = classification_report(true, pred)
            cfm = confusion_matrix(true, pred)
            r2 = r2_score(true, pred)

            print("The model used is ", model[i])
            print("The Accuracy of the Model is ", acc)
            # print("The f1 score of the Model is ", f1)
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
