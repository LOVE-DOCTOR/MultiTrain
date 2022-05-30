# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42) In your
# code, you can adjust the names of X_train, X_test, y_train and y_test if you named them differently when splitting
# (line 58 - 60)
from sklearnex import patch_sklearn
from sklearnex.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearnex.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearnex.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

patch_sklearn()


class Models:

    def __init__(self, lr=0, sgdc=0, pagg=0, rfc=0, gbc=0, cat=0, xgb=0, gnb=0, lda=0, knc=0, mlp=0, svc=0,
                 index=0, highest_acc=0, check_best_model=[]) -> None:
        """
        :param lr = Logistic Regression
        :param sgdc = Stochastic Gradient Descent Classifier
        :param pagg = Passive Aggressive Classifier
        :param rfc = Random Forest Classifier
        :param gbc = Gradient Boosting Classifier
        :param cat = CatBoostClassifier
        :param xgb = XGBoost Classifier
        :param gnb = GaussianNB Classifier
        :param lda = Linear Discriminant Analysis
        :param knc = KNeighborsClassifier
        :param mlp = MLPClassifier
        :param svc = Support Vector Classifier
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

    def initialize(self):
        self.lr = LogisticRegression(random_state=42, warm_start=True)
        self.sgdc = SGDClassifier(random_state=42)
        self.pagg = PassiveAggressiveClassifier(shuffle=True, fit_intercept=True, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=20, n_jobs=-1)
        self.rfc = RandomForestClassifier(random_state=42, warm_start=True)
        self.gbc = GradientBoostingClassifier(random_state=42, learning_rate=0.01, validation_fraction=0.1,
                                              n_iter_no_change=20)
        self.cat = CatBoostClassifier(random_state=42, learning_rate=0.01)
        self.xgb = XGBClassifier()
        self.gnb = GaussianNB()
        self.lda = LinearDiscriminantAnalysis()
        self.knc = KNeighborsClassifier(n_jobs=-1)
        self.mlp = MLPClassifier(batch_size=10, shuffle=True, random_state=42, warm_start=True,
                                 early_stopping=True, n_iter_no_change=20)
        self.svc = SVC(random_state=42)

        return self.lr, self.sgdc, self.pagg, self.rfc, self.gbc, self.cat, self.xgb, self.gnb, self.lda, self.knc, self.mlp, self.svc

    def fit_eval_models(self, X_train="X_train", X_test="X_test", y_train="y_train", y_test="y_test"):
        """
        This function takes in the training and testing data and fits the model to the training data. It then predicts the
        testing data and compares the predicted values to the actual values.

        :param X_train: The training data, defaults to X_train (optional)
        :param X_test: The test data, defaults to X_test (optional)
        :param y_train: The target variable for supervised learning, defaults to y_train (optional)
        :param y_test: The actual values of the target variable, defaults to y_test (optional)
        """
        model = self.initialize()
        check_best_model = []
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
            check_best_model.append(acc)
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
        highest_acc = max(check_best_model)
        index = check_best_model.index(highest_acc)


# create an instance of the class and run fit_eval_models to begin training.
fire = Models()
fire.fit_eval_models()
