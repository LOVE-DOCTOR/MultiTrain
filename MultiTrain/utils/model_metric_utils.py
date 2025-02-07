from MultiTrain.errors.errors import *
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    RidgeClassifierCV,
    Perceptron,
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
    SGDRegressor,
)

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeRegressor,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_log_error,
    explained_variance_score,
)

from utils.utils import inMultiClassifier, inMultiRegressor

gpu_multiclassifier = inMultiClassifier()
gpu_multiregressor = inMultiRegressor()

use_gpu_classifier = gpu_multiclassifier.use_gpu
use_gpu_regressor = gpu_multiregressor.use_gpu

if use_gpu_classifier or use_gpu_regressor:
    from sklearnex import patch_sklearn
    patch_sklearn()
    set_gpu = True


def _models_classifier(random_state, n_jobs, max_iter):
    """
    Returns a dictionary of classifier models from various libraries.

    Each key is a string representing the name of the classifier, and the value is an instance of the classifier.

    Args:
        random_state (int): Seed used by the random number generator.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        dict: A dictionary of classifier models.
    """
    models_dict = {
        LogisticRegression.__name__: LogisticRegression(
            random_state=random_state, n_jobs=n_jobs, max_iter=max_iter
        ),
        LogisticRegressionCV.__name__: LogisticRegressionCV(
            n_jobs=n_jobs,
            max_iter=max_iter,
            cv=5,
        ),
        SGDClassifier.__name__: SGDClassifier(n_jobs=n_jobs, max_iter=max_iter),
        PassiveAggressiveClassifier.__name__: PassiveAggressiveClassifier(
            n_jobs=n_jobs, max_iter=max_iter
        ),
        RidgeClassifier.__name__: RidgeClassifier(max_iter=max_iter),
        RidgeClassifierCV.__name__: RidgeClassifierCV(cv=5),
        Perceptron.__name__: Perceptron(n_jobs=n_jobs, max_iter=max_iter),
        LinearSVC.__name__: LinearSVC(random_state=random_state, max_iter=max_iter),
        NuSVC.__name__: NuSVC(random_state=random_state, max_iter=max_iter),
        SVC.__name__: SVC(random_state=random_state, max_iter=max_iter),
        KNeighborsClassifier.__name__: KNeighborsClassifier(n_jobs=n_jobs),
        MLPClassifier.__name__: MLPClassifier(
            random_state=random_state, max_iter=max_iter
        ),
        GaussianNB.__name__: GaussianNB(),
        BernoulliNB.__name__: BernoulliNB(),
        MultinomialNB.__name__: MultinomialNB(),
        ComplementNB.__name__: ComplementNB(),
        DecisionTreeClassifier.__name__: DecisionTreeClassifier(
            random_state=random_state
        ),
        ExtraTreeClassifier.__name__: ExtraTreeClassifier(random_state=random_state),
        GradientBoostingClassifier.__name__: GradientBoostingClassifier(
            random_state=random_state
        ),
        ExtraTreesClassifier.__name__: ExtraTreesClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        BaggingClassifier.__name__: BaggingClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        CatBoostClassifier.__name__: CatBoostClassifier(
            random_state=random_state,
            thread_count=n_jobs,
            silent=True,
            iterations=max_iter,
        ),
        RandomForestClassifier.__name__: RandomForestClassifier(
            random_state=random_state, n_jobs=n_jobs
        ),
        AdaBoostClassifier.__name__: AdaBoostClassifier(
            random_state=random_state, n_estimators=max_iter
        ),
        HistGradientBoostingClassifier.__name__: HistGradientBoostingClassifier(
            random_state=random_state, max_iter=max_iter
        ),
        LGBMClassifier.__name__: LGBMClassifier(
            random_state=random_state, n_jobs=n_jobs, verbose=-1, n_estimators=max_iter
        ),
        XGBClassifier.__name__: XGBClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter,
        ),
    }
    
    if set_gpu is True:
        models_dict[CatBoostClassifier.__name__].set_params(task_type='GPU', devices=gpu_multiclassifier.device)
        models_dict[XGBClassifier.__name__].set_params(tree_method='gpu_hist', predictor='gpu_predictor')

    return models_dict

def _models_regressor(random_state, n_jobs, max_iter):
    """
    Returns a dictionary of regressor models from various libraries.

    Each key is a string representing the name of the regressor, and the value is an instance of the regressor.

    Args:
        random_state (int): Seed used by the random number generator.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        dict: A dictionary of regressor models.
    """
    models_dict = {
        LinearRegression.__name__: LinearRegression(n_jobs=n_jobs),
        Ridge.__name__: Ridge(random_state=random_state),
        RidgeCV.__name__: RidgeCV(),
        Lasso.__name__: Lasso(random_state=random_state),
        LassoCV.__name__: LassoCV(),
        ElasticNet.__name__: ElasticNet(random_state=random_state),
        ElasticNetCV.__name__: ElasticNetCV(),
        SGDRegressor.__name__: SGDRegressor(
            random_state=random_state, max_iter=max_iter
        ),
        KNeighborsRegressor.__name__: KNeighborsRegressor(n_jobs=n_jobs),
        DecisionTreeRegressor.__name__: DecisionTreeRegressor(
            random_state=random_state
        ),
        ExtraTreeRegressor.__name__: ExtraTreeRegressor(random_state=random_state),
        RandomForestRegressor.__name__: RandomForestRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        ExtraTreesRegressor.__name__: ExtraTreesRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        GradientBoostingRegressor.__name__: GradientBoostingRegressor(
            random_state=random_state
        ),
        AdaBoostRegressor.__name__: AdaBoostRegressor(
            random_state=random_state, n_estimators=max_iter
        ),
        BaggingRegressor.__name__: BaggingRegressor(
            random_state=random_state, n_jobs=n_jobs
        ),
        CatBoostRegressor.__name__: CatBoostRegressor(
            random_state=random_state,
            thread_count=n_jobs,
            silent=True,
            iterations=max_iter,
        ),
        LGBMRegressor.__name__: LGBMRegressor(
            random_state=random_state, n_jobs=n_jobs, verbose=-1, n_estimators=max_iter
        ),
        XGBRegressor.__name__: XGBRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            verbose=False,
            n_estimators=max_iter,
        ),
        HistGradientBoostingRegressor.__name__: HistGradientBoostingRegressor(
            random_state=random_state, max_iter=max_iter
        ),
    }
    
    if set_gpu is True:
        models_dict[CatBoostRegressor.__name__].set_params(task_type='GPU', devices=gpu_multiclassifier.device)
        models_dict[XGBRegressor.__name__].set_params(tree_method='gpu_hist', predictor='gpu_predictor')

    return models_dict


def _init_metrics():
    """
    Initializes a list of default metric names.

    Returns:
        list: A list of metric names.
    """
    return [
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "roc_auc_score",
        "balanced_accuracy_score",
    ]


def _metrics(custom_metric: str, metric_type: str):
    """
    Returns a dictionary of metric functions from sklearn.

    Each key is a string representing the name of the metric, and the value is the metric function.

    Args:
        custom_metric (str): Name of a custom metric to include.
        metric_type (str): 'classification' or 'regression'

    Returns:
        dict: A dictionary of metric functions.
    """
    valid_metrics = {
        "classification": {
            "precision": precision_score,
            "recall": recall_score,
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        },
        "regression": {
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "mean_absolute_error": mean_absolute_error,
            "median_absolute_error": median_absolute_error,
            "mean_squared_log_error": mean_squared_log_error,
            "explained_variance_score": explained_variance_score,
        },
    }

    if custom_metric:
        # Check if the custom metric is a valid sklearn metric
        valid_sklearn_metrics = [
            name
            for name, obj in inspect.getmembers(sklearn.metrics, inspect.isfunction)
        ]
        if custom_metric not in valid_sklearn_metrics:
            raise MultiTrainMetricError(
                f"Custom metric ({custom_metric}) is not a valid metric. Please check the sklearn documentation for a valid list of metrics."
            )
        # Add the custom metric to the appropriate metric type
        metrics = valid_metrics.get(metric_type, {}).copy()
        metrics[custom_metric] = globals().get(custom_metric)
        return metrics

    return valid_metrics.get(metric_type, {})
