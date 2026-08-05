import numpy as np
import pandas as pd
import pytest
import warnings
from types import SimpleNamespace
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from MultiTrain.classification.classification_models import (
    MultiClassifier,
    subMultiClassifier,
)
import MultiTrain.classification.classification_models as classification_module
from MultiTrain.errors import errors as error_module
from MultiTrain.errors.errors import (
    MultiTrainColumnMissingError,
    MultiTrainDatasetTypeError,
    MultiTrainEncodingError,
    MultiTrainError,
    MultiTrainMetricError,
    MultiTrainModelError,
    MultiTrainNaNError,
    MultiTrainPCAError,
    MultiTrainSplitError,
    MultiTrainTextError,
    MultiTrainTypeError,
)
from MultiTrain.regression.regression_models import (
    MultiRegressor,
    subMultiRegressor,
)
import MultiTrain.regression.regression_models as regression_module
from MultiTrain.utils import utils


@pytest.fixture
def binary_data():
    return pd.DataFrame(
        {
            "number": range(12),
            "category": ["a", "b"] * 6,
            "target": [0, 1] * 6,
        }
    )


@pytest.fixture
def regression_data():
    return pd.DataFrame(
        {
            "number": range(12),
            "category": ["a", "b"] * 6,
            "target": [float(value * 2) for value in range(12)],
        }
    )


def test_every_error_class_is_catchable_and_preserves_message():
    error_classes = [
        value
        for name, value in vars(error_module).items()
        if name.startswith("MultiTrain") and isinstance(value, type)
    ]
    assert error_classes
    for error_class in error_classes:
        with pytest.raises(Exception, match="message"):
            raise error_class("message")


def test_model_factories_apply_defaults_and_requested_values():
    classifiers = utils._models_classifier(random_state=7, n_jobs=2, max_iter=11)
    regressors = utils._models_regressor(random_state=7, n_jobs=2, max_iter=11)

    assert classifiers["LogisticRegression"].max_iter == 11
    assert classifiers["RandomForestClassifier"].n_jobs == 2
    assert classifiers["CatBoostClassifier"].get_param("task_type") is None
    assert classifiers["CatBoostClassifier"].get_param("random_seed") == 7
    assert classifiers["XGBClassifier"].get_params()["tree_method"] is None
    assert regressors["Ridge"].max_iter == 11
    assert regressors["LinearRegression"].n_jobs == 2
    assert regressors["OrthogonalMatchingPursuitCV"].max_iter is None
    assert regressors["OrthogonalMatchingPursuitCV"].n_jobs == 2
    assert regressors["CatBoostRegressor"].get_param("task_type") is None
    assert regressors["CatBoostRegressor"].get_param("random_seed") == 7
    assert regressors["XGBRegressor"].get_params()["tree_method"] is None


def test_model_factory_defaults_for_every_configurable_estimator():
    classifiers = utils._models_classifier()
    classifier_defaults = {
        "LogisticRegression": ("max_iter", 100),
        "LogisticRegressionCV": ("max_iter", 100),
        "SGDClassifier": ("max_iter", 1000),
        "PassiveAggressiveClassifier": ("max_iter", 1000),
        "RidgeClassifier": ("max_iter", 1000),
        "Perceptron": ("max_iter", 1000),
        "LinearSVC": ("max_iter", 1000),
        "NuSVC": ("max_iter", 1000),
        "SVC": ("max_iter", 1000),
        "MLPClassifier": ("max_iter", 1000),
        "AdaBoostClassifier": ("n_estimators", 50),
        "HistGradientBoostingClassifier": ("max_iter", 100),
        "LGBMClassifier": ("n_estimators", 100),
        "XGBClassifier": ("n_estimators", 100),
    }
    for model_name, (parameter, expected) in classifier_defaults.items():
        assert classifiers[model_name].get_params()[parameter] == expected
    assert classifiers["CatBoostClassifier"].get_param("iterations") == 1000
    assert classifiers["CatBoostClassifier"].get_param("verbose") is False
    assert classifiers["CatBoostClassifier"].get_param("allow_writing_files") is False
    assert classifiers["LGBMClassifier"].get_params()["verbosity"] == -1
    assert "verbose" not in classifiers["XGBClassifier"].get_params()
    for model_name in [
        "LogisticRegression",
        "LogisticRegressionCV",
        "SGDClassifier",
        "PassiveAggressiveClassifier",
        "Perceptron",
        "KNeighborsClassifier",
        "ExtraTreesClassifier",
        "BaggingClassifier",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
    ]:
        assert classifiers[model_name].get_params()["n_jobs"] == 1

    regressors = utils._models_regressor()
    regressor_defaults = {
        "Ridge": ("max_iter", 1000),
        "Lasso": ("max_iter", 1000),
        "LassoCV": ("max_iter", 1000),
        "ElasticNet": ("max_iter", 1000),
        "ElasticNetCV": ("max_iter", 1000),
        "OrthogonalMatchingPursuitCV": ("max_iter", None),
        "BayesianRidge": ("max_iter", 300),
        "ARDRegression": ("max_iter", 300),
        "HuberRegressor": ("max_iter", 100),
        "TheilSenRegressor": ("max_iter", 300),
        "RANSACRegressor": ("max_trials", 100),
        "PoissonRegressor": ("max_iter", 100),
        "GammaRegressor": ("max_iter", 100),
        "TweedieRegressor": ("max_iter", 100),
        "SGDRegressor": ("max_iter", 1000),
        "PassiveAggressiveRegressor": ("max_iter", 1000),
        "GradientBoostingRegressor": ("n_estimators", 100),
        "AdaBoostRegressor": ("n_estimators", 50),
        "MLPRegressor": ("max_iter", 1000),
        "SVR": ("max_iter", -1),
        "LinearSVR": ("max_iter", 1000),
        "NuSVR": ("max_iter", -1),
        "LGBMRegressor": ("n_estimators", 100),
        "XGBRegressor": ("n_estimators", 100),
        "HistGradientBoostingRegressor": ("max_iter", 100),
    }
    for model_name, (parameter, expected) in regressor_defaults.items():
        assert regressors[model_name].get_params()[parameter] == expected
    assert regressors["CatBoostRegressor"].get_param("iterations") == 1000
    assert regressors["CatBoostRegressor"].get_param("verbose") is False
    assert regressors["CatBoostRegressor"].get_param("allow_writing_files") is False
    assert regressors["LGBMRegressor"].get_params()["verbosity"] == -1
    assert "verbose" not in regressors["XGBRegressor"].get_params()


@pytest.mark.parametrize("factory,catboost_name,xgboost_name", [
    (utils._models_classifier, "CatBoostClassifier", "XGBClassifier"),
    (utils._models_regressor, "CatBoostRegressor", "XGBRegressor"),
])
def test_model_factories_propagate_gpu_settings(
    monkeypatch, factory, catboost_name, xgboost_name
):
    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")

    models = factory(use_gpu=True, device="2", max_iter=2)

    assert models[catboost_name].get_param("task_type") == "GPU"
    assert models[catboost_name].get_param("devices") == "2"
    assert models[xgboost_name].get_params()["tree_method"] == "hist"
    assert models[xgboost_name].get_params()["device"] == "cuda:2"


def test_model_factories_skip_gpu_patch_on_macos(monkeypatch):
    monkeypatch.setattr(utils.platform, "system", lambda: "Darwin")
    models = utils._models_classifier(use_gpu=True, max_iter=2)
    assert models["CatBoostClassifier"].get_param("task_type") is None


def test_encoders_cover_success_and_validation_paths():
    frame = pd.DataFrame({"cat": ["a", "b"], "other": ["x", "y"]})

    with pytest.raises(MultiTrainEncodingError):
        utils._cat_encoder(frame, False)
    with pytest.raises(MultiTrainTypeError):
        utils._manual_encoder([], frame)
    with pytest.raises(MultiTrainDatasetTypeError):
        utils._manual_encoder({"label": ["cat"]}, [])
    with pytest.raises(MultiTrainEncodingError):
        utils._manual_encoder({"ordinal": ["cat"]}, frame)
    with pytest.raises(MultiTrainTypeError):
        utils._manual_encoder({"label": "cat"}, frame)
    with pytest.raises(MultiTrainColumnMissingError):
        utils._manual_encoder({"label": ["missing"]}, frame)

    encoded = utils._manual_encoder(
        {"label": ("cat",), "onehot": ["other"]}, frame
    )
    assert encoded["cat"].tolist() == [0, 1]
    assert {"other_x", "other_y"}.issubset(encoded.columns)


def test_non_auto_encoder_validation_allows_complete_manual_mapping():
    frame = pd.DataFrame({"a": ["x"], "b": ["y"]})
    assert utils._non_auto_cat_encode_error(frame, True, None) is None
    assert (
        utils._non_auto_cat_encode_error(
            frame, False, {"label": ["a"], "onehot": ["b"]}
        )
        is None
    )
    with pytest.raises(MultiTrainEncodingError):
        utils._non_auto_cat_encode_error(frame, False, {"label": ["a"]})


def test_fill_missing_values_handles_numeric_categorical_and_all_null():
    numeric = pd.DataFrame({"value": [1.0, np.nan]})
    categorical = pd.DataFrame({"value": ["a", None, "a"]})
    nullable_string = pd.DataFrame(
        {"value": pd.Series(["a", None, "a"], dtype="string")}
    )
    all_null = pd.DataFrame({"value": pd.Series([None, None], dtype="object")})

    assert utils._fill_missing_values(numeric, "value").tolist() == [1.0, 0.0]
    assert utils._fill_missing_values(categorical, "value").tolist() == ["a", "a", "a"]
    assert utils._fill_missing_values(nullable_string, "value").tolist() == [
        "a",
        "a",
        "a",
    ]
    assert utils._fill_missing_values(all_null, "value").tolist() == ["", ""]


def test_handle_missing_values_covers_noop_validation_and_partial_config():
    clean = pd.DataFrame({"a": [1, 2]})
    assert utils._handle_missing_values(clean).equals(clean)
    with pytest.raises(MultiTrainTypeError):
        utils._handle_missing_values(clean, [])
    with pytest.raises(MultiTrainNaNError):
        utils._handle_missing_values(pd.DataFrame({"a": [1, None]}))
    with pytest.raises(MultiTrainNaNError):
        utils._handle_missing_values(
            pd.DataFrame({"a": [1, None], "b": [None, 2]}), {"a": "ffill"}
        )

    leading_null = pd.DataFrame({"a": [None, 2.0, 3.0]})
    assert not utils._handle_missing_values(leading_null, {"a": "ffill"})["a"].isna().any()
    trailing_null = pd.DataFrame({"a": [1.0, 2.0, None]})
    assert not utils._handle_missing_values(trailing_null, {"a": "bfill"})["a"].isna().any()
    interpolated = utils._handle_missing_values(
        pd.DataFrame({"a": [None, 2.0, 4.0]}), {"a": "interpolate"}
    )
    assert interpolated["a"].tolist() == [0.0, 2.0, 4.0]


def test_prepare_train_test_validates_schema_options_and_missing_values():
    train = pd.DataFrame({"a": ["x", None], "b": [1.0, 2.0]})
    test = pd.DataFrame({"a": [None], "b": [3.0]})

    with pytest.raises(MultiTrainDatasetTypeError):
        utils._prepare_train_test([], test)
    with pytest.raises(MultiTrainColumnMissingError):
        utils._prepare_train_test(train, test.drop(columns="b"))
    with pytest.raises(MultiTrainTypeError):
        utils._prepare_train_test(train, test, manual_encode=[])
    with pytest.raises(MultiTrainTypeError):
        utils._prepare_train_test(train, test, fix_nan_custom=[])
    with pytest.raises(MultiTrainEncodingError):
        utils._prepare_train_test(train, test, manual_encode={"bad": ["a"]})
    with pytest.raises(MultiTrainNaNError):
        utils._prepare_train_test(train, test)
    with pytest.raises(MultiTrainColumnMissingError):
        utils._prepare_train_test(train, test, fix_nan_custom={"missing": "ffill"})
    with pytest.raises(MultiTrainNaNError):
        utils._prepare_train_test(train, test, fix_nan_custom={"a": "bad"})
    with pytest.raises(MultiTrainTypeError):
        utils._prepare_train_test(
            train.fillna("x"), test.fillna("x"), manual_encode={"label": "a"}
        )

    encoded_train, encoded_test = utils._prepare_train_test(
        train,
        test,
        manual_encode={"label": ["a"]},
        fix_nan_custom={"a": "ffill"},
    )
    assert not encoded_train.isna().any().any()
    assert not encoded_test.isna().any().any()


def test_prepare_train_test_onehot_aligns_unknown_test_categories():
    train = pd.DataFrame({"cat": ["a", "b"], "target": [1, 2]})
    test = pd.DataFrame({"cat": ["c"], "target": [3]})
    prepared_train, prepared_test = utils._prepare_train_test(
        train, test, manual_encode={"onehot": ["cat"]}
    )
    assert list(prepared_train.columns) == list(prepared_test.columns)
    assert prepared_test[["cat_a", "cat_b"]].sum(axis=1).iloc[0] == 0


def test_prepare_train_test_rejects_unhandled_and_missing_encoder_columns():
    train = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    test = pd.DataFrame({"a": [3.0], "b": [4.0]})
    with pytest.raises(MultiTrainNaNError):
        utils._prepare_train_test(train, test, fix_nan_custom={"a": "ffill"})

    clean_train = train.fillna(0)
    with pytest.raises(MultiTrainColumnMissingError):
        utils._prepare_train_test(
            clean_train, test, manual_encode={"label": ["missing"]}
        )
    with pytest.raises(MultiTrainColumnMissingError):
        utils._prepare_train_test(
            clean_train, test, manual_encode={"onehot": ["missing"]}
        )


def test_custom_model_selection_all_paths():
    models = {"one": object(), "two": object()}
    names, selected = utils._check_custom_models(None, models)
    assert names == ["one", "two"] and len(selected) == 2
    with pytest.raises(MultiTrainModelError):
        utils._check_custom_models(["one", "one"], models)
    with pytest.raises(MultiTrainModelError):
        utils._check_custom_models(["missing"], models)
    with pytest.raises(MultiTrainTypeError):
        utils._check_custom_models("one", models)


def test_prep_model_names_validates_every_dispatch_path(monkeypatch):
    split = (np.array([[1]]), np.array([[2]]), np.array([0]), np.array([1]))
    monkeypatch.setattr(utils, "_models_classifier", lambda **kwargs: {"c": object()})
    monkeypatch.setattr(utils, "_models_regressor", lambda **kwargs: {"r": object()})

    assert utils._prep_model_names_list(split, None, 1, 1, ["c"], "classification", 2)[0] == ["c"]
    assert utils._prep_model_names_list(split, None, 1, 1, ["r"], "regression", 2)[0] == ["r"]
    with pytest.raises(MultiTrainSplitError):
        utils._prep_model_names_list([], None, 1, 1, None, "classification", 2)
    with pytest.raises(MultiTrainTypeError):
        utils._prep_model_names_list(split, 123, 1, 1, None, "classification", 2)
    with pytest.raises(MultiTrainMetricError):
        utils._prep_model_names_list(split, "accuracy_score", 1, 1, None, "classification", 2)
    with pytest.raises(MultiTrainTypeError):
        utils._prep_model_names_list(split, None, 1, 1, None, "unknown", 2)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0.00us"), (0.00005, "50.00us"), (0.02, "20.00ms"), (30.205, "30.20s")],
)
def test_format_time_all_units(seconds, expected):
    assert utils._format_time(seconds) == expected


def test_format_time_rejects_invalid_values():
    with pytest.raises(MultiTrainTypeError):
        utils._format_time("1")
    with pytest.raises(MultiTrainTypeError):
        utils._format_time(True)
    with pytest.raises(MultiTrainError):
        utils._format_time(-1)


class BrokenEstimator(BaseEstimator):
    def fit(self, X, y):
        raise RuntimeError("broken")

    def predict(self, X):
        raise RuntimeError("broken")


def test_sub_fit_covers_pca_failure_and_reraise_paths():
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    X_test = np.array([[4.0, 5.0]])

    pipeline, prediction = utils._sub_fit(
        LinearRegression(), X_train, y_train, X_test, StandardScaler()
    )
    assert len(pipeline.steps) == 3 and len(prediction) == 1

    quantile_pipeline, _ = utils._sub_fit(
        LinearRegression(), X_train, y_train, X_test, QuantileTransformer()
    )
    assert quantile_pipeline.named_steps["QuantileTransformer"].n_quantiles == len(
        X_train
    )

    pipeline, prediction = utils._sub_fit(
        BrokenEstimator(), X_train, y_train, X_test, False
    )
    assert pipeline is not None and np.isnan(prediction).all()
    with pytest.raises(RuntimeError):
        utils._sub_fit(
            BrokenEstimator(), X_train, y_train, X_test, False, raise_on_error=True
        )
    with pytest.raises(MultiTrainTypeError):
        utils._sub_fit(LinearRegression(), [1, 2], y_train[:2], [[3]], False)


def test_fit_pred_accepts_gpu_options_for_models_that_manage_their_own_device():
    _, predictions, _ = utils._fit_pred(
        LogisticRegression(),
        ["LogisticRegression"],
        0,
        np.array([[0], [1], [2], [3]]),
        np.array([0, 0, 1, 1]),
        np.array([[1], [2]]),
        False,
        use_gpu=True,
        device="3",
    )
    assert predictions.tolist() == [0, 1]


def test_calculate_metric_handles_average_nan_and_metric_errors():
    assert utils._calculate_metric(accuracy_score, [0, 1], [0, 1]) == 1
    assert np.isnan(utils._calculate_metric(accuracy_score, [0], [np.nan]))
    assert np.isnan(utils._calculate_metric(None, [0], [0]))
    weighted = utils._calculate_metric(
        utils.precision_score, [0, 1, 2], [0, 1, 1], average="weighted"
    )
    assert 0 <= weighted <= 1
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        undefined_precision = utils._calculate_metric(
            utils.precision_score, [0, 0], [0, 0]
        )
    assert undefined_precision == 0
    assert not captured


def test_metrics_reject_invalid_metric_type_and_custom_metric_type():
    with pytest.raises(MultiTrainTypeError):
        utils._metrics(None, "unknown")
    with pytest.raises(MultiTrainTypeError):
        utils._metrics(123, "classification")


class ProbabilityModel:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return self.probabilities


class DecisionModel:
    classes_ = np.array([0, 1])

    def decision_function(self, X):
        return np.array([-1.0, 1.0])


def test_classification_roc_auc_all_score_paths():
    binary = ProbabilityModel([0, 1], [[0.9, 0.1], [0.1, 0.9]])
    assert utils._classification_roc_auc(binary, [[0], [1]], [0, 1]) == 1
    multi = ProbabilityModel(
        [0, 1, 2],
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
    )
    assert utils._classification_roc_auc(multi, [[0], [1], [2]], [0, 1, 2]) == 1
    assert utils._classification_roc_auc(DecisionModel(), [[0], [1]], [0, 1]) == 1
    assert np.isnan(utils._classification_roc_auc(binary, [[0]], [0]))
    assert np.isnan(utils._classification_roc_auc(object(), [[0], [1]], [0, 1]))
    bad = ProbabilityModel([0, 1], [[0.5], [0.5]])
    assert np.isnan(utils._classification_roc_auc(bad, [[0], [1]], [0, 1]))


def test_fit_pred_text_validates_every_input():
    model = LogisticRegression()
    data = pd.Series(["red", "blue", "red sky", "blue sea"])
    target = pd.Series([0, 1, 0, 1])
    config = {
        "ngram_range": (1, 1),
        "encoding": "utf-8",
        "max_features": 20,
        "analyzer": "word",
    }
    with pytest.raises(MultiTrainPCAError):
        utils._fit_pred_text("count", config, model, data, target, data, True)
    with pytest.raises(MultiTrainTextError):
        utils._fit_pred_text("bad", config, model, data, target, data, False)
    with pytest.raises(MultiTrainTextError):
        utils._fit_pred_text("count", None, model, data, target, data, False)
    with pytest.raises(MultiTrainTextError):
        utils._fit_pred_text("count", {"encoding": "utf-8"}, model, data, target, data, False)

    _, prediction, _ = utils._fit_pred_text(
        "count", config, model, data, target, pd.Series(["red", "blue"]), False
    )
    assert len(prediction) == 2


def test_fit_pred_text_dense_fallback_accepts_gpu_options():
    config = {
        "ngram_range": (1, 1),
        "encoding": "utf-8",
        "max_features": 20,
        "analyzer": "word",
    }
    data = pd.Series(["red", "blue", "red sky", "blue sea"])
    _, predictions, _ = utils._fit_pred_text(
        "count",
        config,
        utils.GaussianNB(),
        data,
        pd.Series([0, 1, 0, 1]),
        pd.Series(["red", "blue"]),
        False,
        use_gpu=True,
        device="4",
    )
    assert not pd.isna(predictions).any()


def test_display_table_all_sorting_and_validation_paths():
    classification = {
        "a": {"accuracy": 0.5, "custom": 0.2},
        "b": {"accuracy": 0.9, "custom": 0.1},
    }
    regression = {
        "a": {"mean_squared_error": 4.0, "r2_score": 0.8},
        "b": {"mean_squared_error": 1.0, "r2_score": 0.2},
    }
    assert list(utils._display_table(classification, sort="accuracy", task="classification").index) == ["b", "a"]
    assert list(utils._display_table(regression, sort="mean_squared_error", task="regression").index) == ["b", "a"]
    assert list(utils._display_table(classification, return_best_model="accuracy", task="classification").index) == ["b"]
    assert list(utils._display_table(regression, return_best_model="r2_score", task="regression").index) == ["a"]
    assert list(utils._display_table(regression, return_best_model="mean_squared_error", task="regression").index) == ["b"]

    with pytest.raises(MultiTrainTypeError):
        utils._display_table(classification, task="bad")
    with pytest.raises(MultiTrainTypeError):
        utils._display_table({}, task="classification")
    with pytest.raises(MultiTrainError):
        utils._display_table(classification, sort="accuracy", return_best_model="accuracy", task="classification")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table(classification, sort="missing", task="classification")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table(classification, sort="custom", custom_metric="custom", task="classification")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table(classification, return_best_model="missing", task="classification")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table(classification, return_best_model="custom", task="classification")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table({"a": {"custom": 1}}, return_best_model="custom", task="regression")
    with pytest.raises(MultiTrainMetricError):
        utils._display_table(classification, sort="", task="classification")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": "all"},
        {"random_state": "42"},
        {"max_iter": 1.5},
        {"use_gpu": "yes"},
        {"device": 0},
        {"text": "yes"},
        {"custom_models": "LogisticRegression"},
    ],
)
def test_classifier_constructor_validates_every_field(kwargs):
    with pytest.raises(MultiTrainTypeError):
        MultiClassifier(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": "all"},
        {"random_state": "42"},
        {"max_iter": 1.5},
        {"use_gpu": "yes"},
        {"device": 0},
        {"custom_models": "LinearRegression"},
    ],
)
def test_regressor_constructor_validates_every_field(kwargs):
    with pytest.raises(MultiTrainTypeError):
        MultiRegressor(**kwargs)


def test_gpu_constructor_paths_and_subclasses(monkeypatch):
    monkeypatch.setattr("MultiTrain.classification.classification_models.platform.system", lambda: "Windows")
    assert MultiClassifier(use_gpu=True).use_gpu
    assert MultiRegressor(use_gpu=True).use_gpu
    classifier_subclass = subMultiClassifier()
    regressor_subclass = subMultiRegressor()
    assert isinstance(classifier_subclass, MultiClassifier)
    assert isinstance(regressor_subclass, MultiRegressor)
    for subclass in (classifier_subclass, regressor_subclass):
        assert subclass.n_jobs == 1
        assert subclass.model_workers is None
        assert subclass.random_state == 42
        assert subclass.custom_models is None
        assert subclass.max_iter == 1000
        assert subclass.use_gpu is False
        assert subclass.device == "0"
    with pytest.raises(MultiTrainTypeError):
        subMultiClassifier(use_gpu="yes")
    with pytest.raises(MultiTrainTypeError):
        subMultiClassifier(device=1)
    with pytest.raises(MultiTrainTypeError):
        subMultiRegressor(use_gpu="yes")
    with pytest.raises(MultiTrainTypeError):
        subMultiRegressor(device=1)


def test_gpu_constructors_remain_valid_on_macos(monkeypatch):
    monkeypatch.setattr("MultiTrain.classification.classification_models.platform.system", lambda: "Darwin")
    assert MultiClassifier(use_gpu=True).use_gpu
    assert MultiRegressor(use_gpu=True).use_gpu


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
def test_split_accepts_csv_and_validates_input_options(tmp_path, model_class, binary_data):
    path = tmp_path / "data.csv"
    binary_data.to_csv(path, index=False)
    model = model_class(custom_models=["LogisticRegression"] if model_class is MultiClassifier else ["LinearRegression"])
    assert len(model.split(str(path), "target", auto_cat_encode=True)) == 4
    dropped = model.split(
        binary_data, "target", drop=["number"], auto_cat_encode=True
    )
    assert "number" not in dropped[0].columns

    with pytest.raises(MultiTrainDatasetTypeError):
        model.split(123, "target")
    with pytest.raises(MultiTrainTypeError):
        model.split(binary_data, "target", manual_encode=[])
    with pytest.raises(MultiTrainTypeError):
        model.split(binary_data, "target", fix_nan_custom=[])
    with pytest.raises(MultiTrainEncodingError):
        model.split(binary_data, "target", manual_encode={"bad": ["category"]})
    with pytest.raises(MultiTrainTypeError):
        model.split(binary_data, "target", manual_encode={"label": "category"})
    with pytest.raises(MultiTrainEncodingError):
        model.split(binary_data, "target", manual_encode={"label": ["category"], "onehot": ["category"]})
    with pytest.raises(MultiTrainEncodingError):
        model.split(binary_data, "target", auto_cat_encode=True, manual_encode={"label": ["category"]})
    with pytest.raises(MultiTrainEncodingError):
        model.split(binary_data, "target", manual_encode={"onehot": ["target", "category"]})
    with pytest.raises(MultiTrainColumnMissingError):
        model.split(binary_data, "target", drop=["missing"], auto_cat_encode=True)
    with pytest.raises(MultiTrainSplitError):
        model.split(binary_data, "target", auto_cat_encode=True, test_size=2.0)


def test_classifier_text_fit_succeeds_and_rejects_multiple_columns():
    data = pd.DataFrame(
        {
            "text": ["red apple", "blue sky", "red berry", "blue sea"] * 3,
            "target": [0, 1, 0, 1] * 3,
        }
    )
    config = {
        "ngram_range": (1, 1),
        "encoding": "utf-8",
        "max_features": 50,
        "analyzer": "word",
    }
    classifier = MultiClassifier(
        custom_models=["LogisticRegression"], text=True, max_iter=100
    )
    results = classifier.fit(
        classifier.split(data, "target", test_size=0.25),
        vectorizer="tfidf",
        pipeline_dict=config,
    )
    assert "LogisticRegression" in results.index

    two_features = data.assign(other=data["text"])
    with pytest.raises(MultiTrainTextError):
        classifier.fit(
            classifier.split(two_features, "target"),
            vectorizer="tfidf",
            pipeline_dict=config,
        )


def test_classifier_text_fit_handles_numpy_and_missing_options():
    classifier = MultiClassifier(
        custom_models=["LogisticRegression"], text=True, max_iter=100
    )
    X_train = np.array([["red"], ["blue"], ["red sky"], ["blue sea"]])
    X_test = np.array([["red apple"], ["blue water"]])
    split = (X_train, X_test, np.array([0, 1, 0, 1]), np.array([0, 1]))
    config = {
        "ngram_range": (1, 1),
        "encoding": "utf-8",
        "max_features": 50,
        "analyzer": "word",
    }
    assert "LogisticRegression" in classifier.fit(
        split, vectorizer="count", pipeline_dict=config
    ).index
    with pytest.raises(MultiTrainTextError):
        classifier.fit(split)
    with pytest.raises(MultiTrainTextError):
        classifier.fit(split, pipeline_dict=config)
    with pytest.raises(MultiTrainTextError):
        classifier.fit(
            (
                np.column_stack([X_train, X_train]),
                np.column_stack([X_test, X_test]),
                split[2],
                split[3],
            ),
            vectorizer="count",
            pipeline_dict=config,
        )


class PredictingResult:
    classes_ = np.array([0, 1])

    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, X):
        if self.fail:
            raise RuntimeError("prediction failed")
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))


def test_classifier_fit_covers_gpu_conversion_and_failed_train_prediction(monkeypatch):
    X_train = pd.DataFrame({"x": [0, 1, 2, 3]})
    X_test = pd.DataFrame({"x": [4, 5]})
    y_train = pd.Series([0, 0, 1, 1])
    y_test = pd.Series([0, 1])
    classifier = MultiClassifier(custom_models=["LogisticRegression"])
    classifier.use_gpu = True

    monkeypatch.setattr(classification_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        classification_module,
        "_prep_model_names_list",
        lambda *a, **k: (
            ["fake"],
            [object()],
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    )
    monkeypatch.setattr(
        classification_module,
        "run_models",
        lambda *a, **k: [
            SimpleNamespace(
                name="fake",
                test_prediction=np.array([0, 1]),
                train_prediction=np.full(4, np.nan),
                test_roc_auc=np.nan,
                train_roc_auc=np.nan,
                elapsed="1ms",
            )
        ],
    )
    results = classifier.fit((X_train, X_test, y_train, y_test), show_train_score=True)
    assert np.isnan(results.loc["fake", "accuracy_train"])


def test_regressor_fit_handles_failed_train_prediction(monkeypatch):
    X_train = pd.DataFrame({"x": [0, 1, 2, 3]})
    X_test = pd.DataFrame({"x": [4, 5]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    y_test = pd.Series([4.0, 5.0])
    regressor = MultiRegressor(custom_models=["LinearRegression"])
    monkeypatch.setattr(
        regression_module,
        "_prep_model_names_list",
        lambda *a, **k: (
            ["fake"],
            [object()],
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    )
    monkeypatch.setattr(
        regression_module,
        "run_models",
        lambda *a, **k: [
            SimpleNamespace(
                name="fake",
                test_prediction=np.array([4.0, 5.0]),
                train_prediction=np.full(4, np.nan),
                elapsed="1ms",
            )
        ],
    )
    results = regressor.fit((X_train, X_test, y_train, y_test), show_train_score=True)
    assert np.isnan(results.loc["fake", "mean_squared_error_train"])


def test_classifier_fit_covers_sort_custom_metric_and_vectorizer_validation(binary_data):
    classifier = MultiClassifier(custom_models=["LogisticRegression"])
    split = classifier.split(binary_data, "target", auto_cat_encode=True, test_size=0.4)
    assert classifier.fit(split, sort="accuracy").index[0] == "LogisticRegression"
    custom = classifier.fit(split, custom_metric="jaccard_score")
    assert "jaccard_score" in custom.columns
    without_train = classifier.fit(split, show_train_score=False)
    assert not any(column.endswith("_train") for column in without_train.columns)
    with_train = classifier.fit(split, show_train_score=True)
    assert not np.isnan(with_train.loc["LogisticRegression", "accuracy_train"])

    text_classifier = MultiClassifier(custom_models=["LogisticRegression"], text=True)
    text_split = text_classifier.split(binary_data[["category", "target"]], "target")
    with pytest.raises(MultiTrainTextError):
        text_classifier.fit(text_split, pipeline_dict={"present": True})


def test_regressor_fit_covers_sort_and_custom_metric(regression_data):
    regressor = MultiRegressor(custom_models=["LinearRegression"])
    split = regressor.split(regression_data, "target", auto_cat_encode=True, test_size=0.4)
    sorted_results = regressor.fit(split, sort="mean_squared_error")
    assert sorted_results.index[0] == "LinearRegression"
    custom = regressor.fit(split, custom_metric="max_error")
    assert "max_error" in custom.columns


def test_format_time_property_over_wide_nonnegative_range():
    values = np.concatenate(
        [
            np.linspace(0, 0.0009, 20),
            np.linspace(0.001, 0.999, 20),
            np.linspace(1, 3599, 20),
            np.linspace(3600, 100000, 20),
        ]
    )
    for seconds in values:
        formatted = utils._format_time(float(seconds))
        assert formatted
        assert formatted.endswith(("us", "ms", "s"))
        assert "Â" not in formatted and "µ" not in formatted


def test_prepare_train_test_property_preserves_inputs_and_schema():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        train = pd.DataFrame(
            {
                "category": rng.choice(["a", "b", "c"], size=20),
                "number": rng.normal(size=20),
                "target": rng.integers(0, 2, size=20),
            }
        )
        test = pd.DataFrame(
            {
                "category": rng.choice(["a", "b", "c", "unknown"], size=8),
                "number": rng.normal(size=8),
                "target": rng.integers(0, 2, size=8),
            }
        )
        original_train = train.copy(deep=True)
        original_test = test.copy(deep=True)

        encoded_train, encoded_test = utils._prepare_train_test(
            train, test, auto_cat_encode=True
        )

        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(test, original_test)
        assert list(encoded_train.columns) == list(encoded_test.columns)
        assert encoded_train["category"].min() >= 0
        assert encoded_test["category"].min() >= 0
        assert encoded_test["category"].max() <= encoded_train["category"].nunique()


def test_display_table_property_matches_python_sorting():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        results = {
            f"model_{index}": {
                "accuracy": float(rng.random()),
                "mean_squared_error": float(rng.random()),
            }
            for index in range(15)
        }
        classification = utils._display_table(
            results, sort="accuracy", task="classification"
        )
        regression = utils._display_table(
            results, sort="mean_squared_error", task="regression"
        )

        expected_high = sorted(results, key=lambda name: results[name]["accuracy"], reverse=True)
        expected_low = sorted(results, key=lambda name: results[name]["mean_squared_error"])
        assert list(classification.index) == expected_high
        assert list(regression.index) == expected_low


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
def test_split_property_is_a_complete_disjoint_partition(model_class):
    data = pd.DataFrame(
        {"feature": np.arange(101), "target": np.arange(101) % 2}
    )
    for seed in range(20):
        model = model_class()
        X_train, X_test, y_train, y_test = model.split(
            data, "target", random_state=seed, test_size=0.23
        )
        assert set(X_train.index).isdisjoint(X_test.index)
        assert set(X_train.index) | set(X_test.index) == set(data.index)
        assert X_train.index.equals(y_train.index)
        assert X_test.index.equals(y_test.index)
