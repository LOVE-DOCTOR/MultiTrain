"""Interaction and boundary tests for MultiTrain's complete public surface.

Finite option sets are exercised as Cartesian products.  Unbounded domains such
as numbers, strings, and datasets are represented by type, boundary, and
equivalence partitions instead of pretending they can literally be enumerated.
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

import MultiTrain.classification.classification_models as classification_module
import MultiTrain.regression.regression_models as regression_module
from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import (
    MultiTrainPCAError,
    MultiTrainSplitError,
    MultiTrainTypeError,
)
from MultiTrain.regression.regression_models import MultiRegressor
from MultiTrain.utils import utils


CLASSIFIER_NAMES = list(utils._models_classifier(max_iter=20))
REGRESSOR_NAMES = list(utils._models_regressor(max_iter=20))


class QuietRange:
    """Small tqdm-compatible iterator so exhaustive tests remain readable."""

    def __init__(self, length):
        self._range = range(length)

    def __iter__(self):
        return iter(self._range)

    def set_postfix_str(self, _value):
        return None


@pytest.fixture(autouse=True)
def quiet_progress(monkeypatch):
    for module in (classification_module, regression_module):
        monkeypatch.setattr(module, "trange", lambda length, **_kwargs: QuietRange(length))
        monkeypatch.setattr(module, "tqdm", lambda iterable, **_kwargs: iterable)


def _classification_frame(rows=80):
    rng = np.random.default_rng(711)
    values = rng.uniform(0.1, 4.0, size=(rows, 5))
    frame = pd.DataFrame(values, columns=[f"x{index}" for index in range(5)])
    frame["target"] = (values[:, 0] + values[:, 1] > np.median(values[:, 0] + values[:, 1])).astype(int)
    return frame


def _regression_frame(rows=80):
    rng = np.random.default_rng(912)
    values = rng.uniform(0.2, 3.0, size=(rows, 5))
    frame = pd.DataFrame(values, columns=[f"x{index}" for index in range(5)])
    frame["target"] = 1.0 + values @ np.array([1.2, 0.7, 2.1, 0.4, 1.5])
    return frame


@pytest.mark.parametrize("factory", [utils._models_classifier, utils._models_regressor])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"random_state": True},
        {"random_state": 1.5},
        {"random_state": "7"},
        {"n_jobs": True},
        {"n_jobs": 0},
        {"n_jobs": 1.5},
        {"n_jobs": "1"},
        {"max_iter": True},
        {"max_iter": 0},
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"max_iter": "10"},
        {"use_gpu": 1},
        {"use_gpu": None},
        {"device": ""},
        {"device": 0},
    ],
)
def test_every_model_factory_rejects_invalid_equivalence_classes(factory, kwargs):
    with pytest.raises(MultiTrainTypeError):
        factory(**kwargs)


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_jobs": True},
        {"n_jobs": 0},
        {"random_state": False},
        {"max_iter": True},
        {"max_iter": 0},
        {"max_iter": -1},
        {"use_gpu": 1},
        {"device": ""},
        {"custom_models": "model"},
        {"custom_models": [1]},
    ],
)
def test_constructors_reject_invalid_value_partitions(model_class, kwargs):
    with pytest.raises(MultiTrainTypeError):
        model_class(**kwargs)


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
@pytest.mark.parametrize("source", ["dataframe", "csv"])
@pytest.mark.parametrize("encoding", ["numeric", "automatic", "label", "onehot"])
@pytest.mark.parametrize("missing", ["none", "ffill", "bfill", "interpolate"])
@pytest.mark.parametrize("drop_column", [False, True])
@pytest.mark.parametrize("test_size", [0.2, 0.4])
def test_split_cartesian_product(
    tmp_path,
    model_class,
    source,
    encoding,
    missing,
    drop_column,
    test_size,
):
    rows = 40
    category = np.tile(["a", "b", "c", "d"], rows // 4)
    frame = pd.DataFrame(
        {
            "number": np.arange(rows, dtype=float),
            "category": category,
            "drop_me": np.arange(rows) * 3,
            "target": np.arange(rows) % 2 if model_class is MultiClassifier else np.arange(rows) * 1.5 + 1,
        }
    )
    if encoding == "numeric":
        frame["category"] = pd.factorize(frame["category"])[0]
        auto, manual = False, None
    elif encoding == "automatic":
        auto, manual = True, None
    elif encoding == "label":
        auto, manual = False, {"label": ["category"]}
    else:
        auto, manual = False, {"onehot": ["category"]}

    fix_nan = False
    if missing != "none":
        frame.loc[[0, rows // 2, rows - 1], "number"] = np.nan
        fix_nan = {"number": missing}

    data = frame
    if source == "csv":
        path = tmp_path / "matrix.csv"
        frame.to_csv(path, index=False)
        data = str(path)

    original = frame.copy(deep=True)
    splits = model_class(n_jobs=1).split(
        data,
        "target",
        random_state=13,
        test_size=test_size,
        auto_cat_encode=auto,
        manual_encode=manual,
        fix_nan_custom=fix_nan,
        drop=["drop_me"] if drop_column else None,
    )
    X_train, X_test, y_train, y_test = splits
    assert len(X_train) + len(X_test) == rows
    assert len(y_train) + len(y_test) == rows
    assert set(X_train.index).isdisjoint(X_test.index)
    assert list(X_train.columns) == list(X_test.columns)
    assert not pd.DataFrame(X_train).isna().any().any()
    assert not pd.DataFrame(X_test).isna().any().any()
    assert ("drop_me" not in X_train.columns) is drop_column
    assert frame.equals(original)
    assert not any(dtype == object for dtype in X_train.dtypes)


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
@pytest.mark.parametrize(
    "parameter,value,error",
    [
        ("target", None, MultiTrainTypeError),
        ("target", "", MultiTrainTypeError),
        ("target", 1, MultiTrainTypeError),
        ("random_state", True, MultiTrainTypeError),
        ("random_state", 1.5, MultiTrainTypeError),
        ("random_state", "1", MultiTrainTypeError),
        ("test_size", True, MultiTrainTypeError),
        ("test_size", "0.2", MultiTrainTypeError),
        ("test_size", 0, MultiTrainSplitError),
        ("test_size", -0.1, MultiTrainSplitError),
        ("test_size", 1, MultiTrainSplitError),
        ("test_size", 1.1, MultiTrainSplitError),
        ("auto_cat_encode", 1, MultiTrainTypeError),
        ("auto_cat_encode", None, MultiTrainTypeError),
    ],
)
def test_split_rejects_every_parameter_boundary(model_class, parameter, value, error):
    frame = pd.DataFrame({"x": range(10), "target": range(10)})
    kwargs = {"target": "target", parameter: value}
    with pytest.raises(error):
        model_class().split(frame, **kwargs)


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
def test_split_translates_underlying_split_failures(model_class):
    one_row = pd.DataFrame({"x": [1], "target": [0]})
    with pytest.raises(MultiTrainSplitError, match="Unable to split"):
        model_class().split(one_row, "target")


CLASSIFIER_FIT_CASES = list(
    product(
        [False, *classification_module.SUPPORTED_SCALERS],
        [False, True],
        [False, True],
        ["plain", "sort", "best"],
    )
)


@pytest.mark.parametrize("pca,show_train,imbalanced,result_mode", CLASSIFIER_FIT_CASES)
def test_classifier_fit_cartesian_options(pca, show_train, imbalanced, result_mode):
    model = MultiClassifier(n_jobs=1, custom_models=["LogisticRegression"], max_iter=100)
    splits = model.split(_classification_frame(), "target", test_size=0.25)
    kwargs = {
        "pca": pca,
        "show_train_score": show_train,
        "imbalanced": imbalanced,
    }
    if result_mode == "sort":
        kwargs["sort"] = "accuracy"
    elif result_mode == "best":
        kwargs["return_best_model"] = "accuracy"
    result = model.fit(splits, **kwargs)
    assert list(result.index) == ["LogisticRegression"]
    assert np.isfinite(result.loc["LogisticRegression", "accuracy"])
    assert any(column.endswith("_train") for column in result.columns) is show_train


REGRESSOR_FIT_CASES = list(
    product(
        [False, *regression_module.SUPPORTED_SCALERS],
        [False, True],
        ["plain", "sort", "best"],
    )
)


@pytest.mark.parametrize("pca,show_train,result_mode", REGRESSOR_FIT_CASES)
def test_regressor_fit_cartesian_options(pca, show_train, result_mode):
    model = MultiRegressor(n_jobs=1, custom_models=["LinearRegression"], max_iter=100)
    splits = model.split(_regression_frame(), "target", test_size=0.25)
    kwargs = {"pca": pca, "show_train_score": show_train}
    if result_mode == "sort":
        kwargs["sort"] = "mean_squared_error"
    elif result_mode == "best":
        kwargs["return_best_model"] = "mean_squared_error"
    result = model.fit(splits, **kwargs)
    assert list(result.index) == ["LinearRegression"]
    assert np.isfinite(result.loc["LinearRegression", "mean_squared_error"])
    assert any(column.endswith("_train") for column in result.columns) is show_train


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"custom_metric": 1}, MultiTrainTypeError),
        ({"show_train_score": 1}, MultiTrainTypeError),
        ({"sort": 1}, MultiTrainTypeError),
        ({"pca": None}, MultiTrainPCAError),
        ({"pca": True}, MultiTrainPCAError),
        ({"return_best_model": 1}, MultiTrainTypeError),
    ],
)
def test_fit_rejects_common_invalid_value_partitions(model_class, kwargs, error):
    with pytest.raises(error):
        model_class(custom_models=[CLASSIFIER_NAMES[0] if model_class is MultiClassifier else REGRESSOR_NAMES[0]]).fit(
            (np.ones((4, 2)), np.ones((2, 2)), np.array([0, 1, 0, 1]), np.array([0, 1])),
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"imbalanced": 1},
        {"imbalanced": None},
        {"vectorizer": 1},
        {"vectorizer": False},
        {"pipeline_dict": []},
        {"pipeline_dict": "pipeline"},
    ],
)
def test_classifier_fit_rejects_classification_specific_invalid_types(kwargs):
    with pytest.raises(MultiTrainTypeError):
        MultiClassifier(custom_models=["LogisticRegression"]).fit(
            (np.ones((4, 2)), np.ones((2, 2)), np.array([0, 1, 0, 1]), np.array([0, 1])),
            **kwargs,
        )


@pytest.mark.parametrize("vectorizer", ["count", "tfidf"])
@pytest.mark.parametrize("analyzer", ["word", "char"])
@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2)])
@pytest.mark.parametrize("show_train", [False, True])
@pytest.mark.parametrize("model_name", ["LogisticRegression", "GaussianNB"])
def test_text_fit_cartesian_pipeline_options(
    vectorizer, analyzer, ngram_range, show_train, model_name
):
    positive = ["bright sunny excellent", "great happy wonderful", "love pleasant success"]
    negative = ["dark rainy awful", "bad sad terrible", "hate unpleasant failure"]
    texts = (positive + negative) * 5
    labels = ([1] * 3 + [0] * 3) * 5
    frame = pd.DataFrame({"text": texts, "target": labels})
    model = MultiClassifier(
        n_jobs=1,
        custom_models=[model_name],
        max_iter=100,
        text=True,
    )
    splits = model.split(frame, "target", test_size=0.2)
    result = model.fit(
        splits,
        vectorizer=vectorizer,
        pipeline_dict={
            "ngram_range": ngram_range,
            "encoding": "utf-8",
            "max_features": 100,
            "analyzer": analyzer,
        },
        show_train_score=show_train,
    )
    assert list(result.index) == [model_name]
    assert np.isfinite(result.loc[model_name, "accuracy"])


def test_text_fit_accepts_one_dimensional_series_splits():
    X_train = pd.Series(["good bright", "bad dark", "great day", "awful night"] * 3)
    y_train = pd.Series([1, 0, 1, 0] * 3)
    X_test = pd.Series(["good day", "bad night", "bright great", "dark awful"])
    y_test = pd.Series([1, 0, 1, 0])
    result = MultiClassifier(
        custom_models=["LogisticRegression"], text=True, max_iter=100
    ).fit(
        (X_train, X_test, y_train, y_test),
        vectorizer="count",
        pipeline_dict={
            "ngram_range": (1, 1),
            "encoding": "utf-8",
            "max_features": 50,
            "analyzer": "word",
        },
    )
    assert np.isfinite(result.loc["LogisticRegression", "accuracy"])


def test_classifier_fit_runs_every_bundled_estimator_without_silent_failure():
    model = MultiClassifier(n_jobs=1, max_iter=20)
    result = model.fit(model.split(_classification_frame(100), "target", test_size=0.2))
    assert set(result.index) == set(CLASSIFIER_NAMES)
    failed = result.index[~np.isfinite(pd.to_numeric(result["accuracy"], errors="coerce"))].tolist()
    assert failed == [], f"Classifiers that failed to fit or predict: {failed}"


def test_regressor_fit_runs_every_bundled_estimator_without_silent_failure():
    model = MultiRegressor(n_jobs=1, max_iter=20)
    result = model.fit(model.split(_regression_frame(100), "target", test_size=0.2))
    assert set(result.index) == set(REGRESSOR_NAMES)
    failed = result.index[
        ~np.isfinite(pd.to_numeric(result["mean_squared_error"], errors="coerce"))
    ].tolist()
    assert failed == [], f"Regressors that failed to fit or predict: {failed}"
