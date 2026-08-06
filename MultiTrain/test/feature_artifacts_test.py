"""Tests for retained fit artifacts and configurable estimator selection."""

import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import MultiTrainModelError, MultiTrainTypeError
from MultiTrain.regression.regression_models import MultiRegressor
from MultiTrain.utils.utils import _check_custom_models


class DiagnosticClassifier(BaseEstimator):
    """Small estimator that can emit a warning or fail during fitting."""

    def __init__(self, warn=False, fail=False):
        self.warn = warn
        self.fail = fail

    def fit(self, X, y):
        if self.warn:
            warnings.warn("diagnostic warning", UserWarning)
        if self.fail:
            raise ValueError("diagnostic failure")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def classification_splits():
    X_train = np.arange(32, dtype=float).reshape(16, 2)
    X_test = np.arange(32, 48, dtype=float).reshape(8, 2)
    y_train = np.array([0, 1] * 8)
    y_test = np.array([0, 1] * 4)
    return X_train, X_test, y_train, y_test


def regression_splits():
    X_train = np.arange(24, dtype=float).reshape(12, 2)
    X_test = np.arange(24, 36, dtype=float).reshape(6, 2)
    y_train = np.arange(12, dtype=float)
    y_test = np.arange(12, 18, dtype=float)
    return X_train, X_test, y_train, y_test


def test_classifier_retains_models_predictions_probabilities_and_results():
    classifier = MultiClassifier(
        custom_models=["DecisionTreeClassifier"],
        model_params={"DecisionTreeClassifier": {"max_depth": 1}},
    )

    results = classifier.fit(classification_splits(), show_train_score=True)

    assert classifier.results_ is results
    assert classifier.models_["DecisionTreeClassifier"].max_depth == 1
    assert classifier.predictions_["test"]["DecisionTreeClassifier"].shape == (8,)
    assert classifier.predictions_["train"]["DecisionTreeClassifier"].shape == (16,)
    assert classifier.probabilities_["test"]["DecisionTreeClassifier"].shape == (8, 2)
    assert classifier.probabilities_["train"]["DecisionTreeClassifier"].shape == (16, 2)
    assert list(classifier.warnings_.columns) == ["Model", "Category", "Message"]
    assert list(classifier.failures_.columns) == [
        "Model",
        "Stage",
        "Exception",
        "Message",
    ]


def test_regressor_retains_predictions_without_inventing_probabilities():
    regressor = MultiRegressor(custom_models=["LinearRegression"])

    results = regressor.fit(regression_splits())

    assert regressor.results_ is results
    assert isinstance(regressor.models_["LinearRegression"], LinearRegression)
    assert regressor.predictions_["test"]["LinearRegression"].shape == (6,)
    assert regressor.predictions_["train"] == {}
    assert regressor.probabilities_ == {"test": {}, "train": {}}


def test_named_estimator_dictionary_is_cloned_and_keeps_the_custom_name():
    source_estimator = DecisionTreeClassifier(random_state=7)
    classifier = MultiClassifier(
        custom_models={"small tree": source_estimator},
        model_params={"small tree": {"max_depth": 2}},
    )

    results = classifier.fit(classification_splits())

    assert list(results.index) == ["small tree"]
    assert classifier.models_["small tree"] is not source_estimator
    assert classifier.models_["small tree"].max_depth == 2
    assert hasattr(classifier.models_["small tree"], "tree_")
    assert not hasattr(source_estimator, "tree_")


def test_custom_estimators_survive_process_based_model_parallelism():
    classifier = MultiClassifier(
        custom_models={
            "first tree": DecisionTreeClassifier(max_depth=1, random_state=3),
            "second tree": DecisionTreeClassifier(max_depth=2, random_state=3),
        },
        model_workers=2,
    )

    results = classifier.fit(classification_splits())

    assert list(results.index) == ["first tree", "second tree"]
    assert list(classifier.models_) == ["first tree", "second tree"]


def test_warnings_and_failures_are_attributed_to_the_responsible_model():
    classifier = MultiClassifier(
        custom_models={
            "warning model": DiagnosticClassifier(warn=True),
            "failing model": DiagnosticClassifier(fail=True),
        },
        model_workers=1,
    )

    with pytest.warns(UserWarning, match="diagnostic warning"):
        results = classifier.fit(classification_splits())

    assert "warning model" in classifier.models_
    assert "failing model" not in classifier.models_
    assert classifier.warnings_.to_dict("records") == [
        {
            "Model": "warning model",
            "Category": "UserWarning",
            "Message": "diagnostic warning",
        }
    ]
    assert classifier.failures_.to_dict("records") == [
        {
            "Model": "failing model",
            "Stage": "fit",
            "Exception": "ValueError",
            "Message": "diagnostic failure",
        }
    ]
    assert results.loc["failing model"].drop("Time").isna().all()


def test_artifacts_are_cleared_before_a_new_invalid_fit_attempt():
    classifier = MultiClassifier(custom_models=["DecisionTreeClassifier"])
    classifier.fit(classification_splits())

    with pytest.raises(MultiTrainTypeError):
        classifier.fit(classification_splits(), show_train_score="yes")

    assert classifier.results_ is None
    assert classifier.models_ == {}
    assert classifier.predictions_ == {"test": {}, "train": {}}


@pytest.mark.parametrize(
    "custom_models,model_params,error_type",
    [
        ({}, None, MultiTrainModelError),
        ({"broken": object()}, None, MultiTrainModelError),
        ({1: DecisionTreeClassifier()}, None, MultiTrainTypeError),
        (
            {"tree": DecisionTreeClassifier()},
            {"missing": {"max_depth": 1}},
            MultiTrainModelError,
        ),
        (
            {"tree": DecisionTreeClassifier()},
            {"tree": {"not_a_parameter": 1}},
            MultiTrainModelError,
        ),
    ],
)
def test_custom_estimator_configuration_rejects_invalid_combinations(
    custom_models,
    model_params,
    error_type,
):
    with pytest.raises(error_type):
        _check_custom_models(
            custom_models,
            {"DecisionTreeClassifier": DecisionTreeClassifier()},
            model_params=model_params,
        )
