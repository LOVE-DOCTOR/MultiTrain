from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import MultiTrainPCAError, MultiTrainTextError, MultiTrainTypeError
from MultiTrain.regression.regression_models import MultiRegressor
from MultiTrain.utils import execution
from MultiTrain.utils.utils import _models_classifier, _models_regressor


TEXT_OPTIONS = {
    "ngram_range": (1, 2),
    "encoding": "utf-8",
    "max_features": 100,
    "analyzer": "word",
}


class RowCountRegressor:
    def fit(self, X, y):
        self.rows_seen_ = len(X)
        return self

    def predict(self, X):
        return np.full(len(X), self.rows_seen_, dtype=float)


@pytest.mark.parametrize(
    "workers,threads,count,expected",
    [
        (None, 1, 1, 1),
        (1, 1, 10, 1),
        (2, 1, 10, 2),
        (-1, 1, 2, 2),
        (None, -1, 10, 1),
        (None, 1, 0, 1),
    ],
)
def test_worker_resolution_covers_boundaries(workers, threads, count, expected):
    assert execution.resolve_model_workers(workers, threads, count) == expected


@pytest.mark.parametrize("workers", [0, -2, 1.5, True, "2"])
def test_worker_resolution_rejects_invalid_values(workers):
    with pytest.raises(MultiTrainTypeError):
        execution.resolve_model_workers(workers, 1, 2)


def test_negative_estimator_threads_cannot_be_combined_with_model_processes():
    with pytest.raises(MultiTrainTypeError, match="oversubscription"):
        execution.resolve_model_workers(2, -1, 2)


@pytest.mark.parametrize("n_components,expected_columns", [(1, 1), (2, 2)])
def test_shared_pca_honors_component_count(n_components, expected_columns):
    train = np.arange(60, dtype=float).reshape(20, 3)
    test = np.arange(12, dtype=float).reshape(4, 3)
    transformed_train, transformed_test = execution.prepare_tabular_features(
        train,
        test,
        scaler=execution.QuantileTransformer(),
        n_components=n_components,
    )
    assert transformed_train.shape == (20, expected_columns)
    assert transformed_test.shape == (4, expected_columns)


@pytest.mark.parametrize("n_components", [0, -1, 4, 1.0, -0.1, "2", True])
def test_shared_pca_rejects_invalid_component_counts(n_components):
    data = np.ones((3, 3))
    with pytest.raises(MultiTrainPCAError):
        execution.prepare_tabular_features(
            data,
            data,
            scaler=execution.QuantileTransformer(),
            n_components=n_components,
        )


def test_text_is_vectorized_once_for_multiple_models(monkeypatch):
    calls = {"fit_transform": 0, "transform": 0}
    original = execution.CountVectorizer

    class CountingVectorizer(original):
        def fit_transform(self, raw_documents, y=None):
            calls["fit_transform"] += 1
            return super().fit_transform(raw_documents, y)

        def transform(self, raw_documents):
            calls["transform"] += 1
            return super().transform(raw_documents)

    monkeypatch.setattr(execution, "CountVectorizer", CountingVectorizer)
    train = ["good day", "bad day", "good result", "bad result"]
    test = ["good", "bad"]
    prepared = execution.prepare_text_features(
        "count",
        TEXT_OPTIONS,
        train,
        test,
        [LogisticRegression(), GaussianNB()],
        max_dense_bytes=1024 ** 2,
    )

    assert calls == {"fit_transform": 1, "transform": 1}
    assert prepared[0].shape[0] == len(train)
    assert prepared[2] is not None
    assert prepared[4] == {"GaussianNB"}


def test_sparse_support_falls_back_for_scikit_learn_1_3(monkeypatch):
    monkeypatch.setattr(execution, "_get_estimator_tags", None)
    assert execution._supports_sparse_input(LogisticRegression())
    assert not execution._supports_sparse_input(GaussianNB())


def test_legacy_sparse_fallback_matches_current_estimator_tags(monkeypatch):
    current_get_tags = execution._get_estimator_tags
    if current_get_tags is None:
        pytest.skip("The public estimator tag API starts with scikit-learn 1.6")
    models = [
        *_models_classifier(n_jobs=1, max_iter=2).values(),
        *_models_regressor(n_jobs=1, max_iter=2).values(),
    ]
    expected = {
        model.__class__.__name__: current_get_tags(model).input_tags.sparse
        for model in models
    }
    monkeypatch.setattr(execution, "_get_estimator_tags", None)
    actual = {
        model.__class__.__name__: execution._supports_sparse_input(model)
        for model in models
    }
    assert actual == expected


def test_dense_text_allocation_stops_before_exceeding_the_limit():
    with pytest.raises(MultiTrainTextError, match="GaussianNB.*exceeds"):
        execution.prepare_text_features(
            "count",
            TEXT_OPTIONS,
            ["one two three", "four five six"],
            ["seven eight"],
            [GaussianNB()],
            max_dense_bytes=1,
        )


def test_each_parallel_model_receives_the_complete_training_dataset():
    train = np.arange(36, dtype=float).reshape(12, 3)
    test = np.arange(12, dtype=float).reshape(4, 3)
    results = execution.run_models(
        ["first", "second"],
        [RowCountRegressor(), RowCountRegressor()],
        train,
        np.arange(12, dtype=float),
        test,
        np.arange(4, dtype=float),
        show_train_score=True,
        task="regression",
        model_workers=2,
        model_threads=1,
    )

    assert [result.name for result in results] == ["first", "second"]
    assert all(np.all(result.test_prediction == len(train)) for result in results)
    assert all(len(result.train_prediction) == len(train) for result in results)


def test_gpu_models_are_executed_sequentially(monkeypatch):
    calls = []

    def record_fit(name, *_args, **_kwargs):
        calls.append(name)
        return SimpleNamespace(name=name, error=None)

    monkeypatch.setattr(execution, "_fit_model", record_fit)
    results = execution.run_models(
        ["XGBClassifier", "CatBoostClassifier"],
        [object(), object()],
        np.ones((4, 1)),
        np.array([0, 1, 0, 1]),
        np.ones((2, 1)),
        np.array([0, 1]),
        show_train_score=False,
        task="classification",
        model_workers=2,
        model_threads=1,
        use_gpu=True,
    )
    assert calls == ["XGBClassifier", "CatBoostClassifier"]
    assert [result.name for result in results] == calls


def test_parallel_and_sequential_classification_have_equivalent_scores():
    frame = pd.DataFrame(
        {
            "x1": np.arange(80),
            "x2": np.arange(80) % 7,
            "target": np.arange(80) % 2,
        }
    )
    model_names = ["LogisticRegression", "DecisionTreeClassifier"]
    sequential = MultiClassifier(
        custom_models=model_names, model_workers=1, random_state=9
    )
    parallel = MultiClassifier(
        custom_models=model_names, model_workers=2, random_state=9
    )
    split = sequential.split(frame, "target", random_state=9)
    sequential_result = sequential.fit(split)
    parallel_result = parallel.fit(split)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    np.testing.assert_allclose(
        sequential_result.loc[model_names, metrics].astype(float),
        parallel_result.loc[model_names, metrics].astype(float),
        equal_nan=True,
    )


def test_public_fit_supports_shared_pca_and_rejects_unrelated_components():
    X_train = np.arange(60, dtype=float).reshape(20, 3)
    X_test = np.arange(15, dtype=float).reshape(5, 3)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(5, dtype=float)
    model = MultiRegressor(custom_models=["LinearRegression"])
    result = model.fit(
        (X_train, X_test, y_train, y_test),
        pca="StandardScaler",
        n_components=2,
    )
    assert np.isfinite(result.loc["LinearRegression", "mean_squared_error"])
    with pytest.raises(MultiTrainPCAError):
        model.fit((X_train, X_test, y_train, y_test), n_components=2)


@pytest.mark.parametrize("model_class", [MultiClassifier, MultiRegressor])
@pytest.mark.parametrize("model_workers", [0, -2, True, 1.5, "2"])
def test_public_constructors_reject_invalid_worker_counts(model_class, model_workers):
    with pytest.raises(MultiTrainTypeError):
        model_class(model_workers=model_workers)


@pytest.mark.parametrize("max_dense_bytes", [0, -1, True, 1.5, "1024"])
def test_classifier_fit_rejects_invalid_dense_limits(max_dense_bytes):
    split = (
        np.ones((4, 1)),
        np.ones((2, 1)),
        np.array([0, 1, 0, 1]),
        np.array([0, 1]),
    )
    with pytest.raises(MultiTrainTypeError):
        MultiClassifier(custom_models=["LogisticRegression"]).fit(
            split, max_dense_bytes=max_dense_bytes
        )
