from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import (
    MultiTrainDatasetTypeError,
    MultiTrainDatasetValueError,
    MultiTrainNaNError,
    MultiTrainPCAError,
    MultiTrainSplitError,
    MultiTrainTextError,
    MultiTrainTypeError,
)
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


def test_scale_sensitive_estimators_fit_the_scaler_on_training_rows_only():
    train = np.array([[0.0], [2.0], [4.0], [6.0]])
    estimator = execution._prepare_training_estimator(
        "SVC", SVC(), train, "classification"
    )
    estimator.fit(train, np.array([0, 0, 1, 1]))

    assert isinstance(estimator, Pipeline)
    np.testing.assert_allclose(
        estimator.named_steps["standardscaler"].mean_,
        train.mean(axis=0),
    )


def test_sparse_sensitive_estimators_preserve_sparse_input():
    train = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    estimator = execution._prepare_training_estimator(
        "SVC", SVC(), train, "classification"
    )

    assert not estimator.named_steps["standardscaler"].with_mean


@pytest.mark.parametrize(
    "name",
    [
        "SGDRegressor",
        "PassiveAggressiveRegressor",
        "MLPRegressor",
        "LinearSVR",
        "SVR",
        "NuSVR",
    ],
)
def test_scale_sensitive_regressors_restore_original_target_units(name):
    model = _models_regressor(random_state=4, n_jobs=1, max_iter=300)[name]
    train = np.arange(60, dtype=float).reshape(20, 3)
    target = 1_000_000 + train[:, 0] * 50_000
    estimator = execution._prepare_training_estimator(
        name, model, train, "regression"
    )
    estimator.fit(train, target)
    predictions = estimator.predict(train[:3])
    scaled_predictions = estimator.regressor_.predict(train[:3])
    restored_predictions = estimator.transformer_.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()

    assert isinstance(estimator, TransformedTargetRegressor)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, restored_predictions)


def test_libsvm_models_do_not_use_the_shared_iteration_cap():
    classifiers = _models_classifier(random_state=4, n_jobs=1, max_iter=3)
    regressors = _models_regressor(random_state=4, n_jobs=1, max_iter=3)

    assert classifiers["NuSVC"].max_iter == -1
    assert classifiers["SVC"].max_iter == -1
    assert regressors["NuSVR"].max_iter == -1
    assert regressors["SVR"].max_iter == -1


def test_iterative_regressors_use_stable_convergence_settings():
    classifiers = _models_classifier(random_state=4, n_jobs=1, max_iter=300)
    regressors = _models_regressor(random_state=4, n_jobs=1, max_iter=300)

    assert classifiers["MLPClassifier"].tol == 1e-3
    assert regressors["MLPRegressor"].tol == 1e-3
    assert regressors["LinearSVR"].dual is False
    assert regressors["LinearSVR"].loss == "squared_epsilon_insensitive"
    assert regressors["LinearSVR"].max_iter == 300


def test_example_datasets_fit_warning_prone_models_without_convergence_warnings():
    datasets = Path(__file__).parents[2] / "examples" / "datasets"
    classification_data = pd.read_csv(datasets / "train.csv")
    classifier = MultiClassifier(
        n_jobs=1,
        model_workers=1,
        max_iter=300,
        custom_models=["LinearSVC", "NuSVC", "SVC", "MLPClassifier"],
    )
    classification_split = classifier.split(
        classification_data,
        "Survived",
        auto_cat_encode=True,
        fix_nan_custom={"Age": "interpolate", "Embarked": "ffill"},
        drop=["PassengerId", "Name", "Ticket", "Cabin"],
    )

    regression_data = pd.read_csv(datasets / "Housing.csv")
    regressor = MultiRegressor(
        n_jobs=1,
        model_workers=1,
        max_iter=300,
        custom_models=["PoissonRegressor", "MLPRegressor", "LinearSVR"],
    )
    regression_split = regressor.split(
        regression_data,
        "price",
        test_size=0.3,
        auto_cat_encode=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        classification_results = classifier.fit(classification_split)
        regression_results = regressor.fit(regression_split)

    assert classification_results["accuracy"].notna().all()
    assert regression_results["mean_absolute_error"].notna().all()


def test_additional_datasets_fit_gradient_models_without_numerical_failures():
    datasets = Path(__file__).parents[2] / "examples" / "datasets"
    penguins = pd.read_csv(datasets / "penguins.csv")
    assert penguins.shape == (344, 7)
    assert set(penguins["species"]) == {"Adelie", "Chinstrap", "Gentoo"}

    classifier = MultiClassifier(
        n_jobs=1,
        model_workers=1,
        max_iter=1000,
        custom_models=[
            "LogisticRegression",
            "LogisticRegressionCV",
            "SGDClassifier",
            "PassiveAggressiveClassifier",
            "Perceptron",
        ],
    )
    classification_split = classifier.split(
        penguins,
        "species",
        test_size=0.25,
        auto_cat_encode=True,
        fix_nan_custom={
            "bill_length_mm": "interpolate",
            "bill_depth_mm": "interpolate",
            "flipper_length_mm": "interpolate",
            "body_mass_g": "interpolate",
            "sex": "ffill",
        },
    )

    wine = pd.read_csv(datasets / "winequality-red.csv", sep=";")
    assert wine.shape == (1599, 12)
    assert not wine.isna().values.any()

    regressor = MultiRegressor(
        n_jobs=1,
        model_workers=1,
        max_iter=1000,
        custom_models=["SGDRegressor", "PassiveAggressiveRegressor"],
    )
    regression_split = regressor.split(wine, "quality", test_size=0.25)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        classification_results = classifier.fit(classification_split)
        regression_results = regressor.fit(regression_split)

    assert classification_results["accuracy"].notna().all()
    assert regression_results["mean_absolute_error"].notna().all()
    assert regression_results["mean_absolute_error"].lt(2).all()


def test_penguins_metrics_match_independent_sklearn_calculations():
    penguins = pd.read_csv(
        Path(__file__).parents[2] / "examples" / "datasets" / "penguins.csv"
    )
    classifier = MultiClassifier(
        n_jobs=1,
        model_workers=1,
        random_state=42,
        custom_models=["LogisticRegression"],
    )
    split = classifier.split(
        penguins,
        "species",
        test_size=0.25,
        auto_cat_encode=True,
        fix_nan_custom={
            "bill_length_mm": "interpolate",
            "bill_depth_mm": "interpolate",
            "flipper_length_mm": "interpolate",
            "body_mass_g": "interpolate",
            "sex": "ffill",
        },
    )
    result = classifier.fit(split, show_train_score=True).loc[
        "LogisticRegression"
    ]

    X_train, X_test, y_train, y_test = split
    model = _models_classifier(random_state=42, n_jobs=1, max_iter=1000)[
        "LogisticRegression"
    ]
    estimator = execution._prepare_training_estimator(
        "LogisticRegression", model, X_train, "classification"
    )
    estimator.fit(X_train, y_train)
    prediction = estimator.predict(X_test)
    train_prediction = estimator.predict(X_train)
    expected = {
        "accuracy": accuracy_score(y_test, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "precision": precision_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, prediction, average="weighted", zero_division=0
        ),
        "f1": f1_score(y_test, prediction, average="weighted", zero_division=0),
    }

    for metric, expected_value in expected.items():
        assert float(result[metric]) == pytest.approx(expected_value)

    assert float(result["accuracy_train"]) == pytest.approx(
        accuracy_score(y_train, train_prediction)
    )
    assert float(result["roc_auc"]) == pytest.approx(
        roc_auc_score(
            y_test,
            estimator.predict_proba(X_test),
            multi_class="ovr",
            average="weighted",
        )
    )
    assert float(result["roc_auc_train"]) == pytest.approx(
        roc_auc_score(
            y_train,
            estimator.predict_proba(X_train),
            multi_class="ovr",
            average="weighted",
        )
    )


def test_wine_metrics_and_rmse_match_independent_sklearn_calculations():
    wine = pd.read_csv(
        Path(__file__).parents[2] / "examples" / "datasets" / "winequality-red.csv",
        sep=";",
    )
    regressor = MultiRegressor(
        n_jobs=1,
        model_workers=1,
        random_state=42,
        custom_models=["SGDRegressor"],
    )
    split = regressor.split(wine, "quality", test_size=0.25)
    result = regressor.fit(
        split,
        show_train_score=True,
        sort="root_mean_squared_error",
    ).loc["SGDRegressor"]

    X_train, X_test, y_train, y_test = split
    model = _models_regressor(random_state=42, n_jobs=1, max_iter=1000)[
        "SGDRegressor"
    ]
    estimator = execution._prepare_training_estimator(
        "SGDRegressor", model, X_train, "regression"
    )
    estimator.fit(X_train, y_train)
    test_prediction = estimator.predict(X_test)
    train_prediction = estimator.predict(X_train)
    expected_mse = mean_squared_error(y_test, test_prediction)

    assert float(result["mean_absolute_error"]) == pytest.approx(
        mean_absolute_error(y_test, test_prediction)
    )
    assert float(result["mean_squared_error"]) == pytest.approx(expected_mse)
    assert float(result["root_mean_squared_error"]) == pytest.approx(
        np.sqrt(expected_mse)
    )
    assert float(result["root_mean_squared_error_train"]) == pytest.approx(
        np.sqrt(mean_squared_error(y_train, train_prediction))
    )
    assert float(result["r2_score"]) == pytest.approx(
        r2_score(y_test, test_prediction)
    )


def test_split_rejects_dataset_states_that_would_fail_every_model():
    classifier = MultiClassifier(custom_models=["LogisticRegression"])

    infinite = pd.DataFrame({"feature": [0.0, 1.0, np.inf, 3.0], "target": [0, 1, 0, 1]})
    with pytest.raises(MultiTrainDatasetValueError, match="infinite"):
        classifier.split(infinite, "target")

    continuous = pd.DataFrame(
        {"feature": np.arange(20), "target": np.linspace(0.1, 2.0, 20)}
    )
    with pytest.raises(MultiTrainDatasetTypeError, match="discrete"):
        classifier.split(continuous, "target")

    duplicate_columns = pd.DataFrame(
        np.column_stack([np.arange(20), np.arange(20), [0, 1] * 10]),
        columns=["feature", "feature", "target"],
    )
    with pytest.raises(MultiTrainDatasetValueError, match="unique"):
        classifier.split(duplicate_columns, "target")

    with pytest.raises(MultiTrainDatasetValueError, match="feature column"):
        classifier.split(pd.DataFrame({"target": [0, 1] * 10}), "target")

    with pytest.raises(MultiTrainDatasetValueError, match="two classes"):
        classifier.split(
            pd.DataFrame({"feature": np.arange(20), "target": np.ones(20)}),
            "target",
        )


def test_split_rejects_missing_or_semantically_invalid_targets():
    missing_target = pd.DataFrame(
        {"feature": np.arange(20), "target": [0, 1] * 9 + [0, np.nan]}
    )
    with pytest.raises(MultiTrainNaNError, match="Target column"):
        MultiClassifier().split(
            missing_target,
            "target",
            fix_nan_custom={"target": "ffill"},
        )

    categorical_target = pd.DataFrame(
        {
            "feature": np.arange(30),
            "target": ["low", "medium", "high"] * 10,
        }
    )
    with pytest.raises(MultiTrainDatasetTypeError, match="must be numeric"):
        MultiRegressor().split(
            categorical_target,
            "target",
            auto_cat_encode=True,
        )


def test_fit_rejects_corrupt_manual_datasplits_before_training():
    X_train = np.arange(12, dtype=float).reshape(6, 2)
    X_test = np.arange(8, dtype=float).reshape(4, 2)
    y_train = np.array([0, 1, 0, 1, 0, 1])
    y_test = np.array([0, 1, 0, 1])
    classifier = MultiClassifier(custom_models=["LogisticRegression"])

    corrupt_features = X_train.copy()
    corrupt_features[0, 0] = np.inf
    with pytest.raises(MultiTrainDatasetValueError, match="X_train"):
        classifier.fit((corrupt_features, X_test, y_train, y_test))

    with pytest.raises(MultiTrainSplitError, match="row counts"):
        classifier.fit((X_train[:-1], X_test, y_train, y_test))

    with pytest.raises(MultiTrainDatasetTypeError, match="discrete"):
        classifier.fit(
            (X_train, X_test, np.linspace(0.1, 0.6, 6), np.linspace(0.7, 1.0, 4))
        )

    with pytest.raises(MultiTrainSplitError, match="absent from training"):
        classifier.fit((X_train, X_test, y_train, np.array([0, 1, 2, 2])))

    frame_train = pd.DataFrame(X_train, columns=["first", "second"])
    frame_test = pd.DataFrame(X_test, columns=["second", "first"])
    with pytest.raises(MultiTrainSplitError, match="same order"):
        classifier.fit(
            (
                frame_train,
                frame_test,
                pd.Series(y_train, index=frame_train.index),
                pd.Series(y_test, index=frame_test.index),
            )
        )

    misaligned_target = pd.Series(y_train, index=np.arange(10, 16))
    with pytest.raises(MultiTrainSplitError, match="indices must align"):
        classifier.fit(
            (
                frame_train,
                pd.DataFrame(X_test, columns=frame_train.columns),
                misaligned_target,
                pd.Series(y_test),
            )
        )


def test_classifier_split_preserves_every_class_in_both_partitions():
    penguins = pd.read_csv(
        Path(__file__).parents[2] / "examples" / "datasets" / "penguins.csv"
    )
    X_train, X_test, y_train, y_test = MultiClassifier().split(
        penguins,
        "species",
        test_size=0.25,
        auto_cat_encode=True,
        fix_nan_custom={
            "bill_length_mm": "interpolate",
            "bill_depth_mm": "interpolate",
            "flipper_length_mm": "interpolate",
            "body_mass_g": "interpolate",
            "sex": "ffill",
        },
    )

    assert set(y_train.unique()) == set(y_test.unique())
    assert len(X_train) + len(X_test) == len(penguins)

    rare_class = pd.DataFrame(
        {"feature": np.arange(21), "target": ["common"] * 20 + ["rare"]}
    )
    with pytest.raises(MultiTrainSplitError, match="least populated class"):
        MultiClassifier().split(
            rare_class,
            "target",
            auto_cat_encode=True,
            test_size=0.2,
        )


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
