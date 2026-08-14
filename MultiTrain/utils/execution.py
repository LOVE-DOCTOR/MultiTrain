"""Memory-aware preprocessing and model execution shared by both public APIs.

The public ``fit`` methods call this module in the following order:

1. ``prepare_tabular_features`` or ``prepare_text_features`` creates one shared
   representation of the train and test data.
2. ``run_models`` assigns that complete representation to every selected model.
3. ``_fit_model`` wraps, fits, predicts, and returns a ``ModelRunResult``.
4. The public class passes those cached outputs to metric helpers in ``utils.py``.

Keeping these stages here means expensive preprocessing happens once rather
than once per estimator, while model-specific transformations remain isolated
inside each estimator pipeline.
"""

from dataclasses import dataclass
import logging
from numbers import Integral, Real
import os
import time
from typing import Optional, Tuple
import warnings

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    QuantileTransformer,
    StandardScaler,
)
from tqdm.auto import tqdm

try:
    from sklearn.utils import get_tags as _get_estimator_tags
except ImportError:  # scikit-learn 1.3 does not expose the public tag helper.
    _get_estimator_tags = None

from MultiTrain.errors.errors import (
    MultiTrainPCAError,
    MultiTrainTextError,
    MultiTrainTypeError,
)
from MultiTrain.utils.utils import _classification_roc_auc, _format_time


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


GPU_MODEL_NAMES = {
    "CatBoostClassifier",
    "CatBoostRegressor",
    "XGBClassifier",
    "XGBRegressor",
}

# These estimators cannot consume scipy sparse matrices. When any are selected
# for text, they share one guarded dense copy instead of allocating one per model.
DENSE_ONLY_MODEL_NAMES = {
    "ARDRegression",
    "BayesianRidge",
    "GaussianNB",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "Lars",
    "LarsCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "TheilSenRegressor",
}

# These estimators are sensitive to features with very different magnitudes.
# Scaling is fitted inside each model run so test data never influences it.
STANDARD_SCALE_MODEL_NAMES = {
    "LogisticRegression",
    "LogisticRegressionCV",
    "SGDClassifier",
    "PassiveAggressiveClassifier",
    "Perceptron",
    "LinearSVC",
    "NuSVC",
    "SVC",
    "MLPClassifier",
    "PoissonRegressor",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "MLPRegressor",
    "LinearSVR",
    "NuSVR",
    "SVR",
}

# Standardizing large regression targets keeps the default optimization
# settings meaningful while predictions are converted back to the original unit.
TARGET_SCALE_MODEL_NAMES = {
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "MLPRegressor",
    "LinearSVR",
    "NuSVR",
    "SVR",
}

NONNEGATIVE_MODEL_NAMES = {
    "ComplementNB",
    "MultinomialNB",
}

# Poisson and Gamma regression have a stricter target domain than ordinary
# regression, so a reversible target transform is added only when necessary.
POSITIVE_TARGET_MODEL_NAMES = {
    "GammaRegressor",
    "PoissonRegressor",
}

CLASSIFICATION_CV_MODEL_NAMES = {
    "LogisticRegressionCV",
    "RidgeClassifierCV",
}

REGRESSION_CV_MODEL_NAMES = {
    "ElasticNetCV",
    "LarsCV",
    "LassoCV",
    "OrthogonalMatchingPursuitCV",
    "RidgeCV",
}

NEIGHBOR_MODEL_NAMES = {
    "KNeighborsClassifier",
    "KNeighborsRegressor",
}


@dataclass
class ModelRunResult:
    """Cached outputs returned by ``_fit_model`` and consumed by public ``fit``.

    Label predictions feed ordinary metrics. Probability arrays feed log loss,
    Brier score, and ROC AUC. The fitted estimator and captured warnings become
    public post-fit artifacts. A failed estimator still returns this structure,
    but its predictions are NaN and the error fields explain what failed and at
    which execution stage. This lets unrelated selected models finish normally.
    """

    name: str
    test_prediction: np.ndarray
    train_prediction: Optional[np.ndarray]
    test_roc_auc: float
    train_roc_auc: float
    test_probability: Optional[np.ndarray]
    train_probability: Optional[np.ndarray]
    model_classes: Optional[np.ndarray]
    elapsed: str
    estimator: Optional[object] = None
    warnings: Tuple[Tuple[str, str], ...] = ()
    error: Optional[str] = None
    error_stage: Optional[str] = None
    error_type: Optional[str] = None


def build_run_artifacts(completed):
    """Build the reusable fitted outputs exposed by the public model classes.

    Classification and regression call this after ``run_models`` so both APIs
    expose the same predictable artifact layout. Failed models retain their NaN
    predictions for inspection, but only successfully fitted estimators appear
    in ``models``.
    """
    predictions = {"test": {}, "train": {}}
    probabilities = {"test": {}, "train": {}}
    fitted_models = {}
    warning_rows = []
    failure_rows = []

    for result in completed:
        predictions["test"][result.name] = result.test_prediction
        if result.train_prediction is not None:
            predictions["train"][result.name] = result.train_prediction

        if result.test_probability is not None:
            probabilities["test"][result.name] = result.test_probability
        if result.train_probability is not None:
            probabilities["train"][result.name] = result.train_probability

        if result.estimator is not None and result.error is None:
            fitted_models[result.name] = result.estimator

        warning_rows.extend(
            {
                "Model": result.name,
                "Category": category,
                "Message": message,
            }
            for category, message in result.warnings
        )
        if result.error is not None:
            failure_rows.append(
                {
                    "Model": result.name,
                    "Stage": result.error_stage,
                    "Exception": result.error_type,
                    "Message": result.error,
                }
            )

    return {
        "models": fitted_models,
        "predictions": predictions,
        "probabilities": probabilities,
        "warnings": pd.DataFrame(
            warning_rows,
            columns=["Model", "Category", "Message"],
        ),
        "failures": pd.DataFrame(
            failure_rows,
            columns=["Model", "Stage", "Exception", "Message"],
        ),
    }


def _as_shared_array(values):
    """Normalize features without making an unnecessary full-data copy.

    ``prepare_tabular_features`` calls this before ``run_models``. Pandas values
    use ``copy=False`` where possible, and sparse matrices remain sparse, which
    allows joblib to share or memory-map the result between worker processes.
    """
    if hasattr(values, "tocsr"):
        return values.tocsr()
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy(copy=False)
    return np.asarray(values)


def prepare_tabular_features(X_train, X_test, scaler=False, n_components=None):
    """Prepare the shared tabular matrices used by every selected estimator.

    When PCA is enabled, the scaler and PCA are fitted only on ``X_train`` and
    then used to transform ``X_test``. The public classifier and regressor call
    this immediately before ``run_models``, preventing test-data leakage and
    avoiding a repeated PCA fit for every model.
    """
    train = _as_shared_array(X_train)
    test = _as_shared_array(X_test)

    if not scaler:
        if n_components is not None:
            raise MultiTrainPCAError("n_components can only be used when pca is enabled")
        return train, test

    if train.ndim != 2:
        raise MultiTrainPCAError("PCA requires a two-dimensional feature matrix")

    # PCA cannot produce more components than either the number of training
    # rows or the number of input features.
    max_components = min(train.shape[0], train.shape[1])
    if n_components is None:
        component_count = max_components
    elif isinstance(n_components, bool) or not isinstance(n_components, Real):
        raise MultiTrainPCAError(
            "n_components must be None, a positive integer, or a float between 0 and 1"
        )
    elif isinstance(n_components, Integral):
        if not 1 <= n_components <= max_components:
            raise MultiTrainPCAError(
                f"n_components must be between 1 and {max_components} for these training data"
            )
        component_count = int(n_components)
    else:
        if not 0 < n_components < 1:
            raise MultiTrainPCAError("A float n_components value must be between 0 and 1")
        component_count = n_components

    fitted_scaler = clone(scaler)
    if isinstance(fitted_scaler, QuantileTransformer):
        # QuantileTransformer cannot estimate more quantiles than training rows.
        # Cloning keeps the shared scaler constant untouched for later fit calls.
        fitted_scaler.set_params(
            n_quantiles=min(fitted_scaler.n_quantiles, train.shape[0])
        )

    scaled_train = fitted_scaler.fit_transform(train)
    scaled_test = fitted_scaler.transform(test)
    reducer = PCA(n_components=component_count, random_state=42)
    return reducer.fit_transform(scaled_train), reducer.transform(scaled_test)


def _validate_text_pipeline(vectorizer, pipeline_dict):
    """Validate text options and return the requested sklearn vectorizer class.

    ``prepare_text_features`` uses the returned class only after every required
    option is present, so malformed text configuration fails before allocating
    a potentially large vocabulary matrix.
    """
    vectorizers = {"count": CountVectorizer, "tfidf": TfidfVectorizer}
    if vectorizer not in vectorizers:
        raise MultiTrainTextError('vectorizer must be either "count" or "tfidf"')
    if not isinstance(pipeline_dict, dict):
        raise MultiTrainTextError("pipeline_dict must be a dictionary")

    required = {"ngram_range", "encoding", "max_features", "analyzer"}
    missing = required - set(pipeline_dict)
    if missing:
        raise MultiTrainTextError(
            f"pipeline_dict is missing required keys: {sorted(missing)}"
        )
    return vectorizers[vectorizer]


def _supports_sparse_input(model):
    """Tell ``prepare_text_features`` whether a model needs dense text data.

    Scikit-learn 1.6 introduced the public tag helper. MultiTrain also supports
    older releases, where the maintained ``DENSE_ONLY_MODEL_NAMES`` set supplies
    the same decision.
    """
    if _get_estimator_tags is not None:
        return _get_estimator_tags(model).input_tags.sparse
    return model.__class__.__name__ not in DENSE_ONLY_MODEL_NAMES


def prepare_text_features(
    vectorizer,
    pipeline_dict,
    X_train,
    X_test,
    models,
    max_dense_bytes,
):
    """Create the shared sparse text matrices and an optional guarded dense copy.

    The classifier's ``fit`` method passes both representations to
    ``run_models``. Sparse-compatible estimators use the sparse matrices;
    dense-only estimators are routed to the one dense copy by model name.
    ``max_dense_bytes`` prevents that conversion from exhausting memory.
    """
    vectorizer_class = _validate_text_pipeline(vectorizer, pipeline_dict)
    transformer = vectorizer_class(
        ngram_range=pipeline_dict["ngram_range"],
        encoding=pipeline_dict["encoding"],
        max_features=pipeline_dict["max_features"],
        analyzer=pipeline_dict["analyzer"],
        dtype=np.float64,
    )
    try:
        # Keep one floating-point sparse representation that every selected
        # estimator can consume without allocating a second full matrix.
        sparse_train = transformer.fit_transform(X_train).astype(
            np.float64,
            copy=False,
        )
        sparse_test = transformer.transform(X_test).astype(
            np.float64,
            copy=False,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise MultiTrainTextError(f"Unable to vectorize the text data: {exc}") from exc

    requires_dense = {
        model.__class__.__name__
        for model in models
        if not _supports_sparse_input(model)
    }
    if not requires_dense:
        return sparse_train, sparse_test, None, None, requires_dense

    if max_dense_bytes is not None:
        if (
            isinstance(max_dense_bytes, bool)
            or not isinstance(max_dense_bytes, int)
            or max_dense_bytes <= 0
        ):
            raise MultiTrainTypeError(
                "max_dense_bytes must be a positive integer or None"
            )
        # Dense storage uses one value for every row/feature combination. Both
        # train and test matrices exist at once, hence the summed row count.
        required_bytes = (
            sparse_train.shape[0] + sparse_test.shape[0]
        ) * sparse_train.shape[1] * sparse_train.dtype.itemsize
        if required_bytes > max_dense_bytes:
            model_names = ", ".join(sorted(requires_dense))
            required_mib = required_bytes / (1024 ** 2)
            limit_mib = max_dense_bytes / (1024 ** 2)
            raise MultiTrainTextError(
                f"{model_names} require dense text features ({required_mib:.1f} MiB), "
                f"which exceeds max_dense_bytes ({limit_mib:.1f} MiB). Increase the "
                "limit, pass None, or choose sparse-compatible models."
            )

    return (
        sparse_train,
        sparse_test,
        sparse_train.toarray(),
        sparse_test.toarray(),
        requires_dense,
    )


def resolve_model_workers(model_workers, model_threads, model_count):
    """Balance process-level and estimator-level parallelism.

    ``run_models`` calls this with ``model_workers`` from the public constructor
    and ``model_threads`` from ``n_jobs``. Available CPUs are divided by threads
    per estimator, then bounded by the number of models. Negative ``n_jobs``
    already means an estimator may use every CPU, so model processes must remain
    sequential in that case.
    """
    if model_count <= 0:
        return 1
    if model_workers is not None and (
        isinstance(model_workers, bool)
        or not isinstance(model_workers, int)
        or model_workers == 0
        or model_workers < -1
    ):
        raise MultiTrainTypeError(
            "model_workers must be None, -1, or a positive integer"
        )

    available_cpus = os.cpu_count() or 1
    if model_threads < 0:
        if model_workers not in (None, 1):
            raise MultiTrainTypeError(
                "model_workers must be 1 when n_jobs is negative to avoid CPU oversubscription"
            )
        return 1

    # Integer division reserves the requested thread budget for each process.
    # Both max calls guarantee a usable value on unusual or constrained hosts.
    available_workers = max(1, available_cpus // max(1, model_threads))
    requested = min(4, available_workers) if model_workers is None else model_workers
    if requested == -1:
        requested = available_workers
    return max(1, min(requested, available_workers, model_count))


def _fit_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    show_train_score,
    task,
):
    """Fit one estimator and cache everything later scoring can require.

    ``run_models`` invokes this function directly or in a joblib worker. Model
    wrappers are created by ``_prepare_training_estimator`` before fitting.
    Failures are converted to a ``ModelRunResult`` containing NaNs so one broken
    estimator does not cancel unrelated model runs; ``run_models`` logs the
    stored error after restoring result order.
    """
    start = time.perf_counter()
    captured_warnings = []
    stage = "preparation"
    try:
        # Capture warnings for ``warnings_`` and then re-emit them through the
        # caller's warning filters. This keeps ordinary warnings visible while
        # still respecting projects that deliberately promote a category.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training_estimator = _prepare_training_estimator(
                name,
                model,
                X_train,
                task,
                y_train=y_train,
            )
            stage = "fit"
            training_estimator.fit(X_train, y_train)
            stage = "prediction"
            test_prediction = np.asarray(training_estimator.predict(X_test))
            train_prediction = (
                np.asarray(training_estimator.predict(X_train))
                if show_train_score
                else None
            )
            if task == "classification":
                stage = "probability prediction"
                test_probability = _classification_probabilities(
                    training_estimator, X_test
                )
                train_probability = (
                    _classification_probabilities(training_estimator, X_train)
                    if show_train_score
                    else None
                )
                # Pipelines normally forward ``classes_`` from their final model.
                # The fallback keeps class order available for compatible custom
                # estimators that do not expose that attribute explicitly.
                model_classes = np.asarray(
                    getattr(training_estimator, "classes_", np.unique(y_train))
                )
                stage = "ROC AUC scoring"
                test_roc_auc = _classification_roc_auc(
                    training_estimator,
                    X_test,
                    y_test,
                    probabilities=test_probability,
                )
                train_roc_auc = (
                    _classification_roc_auc(
                        training_estimator,
                        X_train,
                        y_train,
                        probabilities=train_probability,
                    )
                    if show_train_score
                    else np.nan
                )
            else:
                test_roc_auc = np.nan
                train_roc_auc = np.nan
                test_probability = None
                train_probability = None
                model_classes = None

        captured_warnings = [
            (warning.category.__name__, str(warning.message))
            for warning in caught
        ]
        stage = "warning handling"
        for warning in caught:
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
        error = None
        error_stage = None
        error_type = None
    except Exception as exc:
        # If fitting failed after emitting a warning, preserve that context even
        # though execution never reached the normal warning-copying block.
        if not captured_warnings and "caught" in locals():
            captured_warnings = [
                (warning.category.__name__, str(warning.message))
                for warning in caught
            ]
        test_prediction = np.full(len(y_test), np.nan)
        train_prediction = (
            np.full(len(y_train), np.nan) if show_train_score else None
        )
        test_roc_auc = np.nan
        train_roc_auc = np.nan
        test_probability = None
        train_probability = None
        model_classes = None
        error = str(exc)
        error_stage = stage
        error_type = type(exc).__name__
        training_estimator = None

    return ModelRunResult(
        name=name,
        test_prediction=test_prediction,
        train_prediction=train_prediction,
        test_roc_auc=test_roc_auc,
        train_roc_auc=train_roc_auc,
        test_probability=test_probability,
        train_probability=train_probability,
        model_classes=model_classes,
        elapsed=_format_time(time.perf_counter() - start),
        estimator=training_estimator,
        warnings=tuple(captured_warnings),
        error=error,
        error_stage=error_stage,
        error_type=error_type,
    )


def _classification_probabilities(model, X):
    """Read probabilities for metrics that cannot use predicted class labels.

    ``_fit_model`` calls this once per requested split. Returning ``None`` is
    intentional for estimators such as LinearSVC; the probability metric helper
    later turns that unavailable measurement into NaN without affecting the
    model's label-based scores.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return np.asarray(model.predict_proba(X))
    except Exception:
        return None


def _prepare_training_estimator(name, model, X_train, task, y_train=None):
    """Wrap one estimator for the current dataset without changing its output unit.

    This function is called only from ``_fit_model``. It adjusts CV/neighbors for
    small training sets, adds feature scaling for convergence-sensitive models,
    makes negative features usable by count-based Naive Bayes, and applies
    reversible target scaling where a regressor requires it. Every fitted value
    comes from training data; the returned pipeline later transforms test data.
    """
    estimator = model
    # Five-fold CV and five-neighbor defaults are invalid on small datasets.
    # Bound them using training data only, without consulting held-out rows.
    if y_train is not None:
        training_rows = len(y_train)
        if name in CLASSIFICATION_CV_MODEL_NAMES:
            _, class_counts = np.unique(y_train, return_counts=True)
            folds = min(5, int(class_counts.min()))
            if folds >= 2:
                estimator.set_params(cv=folds)
        elif name in REGRESSION_CV_MODEL_NAMES:
            # Regression scorers such as R-squared need at least two rows in
            # each validation fold to produce a meaningful value.
            folds = min(5, max(2, training_rows // 2), training_rows)
            if folds >= 2:
                estimator.set_params(cv=folds)

        if name in NEIGHBOR_MODEL_NAMES:
            estimator.set_params(n_neighbors=min(5, training_rows))

    if name in STANDARD_SCALE_MODEL_NAMES:
        sparse_input = hasattr(X_train, "tocsr")
        # Centering a sparse matrix would fill its implicit zeros and destroy its
        # memory advantage, so sparse inputs are scaled without mean subtraction.
        estimator = make_pipeline(
            StandardScaler(with_mean=not sparse_input),
            estimator,
        )

    if name in NONNEGATIVE_MODEL_NAMES:
        # Sparse matrices store only non-zero entries in ``data``. Inspecting
        # that array avoids converting the whole matrix merely to find its minimum.
        values = X_train.data if hasattr(X_train, "tocsr") else np.asarray(X_train)
        if values.size and np.min(values) < 0:
            steps = []
            if hasattr(X_train, "tocsr"):
                steps.append(
                    FunctionTransformer(
                        _dense_array,
                        accept_sparse=True,
                    )
                )
            steps.extend([MinMaxScaler(clip=True), estimator])
            estimator = make_pipeline(*steps)

    # Target transformers are inverted before prediction, so regression scores
    # remain in the units supplied by the user.
    if task == "regression":
        if name in TARGET_SCALE_MODEL_NAMES:
            estimator = TransformedTargetRegressor(
                regressor=estimator,
                transformer=StandardScaler(),
            )
        elif name in POSITIVE_TARGET_MODEL_NAMES and y_train is not None:
            target = np.asarray(y_train, dtype=float)
            if target.size and np.min(target) <= 0:
                estimator = TransformedTargetRegressor(
                    regressor=estimator,
                    transformer=MinMaxScaler(feature_range=(1e-6, 1.0)),
                )
    return estimator


def _dense_array(values):
    """Convert values inside a model pipeline when a scaler cannot accept sparse input."""
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def run_models(
    model_names,
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    show_train_score,
    task,
    model_workers,
    model_threads,
    use_gpu=False,
    dense_train=None,
    dense_test=None,
    dense_model_names=None,
):
    """Schedule every selected model and return results in selection order.

    Public ``fit`` methods call this after shared preprocessing. CPU models may
    run in separate processes, while GPU models run sequentially to avoid device
    contention. ``generator_unordered`` updates progress as soon as models finish;
    the final name lookup restores the user's original order before scoring.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    dense_model_names = dense_model_names or set()
    jobs = []
    for name, model in zip(model_names, models):
        # Text preprocessing may provide both sparse and dense matrices. Each
        # job records the representation accepted by its estimator.
        use_dense = name in dense_model_names
        jobs.append(
            (
                name,
                model,
                dense_train if use_dense else X_train,
                dense_test if use_dense else X_test,
            )
        )

    gpu_jobs = [job for job in jobs if use_gpu and job[0] in GPU_MODEL_NAMES]
    cpu_jobs = [job for job in jobs if not (use_gpu and job[0] in GPU_MODEL_NAMES)]
    results = []
    progress = tqdm(total=len(jobs), desc="Training Models", leave=False)

    workers = resolve_model_workers(model_workers, model_threads, len(cpu_jobs))
    if workers == 1:
        for name, model, train_features, test_features in cpu_jobs:
            progress.set_postfix_str(f"Model: {name}")
            results.append(
                _fit_model(
                    name,
                    model,
                    train_features,
                    y_train,
                    test_features,
                    y_test,
                    show_train_score,
                    task,
                )
            )
            progress.update()
    elif cpu_jobs:
        # Loky uses processes, and inner_max_num_threads prevents each child
        # estimator from silently multiplying the requested CPU usage.
        with parallel_config(
            backend="loky",
            n_jobs=workers,
            inner_max_num_threads=max(1, model_threads),
        ):
            completed = Parallel(return_as="generator_unordered")(
                delayed(_fit_model)(
                    name,
                    model,
                    train_features,
                    y_train,
                    test_features,
                    y_test,
                    show_train_score,
                    task,
                )
                for name, model, train_features, test_features in cpu_jobs
            )
            for result in completed:
                progress.set_postfix_str(f"Model: {result.name}")
                results.append(result)
                progress.update()

    # GPU estimators run one at a time so they do not compete for the same device.
    for name, model, train_features, test_features in gpu_jobs:
        progress.set_postfix_str(f"Model: {name}")
        results.append(
            _fit_model(
                name,
                model,
                train_features,
                y_train,
                test_features,
                y_test,
                show_train_score,
                task,
            )
        )
        progress.update()
    progress.close()

    # Parallel completion order is nondeterministic. Reindexing by model name
    # makes repeated result tables stable and matches ``custom_models`` order.
    by_name = {result.name: result for result in results}
    ordered = [by_name[name] for name in model_names]
    for result in ordered:
        if result.error:
            logger.error(
                "%s unable to fit or predict. Reason: %s",
                result.name,
                result.error,
            )
    return ordered
