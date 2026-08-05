"""Shared feature preparation and model execution for MultiTrain fits."""

from dataclasses import dataclass
import logging
from numbers import Integral, Real
import time
from typing import Optional

from joblib import Parallel, cpu_count, delayed, parallel_config
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import QuantileTransformer
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


@dataclass
class ModelRunResult:
    """The predictions and timings needed to score one completed model run."""

    name: str
    test_prediction: np.ndarray
    train_prediction: Optional[np.ndarray]
    test_roc_auc: float
    train_roc_auc: float
    elapsed: str
    error: Optional[str] = None


def _as_shared_array(values):
    """Expose tabular values as arrays that joblib can memory-map between workers."""
    if hasattr(values, "tocsr"):
        return values.tocsr()
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy(copy=False)
    return np.asarray(values)


def prepare_tabular_features(X_train, X_test, scaler=False, n_components=None):
    """Apply an optional scaler and PCA once, before any models are trained."""
    train = _as_shared_array(X_train)
    test = _as_shared_array(X_test)

    if not scaler:
        if n_components is not None:
            raise MultiTrainPCAError("n_components can only be used when pca is enabled")
        return train, test

    if train.ndim != 2:
        raise MultiTrainPCAError("PCA requires a two-dimensional feature matrix")

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
        fitted_scaler.set_params(
            n_quantiles=min(fitted_scaler.n_quantiles, train.shape[0])
        )

    scaled_train = fitted_scaler.fit_transform(train)
    scaled_test = fitted_scaler.transform(test)
    reducer = PCA(n_components=component_count, random_state=42)
    return reducer.fit_transform(scaled_train), reducer.transform(scaled_test)


def _validate_text_pipeline(vectorizer, pipeline_dict):
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
    """Read sparse support from sklearn, with a fallback for sklearn 1.3."""
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
    """Vectorize text once and prepare one dense copy only when a model needs it."""
    vectorizer_class = _validate_text_pipeline(vectorizer, pipeline_dict)
    transformer = vectorizer_class(
        ngram_range=pipeline_dict["ngram_range"],
        encoding=pipeline_dict["encoding"],
        max_features=pipeline_dict["max_features"],
        analyzer=pipeline_dict["analyzer"],
    )
    sparse_train = transformer.fit_transform(X_train)
    sparse_test = transformer.transform(X_test)

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
    """Choose a bounded process count without oversubscribing estimator threads."""
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

    available_cpus = cpu_count() or 1
    if model_threads < 0:
        if model_workers not in (None, 1):
            raise MultiTrainTypeError(
                "model_workers must be 1 when n_jobs is negative to avoid CPU oversubscription"
            )
        return 1

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
    start = time.perf_counter()
    try:
        model.fit(X_train, y_train)
        test_prediction = np.asarray(model.predict(X_test))
        train_prediction = (
            np.asarray(model.predict(X_train)) if show_train_score else None
        )
        if task == "classification":
            test_roc_auc = _classification_roc_auc(model, X_test, y_test)
            train_roc_auc = (
                _classification_roc_auc(model, X_train, y_train)
                if show_train_score
                else np.nan
            )
        else:
            test_roc_auc = np.nan
            train_roc_auc = np.nan
        error = None
    except Exception as exc:
        test_prediction = np.full(len(y_test), np.nan)
        train_prediction = (
            np.full(len(y_train), np.nan) if show_train_score else None
        )
        test_roc_auc = np.nan
        train_roc_auc = np.nan
        error = str(exc)

    return ModelRunResult(
        name=name,
        test_prediction=test_prediction,
        train_prediction=train_prediction,
        test_roc_auc=test_roc_auc,
        train_roc_auc=train_roc_auc,
        elapsed=_format_time(time.perf_counter() - start),
        error=error,
    )


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
    """Train every selected model on the full training set and return cached outputs."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    dense_model_names = dense_model_names or set()
    jobs = []
    for name, model in zip(model_names, models):
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
