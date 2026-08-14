"""Public classification workflow for splitting data, training models, and scoring them.

This module owns user-facing validation and orchestration. Shared preprocessing
and execution live in ``MultiTrain.utils`` so classification and regression use
the same rules without duplicating expensive work.
"""

from dataclasses import dataclass, field
from numbers import Real
import platform
from typing import Dict, Optional, Union
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from MultiTrain.utils.utils import (
    _cat_encoder,
    _calculate_probability_metric,
    _metrics,
    _calculate_metric,
    _display_table,
    _handle_missing_values,
    _manual_encoder,
    _non_auto_cat_encode_error,
    _prep_model_names_list,
    _prepare_train_test,
    _validate_datasplits,
    _validate_supervised_dataset,
)
from MultiTrain.utils.execution import (
    build_run_artifacts,
    prepare_tabular_features,
    prepare_text_features,
    run_models,
)

import pandas as pd
from sklearn.model_selection import train_test_split
from MultiTrain.errors.errors import *

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SUPPORTED_SCALERS = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'MaxAbsScaler': MaxAbsScaler(), 
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer(),
    'QuantileTransformer': QuantileTransformer(),
    'PowerTransformer': PowerTransformer()
}

@dataclass
class MultiClassifier:
    """Configure and compare MultiTrain's classification estimators.

    ``n_jobs`` controls threads inside an estimator, while ``model_workers``
    controls how many different estimators run at once. Text mode shares one
    vectorizer; tabular mode can share a scaler and PCA transform. Built-in
    models can be selected with a list of names, or ``custom_models`` can map
    user-chosen names to estimator objects. ``model_params`` applies validated
    overrides after that selection. A completed fit retains its estimators,
    predictions, probabilities, warnings, and failures on attributes ending in
    an underscore.

    Parameters
    ----------
    n_jobs : int, default=1
        Thread or process count passed to estimators that support internal
        parallelism. The value cannot be zero.
    random_state : int, default=42
        Seed used when MultiTrain creates estimators. Pass a separate seed to
        :meth:`split` to control the holdout partition.
    custom_models : list of str, dict, or None, default=None
        Built-in estimator names to fit, or a mapping of result names to
        estimator objects. ``None`` selects the complete built-in catalog.
    max_iter : int, default=1000
        Shared iteration or estimator budget applied to built-in models that
        expose a compatible parameter.
    use_gpu : bool, default=False
        Configure supported CatBoost and XGBoost estimators for GPU execution.
        GPU execution is disabled on macOS.
    device : str, default="0"
        GPU device identifier forwarded to supported GPU estimators.
    text : bool, default=False
        Accept a single text feature column and require a vectorizer during
        :meth:`fit`.
    model_workers : int or None, default=None
        Maximum number of estimators trained concurrently. ``None`` chooses a
        bounded automatic value; ``-1`` permits all available CPU allocations.
    model_params : dict or None, default=None
        Parameter overrides keyed by selected model name. Nested pipeline
        parameters use scikit-learn's ``step__parameter`` syntax.

    Attributes
    ----------
    results_ : pandas.DataFrame or None
        Result returned by the most recent successful :meth:`fit` call.
    models_ : dict
        Successfully fitted estimators keyed by result name.
    predictions_ : dict
        Cached test and optional training predictions, grouped by partition and
        model name.
    probabilities_ : dict
        Cached test and optional training probabilities for estimators that
        implement ``predict_proba``.
    warnings_ : pandas.DataFrame
        Model-attributed warning categories and messages from the latest fit.
    failures_ : pandas.DataFrame
        Model, execution stage, exception type, and message for failed models.

    Examples
    --------
    >>> from MultiTrain import MultiClassifier
    >>> train = MultiClassifier(
    ...     custom_models=["LogisticRegression"],
    ...     n_jobs=1,
    ... )
    """

    n_jobs: int = 1
    random_state: int = 42
    custom_models: Optional[Union[list, dict]] = None
    max_iter: int = 1000
    use_gpu: bool = False
    device: str = '0'
    text: bool = False
    model_workers: Optional[int] = None
    model_params: Optional[dict] = None
    results_: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)
    models_: dict = field(init=False, default_factory=dict, repr=False)
    predictions_: dict = field(init=False, default_factory=dict, repr=False)
    probabilities_: dict = field(init=False, default_factory=dict, repr=False)
    warnings_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    failures_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    
    def __post_init__(self):
        """Validate configuration before any dataset or model is allocated."""

        # Booleans are subclasses of int in Python, so integer options need an
        # explicit boolean check to avoid accepting values such as n_jobs=True.
        type_validations = {
            'n_jobs': (self.n_jobs, int),
            'random_state': (self.random_state, int),
            'max_iter': (self.max_iter, int),
            'use_gpu': (self.use_gpu, bool),
            'device': (self.device, str),
            'text': (self.text, bool),
        }
        
        for param_name, (param_value, expected_type) in type_validations.items():
            invalid_int = expected_type is int and (
                not isinstance(param_value, int) or isinstance(param_value, bool)
            )
            if invalid_int or (
                expected_type is not int and not isinstance(param_value, expected_type)
            ):
                raise MultiTrainTypeError(
                    f'Invalid type for {param_name}: expected {expected_type.__name__}, '
                    f'got {type(param_value).__name__}. Please provide a {expected_type.__name__} value.'
                )
                
        if not isinstance(self.custom_models, (list, dict, type(None))):
            raise MultiTrainTypeError(
                f'Invalid type for custom_models: expected a list, dictionary, or None, '
                f'got {type(self.custom_models).__name__}.'
            )
        if isinstance(self.custom_models, list) and not all(
            isinstance(model, str) for model in self.custom_models
        ):
            raise MultiTrainTypeError("Every custom model name must be a string")
        if self.model_params is not None and not isinstance(self.model_params, dict):
            raise MultiTrainTypeError("model_params must be a dictionary or None")
        if isinstance(self.model_params, dict):
            for model_name, parameters in self.model_params.items():
                if not isinstance(model_name, str) or not model_name:
                    raise MultiTrainTypeError(
                        "Every model_params key must be a non-empty string"
                    )
                if not isinstance(parameters, dict):
                    raise MultiTrainTypeError(
                        f"Parameters for {model_name} must be provided as a dictionary"
                    )
        if self.n_jobs == 0:
            raise MultiTrainTypeError("n_jobs cannot be zero")
        if self.model_workers is not None and (
            not isinstance(self.model_workers, int)
            or isinstance(self.model_workers, bool)
            or self.model_workers == 0
            or self.model_workers < -1
        ):
            raise MultiTrainTypeError(
                "model_workers must be None, -1, or a positive integer"
            )
        if self.max_iter <= 0:
            raise MultiTrainTypeError("max_iter must be positive")
        if not self.device:
            raise MultiTrainTypeError("device cannot be empty")

        if self.use_gpu and platform.system() == 'Darwin':
            logger.warning('GPU acceleration is not supported on macOS')
            
        logger.debug('MultiTrain classifier initialized')

    def _reset_fit_artifacts(self):
        """Clear outputs from the previous run before validating a new one."""
        self.results_ = None
        self.models_ = {}
        self.predictions_ = {"test": {}, "train": {}}
        self.probabilities_ = {"test": {}, "train": {}}
        self.warnings_ = pd.DataFrame(columns=["Model", "Category", "Message"])
        self.failures_ = pd.DataFrame(
            columns=["Model", "Stage", "Exception", "Message"]
        )

    def _store_fit_artifacts(self, completed, final_dataframe):
        """Expose the estimators and cached outputs created by one fit call."""
        artifacts = build_run_artifacts(completed)
        self.results_ = final_dataframe
        self.models_ = artifacts["models"]
        self.predictions_ = artifacts["predictions"]
        self.probabilities_ = artifacts["probabilities"]
        self.warnings_ = artifacts["warnings"]
        self.failures_ = artifacts["failures"]
            
    def split(
        self,
        data: Union[pd.DataFrame, str],
        target: str,  # Name of the target column
        random_state: int = 42,  # Random state for reproducibility
        test_size: float = 0.2,  # Proportion of the dataset for the test split
        auto_cat_encode: bool = False,  # Automatically encode all categorical columns if True
        manual_encode: dict = None,  # Manual encoding dictionary, e.g., {'label': ['column1'], 'onehot': ['column2']}
        fix_nan_custom: Optional[
            Dict
        ] = False,  # Custom NaN handling, e.g., {'column1': 'ffill'}
        drop: list = None,  # List of columns to drop, e.g., ['column1', 'column2']
    ):
        """Create stratified training and test partitions.

        The source data is validated before it is split. Missing-value rules and
        categorical encoders are then learned from the training partition and
        applied to the test partition, which prevents test data from influencing
        preprocessing.

        Parameters
        ----------
        data : pandas.DataFrame or str
            Source dataframe or path to a CSV file.
        target : str
            Name of the classification target column.
        random_state : int, default=42
            Seed passed to scikit-learn's train/test splitter.
        test_size : float, default=0.2
            Fraction of rows assigned to the test partition. It must be between
            zero and one.
        auto_cat_encode : bool, default=False
            Automatically label-encode every categorical feature when true.
        manual_encode : dict or None, default=None
            Explicit encoding instructions. Supported keys are ``"label"`` and
            ``"onehot"``; each value is a list of feature names.
        fix_nan_custom : dict, False, or None, default=False
            Per-column missing-value strategies such as
            ``{"age": "interpolate", "city": "ffill"}``.
        drop : list or None, default=None
            Feature columns to remove before the split.

        Returns
        -------
        tuple
            ``(X_train, X_test, y_train, y_test)``. Classification targets are
            stratified so each partition retains the class distribution when the
            data contains enough samples.

        Raises
        ------
        MultiTrainTypeError
            If an argument has an unsupported type.
        MultiTrainColumnMissingError
            If the target, dropped column, or encoded column does not exist.
        MultiTrainEncodingError
            If categorical encoding instructions conflict or leave categorical
            columns unencoded.
        MultiTrainNaNError
            If missing values remain without a configured strategy.
        MultiTrainSplitError
            If the dataset cannot produce a valid stratified holdout split.

        Notes
        -----
        The returned tuple can be passed directly to :meth:`fit`. An equivalent
        four-item tuple created with scikit-learn is also accepted by ``fit``.
        """

        if not isinstance(target, str) or not target:
            raise MultiTrainTypeError("target must be a non-empty string")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise MultiTrainTypeError("random_state must be an integer")
        if not isinstance(test_size, Real) or isinstance(test_size, bool):
            raise MultiTrainTypeError("test_size must be numeric")
        if not 0 < test_size < 1:
            raise MultiTrainSplitError("test_size must be between 0 and 1")
        if not isinstance(auto_cat_encode, bool):
            raise MultiTrainTypeError("auto_cat_encode must be a boolean")

        if isinstance(data, pd.DataFrame):
            dataset = data.copy()
        elif isinstance(data, str):
            dataset = pd.read_csv(data)
        else:
            raise MultiTrainDatasetTypeError('You must either pass in a dataframe or a filepath')

        # Fail before modifying the dataset when preprocessing instructions are malformed.
        if manual_encode is not None and not isinstance(manual_encode, dict):
            raise MultiTrainTypeError(
                f"manual_encode must be a dictionary or None. Got {type(manual_encode)}"
            )
        if (
            fix_nan_custom is not False
            and fix_nan_custom is not None
            and not isinstance(fix_nan_custom, dict)
        ):
            raise MultiTrainTypeError(
                f"fix_nan_custom must be a dictionary. Got {type(fix_nan_custom)}"
            )

        # A column needs one unambiguous encoding strategy.
        if manual_encode:
            invalid_keys = set(manual_encode) - {"label", "onehot"}
            if invalid_keys:
                raise MultiTrainEncodingError(
                    f"Unsupported encoding types: {sorted(invalid_keys)}"
                )
            for encoding_type, columns in manual_encode.items():
                if not isinstance(columns, (list, tuple)):
                    raise MultiTrainTypeError(
                        f"Columns for {encoding_type} must be a list or tuple"
                    )
            overlap = set(manual_encode.get("label", [])) & set(
                manual_encode.get("onehot", [])
            )
            if overlap:
                raise MultiTrainEncodingError(
                    f"Columns cannot use multiple encodings: {sorted(overlap)}"
                )

        if auto_cat_encode and manual_encode:
            raise MultiTrainEncodingError("Cannot use both auto_cat_encode and manual_encode")
        if manual_encode and target in manual_encode.get("onehot", []):
            raise MultiTrainEncodingError("The target column cannot be one-hot encoded")

        # Remove ignored features before checking the columns used for training.
        if drop is not None and not isinstance(drop, list):
            raise MultiTrainTypeError(f"Drop parameter must be a list. Got {type(drop)}")
        if drop:
            missing_drop_columns = [column for column in drop if column not in dataset]
            if missing_drop_columns:
                raise MultiTrainColumnMissingError(
                    f"Columns to drop were not found: {missing_drop_columns}"
                )
            dataset.drop(drop, axis=1, inplace=True)

        if target not in dataset.columns:
            raise MultiTrainColumnMissingError(f"Target column {target} not found in columns")

        _validate_supervised_dataset(dataset, target, "classification")

        if not self.text:
            _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode)

        # Split first so encoders and missing-value rules cannot learn from held-out rows.
        try:
            train_dataset, test_dataset = train_test_split(
                dataset,
                test_size=test_size,
                random_state=random_state,
                stratify=dataset[target],
            )
        except ValueError as e:
            raise MultiTrainSplitError(f"Unable to split the dataset: {e}") from e

        train_dataset, test_dataset = _prepare_train_test(
            train_dataset,
            test_dataset,
            auto_cat_encode=auto_cat_encode,
            manual_encode=manual_encode,
            fix_nan_custom=fix_nan_custom,
        )
        X_train = train_dataset.drop(target, axis=1)
        X_test = test_dataset.drop(target, axis=1)
        y_train = train_dataset[target]
        y_test = test_dataset[target]

        return X_train, X_test, y_train, y_test

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,
        show_train_score: bool = False,
        imbalanced: bool = False,
        sort: str = None,
        pca: Union[bool, str] = False, # If not False, set the type of scaler to use before PCA
        vectorizer: str = None,  # Example: 'count' or 'tfidf'
        pipeline_dict: dict = None,  # Example: {'ngram_range': (1, 2), 'encoding': 'utf-8', 'max_features': 5000, 'analyzer': 'word'}
        return_best_model: Optional[str] = None,  # Example: 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'balanced_accuracy'
        n_components: Optional[Union[int, float]] = None,
        max_dense_bytes: Optional[int] = 1024 ** 3,
    ):
        """Fit and calculate metrics for every selected classification model.

        MultiTrain validates the split, prepares one shared feature representation,
        trains each estimator on the complete training partition, and calculates
        every result from cached predictions. Fitted models and intermediate
        outputs remain available through the post-fit attributes.

        Parameters
        ----------
        datasplits : tuple
            Four items ordered as ``X_train, X_test, y_train, y_test``.
        custom_metric : str or None, default=None
            Name of an additional supported scalar scikit-learn metric.
        show_train_score : bool, default=False
            Include metrics calculated on the training partition.
        imbalanced : bool, default=False
            Use micro averaging for precision, recall, and F1 metrics.
        sort : str or None, default=None
            Result column used to order the returned dataframe.
        pca : str or False, default=False
            Name of the scaler applied before a shared PCA transformation. Pass
            ``False`` to leave tabular features unchanged.
        vectorizer : {"count", "tfidf"} or None, default=None
            Vectorizer used when this classifier was created with ``text=True``.
        pipeline_dict : dict or None, default=None
            Keyword arguments forwarded to the selected text vectorizer.
        return_best_model : str or None, default=None
            Return only the row with the strongest value for this metric.
        n_components : int, float, or None, default=None
            Number of PCA components, or an explained-variance fraction between
            zero and one. This option is unavailable for text classification.
        max_dense_bytes : int or None, default=1073741824
            Maximum estimated allocation allowed when a sparse text matrix must
            be converted to a dense matrix. ``None`` disables the guard.

        Returns
        -------
        pandas.DataFrame
            Metrics for the selected models. The same dataframe is stored in
            :attr:`results_`; fitted estimators, predictions, probabilities,
            warnings, and failures are stored in the other post-fit attributes.

        Raises
        ------
        MultiTrainTypeError
            If an argument or split item has an unsupported type.
        MultiTrainMetricError
            If a metric name or requested ordering is unsupported.
        MultiTrainPCAError
            If PCA configuration is invalid or is combined with text mode.
        MultiTrainTextError
            If text input or vectorizer configuration is invalid.

        See Also
        --------
        split : Create and validate a stratified train/test partition.
        """
        self._reset_fit_artifacts()
        if custom_metric is not None and not isinstance(custom_metric, str):
            raise MultiTrainTypeError("custom_metric must be a string or None")
        if not isinstance(show_train_score, bool):
            raise MultiTrainTypeError("show_train_score must be a boolean")
        if not isinstance(imbalanced, bool):
            raise MultiTrainTypeError("imbalanced must be a boolean")
        if sort is not None and not isinstance(sort, str):
            raise MultiTrainTypeError("sort must be a string or None")
        if pca is not False and not isinstance(pca, str):
            raise MultiTrainPCAError(
                "pca must be False or the name of a supported scaler"
            )
        if vectorizer is not None and not isinstance(vectorizer, str):
            raise MultiTrainTypeError("vectorizer must be a string or None")
        if pipeline_dict is not None and not isinstance(pipeline_dict, dict):
            raise MultiTrainTypeError("pipeline_dict must be a dictionary or None")
        if return_best_model is not None and not isinstance(return_best_model, str):
            raise MultiTrainTypeError("return_best_model must be a string or None")
        if max_dense_bytes is not None and (
            not isinstance(max_dense_bytes, int)
            or isinstance(max_dense_bytes, bool)
            or max_dense_bytes <= 0
        ):
            raise MultiTrainTypeError(
                "max_dense_bytes must be a positive integer or None"
            )
        if self.text and n_components is not None:
            raise MultiTrainPCAError(
                "n_components cannot be used for text classification"
            )

        _validate_datasplits(
            datasplits,
            "classification",
            allow_1d_features=self.text,
            allow_non_numeric_features=self.text,
        )

        # ``pca`` remains the scaler name for API compatibility. The actual PCA
        # reduction is performed once after that scaler has been fitted.
        if pca:
            if pca not in SUPPORTED_SCALERS:
                raise MultiTrainPCAError(f'Supported scalers are {list(SUPPORTED_SCALERS.keys())}, got {pca}')
            pca_scaler = SUPPORTED_SCALERS[pca]
        else:
            pca_scaler = False
        
        gpu_enabled = self.use_gpu and platform.system() != "Darwin"
        model_names, model_list, X_train, X_test, y_train, y_test = _prep_model_names_list(
            datasplits, custom_metric, self.random_state, self.n_jobs,
            self.custom_models, "classification", self.max_iter,
            gpu_enabled, self.device, self.model_params,
        )

        # Vectorizers consume a one-dimensional sequence of documents. Unwrap
        # the common one-column dataframe shape for the user.
        if self.text:
            if isinstance(X_train, pd.DataFrame):
                if X_train.shape[1] != 1:
                    raise MultiTrainTextError(
                        "Text classification requires exactly one feature column"
                    )
                X_train = X_train.iloc[:, 0]
                X_test = X_test.iloc[:, 0]
            elif isinstance(X_train, np.ndarray) and X_train.ndim == 2:
                if X_train.shape[1] != 1:
                    raise MultiTrainTextError(
                        "Text classification requires exactly one feature column"
                    )
                X_train = X_train[:, 0]
                X_test = X_test[:, 0]

        if self.text:
            if pca_scaler:
                raise MultiTrainPCAError(
                    "PCA cannot be used for text classification"
                )
            if not pipeline_dict:
                raise MultiTrainTextError(
                    "Text processing requires pipeline_dict with ngram_range, encoding, max_features, analyzer"
                )
            sparse_train, sparse_test, dense_train, dense_test, dense_names = (
                prepare_text_features(
                    vectorizer,
                    pipeline_dict,
                    X_train,
                    X_test,
                    model_list,
                    max_dense_bytes,
                )
            )
            prepared_train, prepared_test = sparse_train, sparse_test
        else:
            if pipeline_dict:
                raise MultiTrainTextError("Cannot use pipeline_dict without text processing")
            prepared_train, prepared_test = prepare_tabular_features(
                X_train, X_test, pca_scaler, n_components
            )
            dense_train = dense_test = None
            dense_names = set()

        # Every selected model receives every training row. Worker settings
        # change scheduling only; they never partition the dataset.
        completed = run_models(
            model_names,
            model_list,
            prepared_train,
            y_train,
            prepared_test,
            y_test,
            show_train_score,
            "classification",
            self.model_workers,
            self.n_jobs,
            use_gpu=gpu_enabled,
            dense_train=dense_train,
            dense_test=dense_test,
            dense_model_names=dense_names,
        )

        # Score cached predictions so every metric uses the same fitted
        # output without repeating expensive predict calls.
        results = {}
        # Averaging depends on whether the task is binary or multiclass. This is
        # scoring metadata only and is evaluated after every model has finished.
        all_targets = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
        multiclass = len(np.unique(all_targets)) > 2
        for completed_model in completed:
            metric_results = {}
            for metric_name, metric_func in _metrics(
                custom_metric, "classification"
            ).items():
                if metric_name == "roc_auc":
                    # ROC AUC was calculated in ``_fit_model`` from probabilities
                    # or decision scores, never from hard class predictions.
                    if show_train_score:
                        metric_results[f"{metric_name}_train"] = (
                            completed_model.train_roc_auc
                        )
                    metric_results[metric_name] = completed_model.test_roc_auc
                    continue

                if metric_name in {"log_loss", "brier_score_loss"}:
                    # These metrics describe confidence, so passing the
                    # predicted labels here would produce a plausible but wrong score.
                    if show_train_score:
                        metric_results[f"{metric_name}_train"] = (
                            _calculate_probability_metric(
                                metric_name,
                                y_train,
                                completed_model.train_probability,
                                completed_model.model_classes,
                            )
                        )
                    metric_results[metric_name] = _calculate_probability_metric(
                        metric_name,
                        y_test,
                        completed_model.test_probability,
                        completed_model.model_classes,
                    )
                    continue

                average_type = None
                if metric_name in {"precision", "recall", "f1", "jaccard_score"}:
                    # Micro averaging is the explicit imbalanced-data option.
                    # Otherwise multiclass results are weighted by class size,
                    # while binary results retain their positive-class meaning.
                    average_type = (
                        "micro" if imbalanced else ("weighted" if multiclass else "binary")
                    )
                # Scikit-learn orders ``classes_`` consistently with probability
                # columns. Using the final class also supports strings and labels
                # other than the integer 1.
                positive_label = (
                    completed_model.model_classes[-1]
                    if average_type == "binary"
                    and completed_model.model_classes is not None
                    else None
                )
                if show_train_score:
                    metric_results[f"{metric_name}_train"] = _calculate_metric(
                        metric_func,
                        y_train,
                        completed_model.train_prediction,
                        average_type,
                        pos_label=positive_label,
                    )
                metric_results[metric_name] = _calculate_metric(
                    metric_func,
                    y_test,
                    completed_model.test_prediction,
                    average_type,
                    pos_label=positive_label,
                )
            results[completed_model.name] = {
                **metric_results,
                "Time": completed_model.elapsed,
            }
            
        if custom_metric:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                custom_metric=custom_metric,
                return_best_model=return_best_model,
                task="classification",
            )
        else:
            final_dataframe = _display_table(
                results=results,
                sort=sort,
                return_best_model=return_best_model,
                task="classification",
            )
        self._store_fit_artifacts(completed, final_dataframe)
        return final_dataframe
    


@dataclass 
class subMultiClassifier(MultiClassifier):
    """Backward-compatible classifier alias retained for existing users."""

    def __init__(
        self,
        n_jobs: int = 1,
        random_state: int = 42,
        custom_models: Optional[Union[list, dict]] = None,
        max_iter: int = 1000,
        use_gpu: bool = False,
        device: str = '0',
        model_workers: Optional[int] = None,
        model_params: Optional[dict] = None,
    ):
        """Forward legacy constructor arguments to ``MultiClassifier``."""

        super().__init__(
            n_jobs=n_jobs,
            random_state=random_state,
            custom_models=custom_models,
            max_iter=max_iter,
            use_gpu=use_gpu,
            device=device,
            model_workers=model_workers,
            model_params=model_params,
        )
        
    def __post_init__(self):
        """Keep legacy validation messages before running the parent checks."""

        if not isinstance(self.use_gpu, bool):
            raise MultiTrainTypeError(f'Invalid type for use_gpu: expected bool, got {type(self.use_gpu).__name__}. Please provide a boolean value (True or False).')
        
        if not isinstance(self.device, str):
            raise MultiTrainTypeError(f'Invalid type for device: expected str, got {type(self.device).__name__}. Please provide a string value.')
        
        super().__post_init__()
