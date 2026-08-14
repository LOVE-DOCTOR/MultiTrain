"""Public regression workflow for splitting data, training models, and scoring them.

User-facing validation stays here, while shared preprocessing and model
execution are delegated to the same utility modules used by classification.
"""

from dataclasses import dataclass, field
from numbers import Real
import platform
from typing import Dict, List, Optional, Union
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
class MultiRegressor:
    """Configure and compare MultiTrain's regression estimators.

    ``n_jobs`` limits threads inside each estimator. ``model_workers`` controls
    process-level parallelism across estimators so models do not all claim every
    CPU at the same time. ``custom_models`` accepts built-in names or a mapping
    of user-chosen names to estimator objects, and ``model_params`` applies
    validated overrides to the selected models. Fitted estimators, predictions,
    warnings, and failures remain available after training.

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
        Empty test and training dictionaries retained for a consistent artifact
        layout across classification and regression.
    warnings_ : pandas.DataFrame
        Model-attributed warning categories and messages from the latest fit.
    failures_ : pandas.DataFrame
        Model, execution stage, exception type, and message for failed models.

    Examples
    --------
    >>> from MultiTrain import MultiRegressor
    >>> train = MultiRegressor(
    ...     custom_models=["LinearRegression"],
    ...     n_jobs=1,
    ... )
    """

    n_jobs: int = 1
    random_state: int = 42
    custom_models: Optional[Union[list, dict]] = None
    max_iter: int = 1000
    use_gpu: bool = False
    device: str = '0'
    model_workers: Optional[int] = None
    model_params: Optional[dict] = None
    results_: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)
    models_: dict = field(init=False, default_factory=dict, repr=False)
    predictions_: dict = field(init=False, default_factory=dict, repr=False)
    probabilities_: dict = field(init=False, default_factory=dict, repr=False)
    warnings_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    failures_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    
    def __post_init__(self):
        """Validate configuration before loading data or constructing models."""

        # Python treats bool as an int, so integer configuration values need a
        # separate guard against True and False.
        type_validations = {
            'n_jobs': (self.n_jobs, int),
            'random_state': (self.random_state, int),
            'max_iter': (self.max_iter, int),
            'use_gpu': (self.use_gpu, bool),
            'device': (self.device, str)
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

        logger.debug('MultiTrain regressor initialized')

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
        drop: list = None, # List of columns to drop, e.g., ['column1', 'column2']
    ):
        """Create training and test partitions for a regression problem.

        The source data is validated before it is split. Missing-value rules and
        categorical encoders are then learned from the training partition and
        applied to the test partition, which prevents test data from influencing
        preprocessing.

        Parameters
        ----------
        data : pandas.DataFrame or str
            Source dataframe or path to a CSV file.
        target : str
            Name of the numeric regression target column.
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
            ``(X_train, X_test, y_train, y_test)`` ready for :meth:`fit`.

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
            If the dataset cannot produce a valid holdout split.

        Notes
        -----
        An equivalent four-item tuple created with scikit-learn can be passed
        directly to ``fit`` and is validated before training starts.
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

        _validate_supervised_dataset(dataset, target, "regression")

        _non_auto_cat_encode_error(dataset, auto_cat_encode, manual_encode)
        # Split first so encoders and missing-value rules cannot learn from held-out rows.
        try:
            train_dataset, test_dataset = train_test_split(
                dataset,
                test_size=test_size,
                random_state=random_state
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

        # GPU libraries consume contiguous array-like values efficiently. CPU
        # users keep pandas objects, including useful column names and indices.
        if self.use_gpu:
            return (
                np.asarray(X_train),
                np.asarray(X_test),
                np.asarray(y_train),
                np.asarray(y_test),
            )
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        datasplits: tuple,
        custom_metric: str = None,
        show_train_score: bool = False,
        sort: str = None,
        pca: Union[bool, str] = False,
        return_best_model: Optional[str] = None, # example 'mean_squared_error', 'r2_score', 'mean_absolute_error'
        n_components: Optional[Union[int, float]] = None,
    ):
        """Fit and calculate metrics for every selected regression model.

        MultiTrain validates the split, optionally scales and reduces one shared
        feature representation, trains each estimator on the complete training
        partition, and calculates every result from cached predictions.

        Parameters
        ----------
        datasplits : tuple
            Four items ordered as ``X_train, X_test, y_train, y_test``.
        custom_metric : str or None, default=None
            Name of an additional supported scalar scikit-learn metric.
        show_train_score : bool, default=False
            Include metrics calculated on the training partition.
        sort : str or None, default=None
            Result column used to order the returned dataframe.
        pca : str or False, default=False
            Name of the scaler applied before a shared PCA transformation. Pass
            ``False`` to leave the features unchanged.
        return_best_model : str or None, default=None
            Return only the row with the strongest value for this metric.
        n_components : int, float, or None, default=None
            Number of PCA components, or an explained-variance fraction between
            zero and one.

        Returns
        -------
        pandas.DataFrame
            Metrics for the selected models. The same dataframe is stored in
            :attr:`results_`; fitted estimators, predictions, warnings, and
            failures are stored in the other post-fit attributes.

        Raises
        ------
        MultiTrainTypeError
            If an argument or split item has an unsupported type.
        MultiTrainMetricError
            If a metric name or requested ordering is unsupported.
        MultiTrainPCAError
            If PCA configuration is invalid.

        See Also
        --------
        split : Create and validate a regression train/test partition.
        """
        self._reset_fit_artifacts()
        if custom_metric is not None and not isinstance(custom_metric, str):
            raise MultiTrainTypeError("custom_metric must be a string or None")
        if not isinstance(show_train_score, bool):
            raise MultiTrainTypeError("show_train_score must be a boolean")
        if sort is not None and not isinstance(sort, str):
            raise MultiTrainTypeError("sort must be a string or None")
        if pca is not False and not isinstance(pca, str):
            raise MultiTrainPCAError(
                "pca must be False or the name of a supported scaler"
            )
        if return_best_model is not None and not isinstance(return_best_model, str):
            raise MultiTrainTypeError("return_best_model must be a string or None")

        _validate_datasplits(datasplits, "regression")

        # ``pca`` names the scaler fitted before PCA for compatibility with the
        # original API. Both transforms are shared across every selected model.
        if pca:
            if pca not in SUPPORTED_SCALERS:
                raise MultiTrainPCAError(f'Supported scalers are {list(SUPPORTED_SCALERS.keys())}, got {pca}')
            pca_scaler = SUPPORTED_SCALERS[pca]
        else:
            pca_scaler = False
        
        gpu_enabled = self.use_gpu and platform.system() != "Darwin"
        model_names, model_list, X_train, X_test, y_train, y_test = _prep_model_names_list(
            datasplits, custom_metric, self.random_state, self.n_jobs,
            self.custom_models, "regression", self.max_iter,
            gpu_enabled, self.device, self.model_params,
        )

        prepared_train, prepared_test = prepare_tabular_features(
            X_train, X_test, pca_scaler, n_components
        )
        # Every estimator sees the full prepared training matrix. Parallelism
        # changes scheduling, not which rows a model receives.
        completed = run_models(
            model_names,
            model_list,
            prepared_train,
            y_train,
            prepared_test,
            y_test,
            show_train_score,
            "regression",
            self.model_workers,
            self.n_jobs,
            use_gpu=gpu_enabled,
        )

        # Predictions are cached by the execution layer, so all metrics
        # below describe the exact same output from each fitted estimator.
        results = {}
        for completed_model in completed:
            metric_results = {}
            for metric_name, metric_func in _metrics(
                custom_metric, "regression"
            ).items():
                if show_train_score:
                    metric_results[f"{metric_name}_train"] = _calculate_metric(
                        metric_func,
                        y_train,
                        completed_model.train_prediction,
                    )
                metric_results[metric_name] = _calculate_metric(
                    metric_func,
                    y_test,
                    completed_model.test_prediction,
                )

            if show_train_score:
                # RMSE is derived from the exact MSE already stored above, which
                # avoids another pass through the target and prediction arrays.
                metric_results["root_mean_squared_error_train"] = np.sqrt(
                    metric_results["mean_squared_error_train"]
                )
            metric_results["root_mean_squared_error"] = np.sqrt(
                metric_results["mean_squared_error"]
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
                task='regression'
            )
        else:
            final_dataframe = _display_table(
                results=results, 
                sort=sort, 
                return_best_model=return_best_model,
                task='regression'
            )
        self._store_fit_artifacts(completed, final_dataframe)
        return final_dataframe
    

@dataclass
class subMultiRegressor(MultiRegressor):
    """Backward-compatible regressor alias retained for existing users."""

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
        """Forward legacy constructor arguments to ``MultiRegressor``."""

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
