"""Exception hierarchy used to give callers precise, catchable failures."""


class MultiTrainError(Exception):
    """Base class for all errors raised by MultiTrain."""


class MultiTrainDatasetTypeError(MultiTrainError):
    """Raised when a dataset or target has an unsupported data type."""


class MultiTrainDatasetValueError(MultiTrainError):
    """Raised when dataset values are missing, infinite, duplicated, or invalid."""


class MultiTrainColumnMissingError(MultiTrainError):
    """Raised when an operation names a column that is not present."""


class MultiTrainModelError(MultiTrainError):
    """Raised when a requested model name or model selection is invalid."""


class MultiTrainEncodingError(MultiTrainError):
    """Raised when categorical encoding is missing or configured incorrectly."""


class MultiTrainTypeError(MultiTrainError):
    """Raised when a public argument has the wrong Python type."""


class MultiTrainNaNError(MultiTrainError):
    """Raised when missing values cannot be accepted or resolved safely."""


class MultiTrainMetricError(MultiTrainError):
    """Raised when a metric is unsupported, unavailable, or cannot be ranked."""


class MultiTrainSplitError(MultiTrainError):
    """Raised when train and test partitions cannot be created or validated."""


class MultiTrainTextError(MultiTrainError):
    """Raised when text input or vectorizer configuration is invalid."""


class MultiTrainPCAError(MultiTrainError):
    """Raised when PCA or its scaler/component configuration is invalid."""
