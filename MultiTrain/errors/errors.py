class MultiTrainError(Exception):
    """Base class for all errors raised by MultiTrain."""


class MultiTrainDatasetTypeError(MultiTrainError):
    pass


class MultiTrainColumnMissingError(MultiTrainError):
    pass


class MultiTrainModelError(MultiTrainError):
    pass


class MultiTrainEncodingError(MultiTrainError):
    pass


class MultiTrainTypeError(MultiTrainError):
    pass


class MultiTrainNaNError(MultiTrainError):
    pass


class MultiTrainMetricError(MultiTrainError):
    pass


class MultiTrainSplitError(MultiTrainError):
    pass


class MultiTrainTextError(MultiTrainError):
    pass


class MultiTrainPCAError(MultiTrainError):
    pass
