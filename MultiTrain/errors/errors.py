class MultiTrainDatasetTypeError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainColumnMissingError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainEncodingError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainTypeError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainNaNError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainMetricError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainSplitError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class MultiTrainTextError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MultiTrainError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
