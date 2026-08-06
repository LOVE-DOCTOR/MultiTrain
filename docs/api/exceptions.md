# Exceptions

Every package-specific exception inherits from `MultiTrainError`, allowing callers to catch either one precise failure category or all MultiTrain failures.

```{eval-rst}
.. currentmodule:: MultiTrain.errors.errors

.. autoexception:: MultiTrainError
.. autoexception:: MultiTrainDatasetTypeError
.. autoexception:: MultiTrainDatasetValueError
.. autoexception:: MultiTrainColumnMissingError
.. autoexception:: MultiTrainModelError
.. autoexception:: MultiTrainEncodingError
.. autoexception:: MultiTrainTypeError
.. autoexception:: MultiTrainNaNError
.. autoexception:: MultiTrainMetricError
.. autoexception:: MultiTrainSplitError
.. autoexception:: MultiTrainTextError
.. autoexception:: MultiTrainPCAError
```

## Catching validation failures

```python
from MultiTrain import MultiClassifier, MultiTrainError

try:
    split = MultiClassifier().split(frame, target="label")
except MultiTrainError as error:
    print(f"MultiTrain rejected the input: {error}")
```

Model-specific training failures are usually retained in `failures_` so unrelated selected models can finish. Invalid public arguments and invalid shared data raise exceptions before model execution starts.
