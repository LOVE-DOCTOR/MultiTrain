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
import pandas as pd

from MultiTrain import MultiClassifier, MultiTrainError

frame = pd.DataFrame({"feature": [1, 2, 3, 4], "label": [0, 0, 1, 1]})

try:
    split = MultiClassifier().split(frame, target="missing_label")
except MultiTrainError as error:
    print(f"MultiTrain rejected the input: {error}")
```

Model-specific training failures are usually retained in `failures_` so unrelated selected models can finish. Invalid public arguments and invalid shared data raise exceptions before model execution starts.
