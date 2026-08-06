# API reference

The API reference is generated from MultiTrain's public Python objects, so signatures and source links follow the installed version.

```{toctree}
:maxdepth: 2

classifier
regressor
exceptions
```

## Public imports

```python
from MultiTrain import MultiClassifier, MultiRegressor
```

The legacy `subMultiClassifier` and `subMultiRegressor` names remain importable for compatibility. New code should use the primary classes.

Internal functions beginning with an underscore are implementation details. They may change without forming part of the public compatibility contract.
