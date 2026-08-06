# `MultiRegressor`

```{eval-rst}
.. currentmodule:: MultiTrain

.. autoclass:: MultiRegressor
   :members: split, fit
   :show-inheritance:
```

## Post-fit attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `results_` | `pandas.DataFrame` or `None` | The DataFrame returned by the latest successful `fit` call. |
| `models_` | `dict` | Successfully fitted regressors keyed by result name. |
| `predictions_` | `dict` | Cached test and optional training predictions. |
| `probabilities_` | `dict` | Empty test and train dictionaries, retained for a consistent artifact layout. |
| `warnings_` | `pandas.DataFrame` | Model-attributed warning category and message rows. |
| `failures_` | `pandas.DataFrame` | Model, stage, exception type, and message rows for failed estimators. |

See {doc}`../user-guide/artifacts` for examples and lifecycle details.
