# Post-fit artifacts

Each successful `fit` call replaces the artifacts from the previous call. If a new call fails during argument or data validation, the previous artifacts are cleared so stale models cannot be mistaken for current output.

## Results

```python
results = train.fit(split)
assert train.results_ is results
```

`results_` is the same DataFrame returned by `fit`, including requested sorting or one-row reduction.

## Fitted models

```python
fitted_model = train.models_["RandomForestClassifier"]
```

Only successful estimators appear in `models_`. A value may be a pipeline or target-transforming wrapper added during safe model preparation. Use the complete retained object for later predictions.

## Predictions

```python
test_predictions = train.predictions_["test"]["RandomForestClassifier"]
```

The structure always contains `test` and `train` dictionaries. Training predictions are generated and stored only when `show_train_score=True`.

## Classification probabilities

```python
test_probabilities = train.probabilities_["test"]["LogisticRegression"]
```

Only classifiers that successfully expose `predict_proba` appear in the probability dictionaries. Probability columns follow the order in the fitted estimator's `classes_` attribute.

Regressors expose the same structure for consistency, but both probability dictionaries are empty.

## Warnings

`warnings_` has three columns:

| Column | Meaning |
| --- | --- |
| `Model` | Name used in the result table |
| `Category` | Python warning category |
| `Message` | Warning message emitted during the model run |

```python
if not train.warnings_.empty:
    print(train.warnings_)
```

## Failures

`failures_` has four columns:

| Column | Meaning |
| --- | --- |
| `Model` | Estimator that failed |
| `Stage` | Preparation, fit, prediction, probability, scoring, or warning handling |
| `Exception` | Original exception class name |
| `Message` | Original exception message |

```python
if not train.failures_.empty:
    print(train.failures_)
```

A failed estimator retains a `NaN` prediction array for alignment and a row of missing measurements, but it does not appear in `models_`.
