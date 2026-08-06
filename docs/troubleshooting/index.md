# Troubleshooting

## Start with the diagnostic attributes

```python
print(train.warnings_)
print(train.failures_)
```

`failures_` identifies which estimator failed and whether the problem occurred during preparation, fitting, prediction, probability generation, ROC AUC scoring, or warning handling.

(macos-openmp-errors)=

## macOS OpenMP errors

An import error mentioning `libomp`, `libiomp`, or an OpenMP library usually means the runtime required by LightGBM or XGBoost is missing:

```bash
brew install libomp
python -m pip install --force-reinstall MultiTrain
```

Use a native Python matching the machine architecture. Python 3.8 and 3.9 on newer macOS runners may require Intel-compatible environments because of upstream wheel availability.

## Convergence warnings

A convergence warning does not automatically make predictions invalid. Inspect the measurements and warning details first. If further optimization is appropriate, increase `max_iter`, scale features, or pass model-specific parameters:

```python
train = MultiClassifier(
    custom_models=["LogisticRegression"],
    max_iter=3000,
    model_params={"LogisticRegression": {"C": 0.5}},
)
```

MultiTrain applies training-only scaling to known scale-sensitive estimators, but the dataset and estimator configuration still determine whether an optimizer converges.

## Categorical-column error

If tabular fitting reports categorical values, either encode them automatically or name the encoding for each categorical feature:

```python
split = train.split(frame, "target", auto_cat_encode=True)
```

or

```python
split = train.split(
    frame,
    "target",
    manual_encode={"onehot": ["city"], "label": ["education"]},
)
```

## Missing values remain

Forward fill and backward fill may leave a missing value at the boundary of a partition. Choose a strategy appropriate for the data, clean the values before splitting, or confirm that the requested interpolation can fill every missing position.

## A probability metric is `NaN`

Check whether the fitted classifier provides `predict_proba`. Models without probabilities can still have valid accuracy, precision, recall, F1, and balanced accuracy results.

## Worker process errors

User-provided estimators must be serializable when `model_workers` is greater than one. Test the same run with `model_workers=1`. If sequential execution succeeds, ensure custom classes are defined in importable modules rather than inside transient interactive scopes.

## Out-of-memory errors

Reduce `model_workers`, select fewer estimators, keep sparse text input where possible, and leave the dense text allocation guard enabled. Remember that retained fitted ensembles and probability matrices continue occupying memory after `fit` returns.

## Report a reproducible bug

Include:

- Python and MultiTrain versions;
- operating system and architecture;
- dependency versions;
- the smallest reproducible dataset or generated example;
- constructor and `fit` arguments;
- `warnings_` and `failures_` output;
- the complete traceback for an exception raised before model execution.

Open an issue at [github.com/LOVE-DOCTOR/MultiTrain/issues](https://github.com/LOVE-DOCTOR/MultiTrain/issues).
