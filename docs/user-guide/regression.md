# Regression

{class}`MultiTrain.MultiRegressor` compares estimators that predict continuous numeric targets.

## Create the runner and split

```python
from MultiTrain import MultiRegressor

train = MultiRegressor(
    n_jobs=1,
    model_workers=2,
    random_state=42,
    custom_models=["LinearRegression", "RandomForestRegressor"],
)

split = train.split(
    data=frame,
    target="price",
    test_size=0.2,
    auto_cat_encode=True,
)
```

Regression targets must be numeric and finite. Features and targets are checked again when `fit` receives the split, including manually created scikit-learn splits.

## Fit and order the results

```python
results = train.fit(
    datasplits=split,
    show_train_score=True,
    sort="mean_absolute_error",
)
```

The standard regression table includes:

- mean squared error
- root mean squared error
- R-squared
- mean absolute error
- median absolute error
- mean squared logarithmic error
- explained variance

Loss and error columns are sorted from smaller to larger values. Scores such as R-squared and explained variance are sorted from larger to smaller values.

Some estimators require positive targets or benefit from target scaling. MultiTrain learns reversible target transformations from the training partition and converts predictions back to the original target unit before calculating measurements.

## Continue from the fitted regressor

```python
fitted_regressor = train.models_["RandomForestRegressor"]
predictions = fitted_regressor.predict(new_features)
```

Regression uses `predictions_`, but its `probabilities_` structure remains empty because regressors do not produce class probabilities.
