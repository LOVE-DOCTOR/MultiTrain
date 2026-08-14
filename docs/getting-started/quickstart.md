# Quickstart

In this tutorial, you will create generated datasets, compare classification and
regression models, and inspect fitted output. The workflow runs without
downloading data or creating local files.

## Before you begin

Install MultiTrain in a virtual environment by following {doc}`installation`.
The example uses only packages installed with MultiTrain.

## Create a classification dataset

Generate a reproducible binary classification dataset and store it in a pandas
DataFrame:

```python
import pandas as pd
from sklearn.datasets import make_classification

from MultiTrain import MultiClassifier

features, target = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=5,
    random_state=42,
)
data = pd.DataFrame(
    features,
    columns=[f"feature_{number}" for number in range(features.shape[1])],
)
data["target"] = target
```

## Select the models

Create a runner with two models. `n_jobs=1` gives each model one estimator
thread, while `model_workers=2` permits the two models to run concurrently:

```python
train = MultiClassifier(
    custom_models=["LogisticRegression", "RandomForestClassifier"],
    n_jobs=1,
    model_workers=2,
    random_state=42,
)
```

## Split and fit the data

Create a stratified holdout split, fit both models, and order the result by test
accuracy:

```python
split = train.split(data, target="target", random_state=42)
results = train.fit(
    split,
    show_train_score=True,
    sort="accuracy",
)
print(results)
```

The result contains one row per model. Its columns include test and training
metrics such as `accuracy`, `f1`, and `balanced_accuracy`. Exact values can vary
between compatible dependency versions, but the table should contain two rows
and `train.failures_` should be empty.

## Inspect a fitted model

The result table contains metrics; `models_` contains the fitted estimators. Use
the model name shown in the result table to retrieve one:

```python
fitted_forest = train.models_["RandomForestClassifier"]
test_predictions = train.predictions_["test"]["RandomForestClassifier"]

print(type(fitted_forest).__name__)
print(test_predictions.shape)
print(train.warnings_)
print(train.failures_)
```

The prediction count matches the number of rows in `split[1]`. MultiTrain
captures model-attributed warnings separately from failures, so a warning does
not automatically discard a fitted estimator.

## Run a regression comparison

```python
import pandas as pd
from sklearn.datasets import make_regression

from MultiTrain import MultiRegressor

features, target = make_regression(
    n_samples=300,
    n_features=8,
    n_informative=6,
    noise=8.0,
    random_state=42,
)
data = pd.DataFrame(
    features,
    columns=[f"feature_{number}" for number in range(features.shape[1])],
)
data["target"] = target

train = MultiRegressor(
    custom_models=["LinearRegression", "RandomForestRegressor"],
    n_jobs=1,
    model_workers=2,
    random_state=42,
)
split = train.split(data, target="target", random_state=42)

results = train.fit(
    split,
    show_train_score=True,
    sort="mean_absolute_error",
)
print(results)
```
