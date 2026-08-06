# Quickstart

This page shows the shortest complete workflows. The detailed behavior of every option is covered in the {doc}`../user-guide/index`.

## Classification

```python
import pandas as pd
from MultiTrain import MultiClassifier

data = pd.read_csv("customers.csv")

train = MultiClassifier(
    custom_models=[
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
    ],
    n_jobs=1,
    model_workers=3,
    random_state=42,
)

split = train.split(
    data=data,
    target="will_cancel",
    test_size=0.2,
    auto_cat_encode=True,
    fix_nan_custom={"age": "interpolate"},
)

results = train.fit(
    datasplits=split,
    show_train_score=True,
    sort="accuracy",
)
print(results)
```

## Regression

```python
import pandas as pd
from MultiTrain import MultiRegressor

data = pd.read_csv("houses.csv")

train = MultiRegressor(
    custom_models=[
        "LinearRegression",
        "RandomForestRegressor",
        "LGBMRegressor",
    ],
    n_jobs=1,
    model_workers=3,
    random_state=42,
)

split = train.split(
    data=data,
    target="price",
    test_size=0.2,
    auto_cat_encode=True,
)

results = train.fit(
    datasplits=split,
    show_train_score=True,
    sort="mean_absolute_error",
)
print(results)
```

## Continue with a fitted estimator

`fit` returns measurements, not a replacement for the fitted models. Choose a model after inspecting the table and retrieve it by name:

```python
fitted_model = train.models_["RandomForestRegressor"]
new_predictions = fitted_model.predict(new_features)
```

If MultiTrain added a preprocessing wrapper for that estimator, `models_` contains the complete fitted wrapper. Pass new data in the same feature order and representation used during training.
