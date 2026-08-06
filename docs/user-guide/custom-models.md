# Custom estimators and parameters

## Override built-in parameters

`model_params` is keyed by selected model name. MultiTrain applies the overrides to fresh estimator instances before training.

```python
from MultiTrain import MultiClassifier

train = MultiClassifier(
    custom_models=["LogisticRegression", "RandomForestClassifier"],
    model_params={
        "LogisticRegression": {"C": 0.5},
        "RandomForestClassifier": {
            "n_estimators": 300,
            "max_depth": 12,
        },
    },
)
```

An unknown model name or estimator parameter raises a `MultiTrainModelError` before expensive fitting begins.

## Pass estimator objects

Use a dictionary when you want to provide configured estimators or pipelines. Dictionary keys become result-table and artifact names.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from MultiTrain import MultiClassifier

train = MultiClassifier(
    custom_models={
        "scaled SVC": make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=42),
        ),
        "small tree": DecisionTreeClassifier(random_state=42),
    },
    model_params={
        "scaled SVC": {"svc__C": 0.5},
        "small tree": {"max_depth": 4},
    },
)
```

Nested pipeline parameters use scikit-learn's `step__parameter` syntax. Call `pipeline.get_params()` when you need to inspect the available names.

## Estimator requirements

A supplied object must provide callable `fit` and `predict` methods. Supporting scikit-learn's `get_params` and `set_params` protocol is strongly recommended because it enables cloning and parameter overrides.

MultiTrain copies the supplied object before fitting it. Retrieve the trained copy from `models_`:

```python
results = train.fit(split)
fitted_svc = train.models_["scaled SVC"]
```

The original object remains available for another experiment without learned state from this run.
