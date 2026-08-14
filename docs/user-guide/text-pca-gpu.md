# Run text, PCA, and GPU workflows

## Text classification

Create the classifier with `text=True`. The feature partition must contain exactly one text column.

```python
import pandas as pd

from MultiTrain import MultiClassifier

messages = pd.DataFrame(
    {
        "message": [
            "claim your prize now",
            "project meeting at ten",
            "limited offer today",
            "please review the report",
            "winner call this number",
            "lunch has moved to noon",
            "exclusive discount available",
            "the build completed successfully",
            "urgent account reward",
            "can we reschedule tomorrow",
        ],
        "label": ["spam", "ham"] * 5,
    }
)

train = MultiClassifier(
    text=True,
    custom_models=["LogisticRegression", "LinearSVC"],
)

split = train.split(data=messages, target="label")
results = train.fit(
    split,
    vectorizer="tfidf",
    pipeline_dict={
        "ngram_range": (1, 2),
        "encoding": "utf-8",
        "max_features": 5000,
        "analyzer": "word",
    },
)
```

Supported vectorizers are `count` and `tfidf`. The vectorizer is fitted once on training documents, and all selected models share the resulting matrix.

Some estimators require dense text input. Before allocating a dense copy,
MultiTrain estimates its size and compares it with `max_dense_bytes`, which
defaults to one GiB. Increase the limit, or pass `None` to disable it, only after
checking available memory.

## Shared PCA

The `pca` argument names the scaler fitted before PCA. This snippet assumes that
`train` is a tabular classifier or regressor and `split` is its four-item data
split:

```python
results = train.fit(
    split,
    pca="StandardScaler",
    n_components=20,
)
```

Supported scaler names are:

- `StandardScaler`
- `MinMaxScaler`
- `MaxAbsScaler`
- `RobustScaler`
- `Normalizer`
- `QuantileTransformer`
- `PowerTransformer`

An integer `n_components` selects an exact component count. A float strictly between zero and one selects an explained-variance target. PCA is not available in text mode.

## GPU estimators

```python
from MultiTrain import MultiClassifier

train = MultiClassifier(
    use_gpu=True,
    device="0",
    custom_models=["CatBoostClassifier", "XGBClassifier"],
)
```

GPU configuration is applied only to supported external estimators. GPU models run sequentially so they do not compete for the same device. GPU acceleration is disabled on macOS.

GPU support also depends on how CatBoost and XGBoost were installed and on the local driver/runtime configuration. A successful MultiTrain installation does not by itself prove that the machine has a usable GPU runtime.
