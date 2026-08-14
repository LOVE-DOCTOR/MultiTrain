# Prepare tabular data

MultiTrain can create a split from a pandas DataFrame or accept a four-item split created elsewhere.

The snippets in the first five sections assume that `train` is a
`MultiClassifier` or `MultiRegressor` instance and `frame` is a pandas DataFrame
containing the named columns. See the complete, runnable
{doc}`../getting-started/quickstart` before adapting the snippets to your data.

## DataFrame or CSV path

```python
split = train.split(data=frame, target="label")
split = train.split(data="dataset.csv", target="label")
```

The source DataFrame is copied before columns are encoded, filled, or removed.

## Automatically encode categorical features

```python
split = train.split(
    data=frame,
    target="label",
    auto_cat_encode=True,
)
```

Encodings are learned from training rows. A category that appears only in the test partition receives a reserved code instead of changing the training mapping.

## Choose encodings by column

```python
split = train.split(
    data=frame,
    target="label",
    manual_encode={
        "label": ["education", "region"],
        "onehot": ["contract_type"],
    },
)
```

The same column cannot be present in both encoding groups. A target cannot be one-hot encoded because `fit` expects a one-dimensional target.

## Fill missing values

```python
split = train.split(
    data=frame,
    target="label",
    fix_nan_custom={
        "age": "interpolate",
        "city": "ffill",
        "income": "bfill",
    },
)
```

Supported strategies are `ffill`, `bfill`, and `interpolate`. Name every
affected column explicitly. If the requested operation leaves an edge value
missing, MultiTrain uses a fallback learned from the training partition: the
training mode for a categorical column or zero for a numeric column. A column
without a configured strategy is rejected if either partition contains a
missing value.

## Remove unused columns

```python
split = train.split(
    data=frame,
    target="label",
    drop=["row_id", "free_text_notes"],
)
```

Remove identifiers, post-outcome information, or other columns that should not be available to the model. MultiTrain checks that every requested column exists.

## Supply a manual split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
)
results = train.fit((X_train, X_test, y_train, y_test))
```

`fit` validates manual splits for:

- four items in the correct order
- matching feature columns and order
- matching feature and target row counts
- compatible pandas indices
- numeric tabular features
- missing or infinite values
- valid target shape and type
- classification class coverage

Perform any custom preprocessing by fitting it only on `X_train`, then applying the learned transform to `X_test`.
