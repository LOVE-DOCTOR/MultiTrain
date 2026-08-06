# Classification

{class}`MultiTrain.MultiClassifier` compares estimators that predict discrete labels.

## Create the runner

```python
from MultiTrain import MultiClassifier

train = MultiClassifier(
    n_jobs=1,
    model_workers=2,
    random_state=42,
    max_iter=1000,
    custom_models=["LogisticRegression", "RandomForestClassifier"],
)
```

Leave `custom_models=None` to use the complete classifier catalog. Selecting a focused list makes experimentation faster and reduces peak memory.

## Create a classification split

```python
split = train.split(
    data=frame,
    target="label",
    test_size=0.2,
    random_state=42,
    auto_cat_encode=True,
)
```

Classification splits are stratified by the target. MultiTrain rejects continuous targets, a training partition with fewer than two classes, and test labels that do not appear in training.

## Fit and inspect measurements

```python
results = train.fit(
    datasplits=split,
    show_train_score=True,
    sort="balanced_accuracy",
    imbalanced=False,
)
```

The default test measurements are:

- accuracy
- precision
- recall
- F1
- ROC AUC
- balanced accuracy

With `show_train_score=True`, corresponding columns ending in `_train` are added. Binary precision, recall, and F1 use the final fitted class as the positive label. Multiclass values use weighted averaging by default. Set `imbalanced=True` to request micro averaging for precision, recall, and F1.

## Probability-based measurements

`roc_auc`, `log_loss`, and `brier_score_loss` use probability or decision outputs rather than hard class predictions. A classifier without the required output can still receive label-based measurements; unavailable probability measurements become `NaN`.

## Return one result row

```python
one_row = train.fit(split, return_best_model="f1")
```

Do not combine `return_best_model` with `sort`. Sorting returns the complete table in a new order, while `return_best_model` deliberately reduces the table to one row.
