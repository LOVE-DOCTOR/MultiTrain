# Metrics and result ordering

## Default classifier metrics

| Result column | Meaning | Direction when sorting |
| --- | --- | --- |
| `accuracy` | Fraction of labels predicted correctly | Higher first |
| `precision` | Positive prediction precision or multiclass average | Higher first |
| `recall` | Positive-class recall or multiclass average | Higher first |
| `f1` | Harmonic mean of precision and recall | Higher first |
| `roc_auc` | Ranking quality from probability or decision output | Higher first |
| `balanced_accuracy` | Mean recall across classes | Higher first |

Supported additional classifier metrics are:

- `brier_score_loss`
- `cohen_kappa_score`
- `hamming_loss`
- `jaccard_score`
- `log_loss`
- `matthews_corrcoef`
- `zero_one_loss`

```python
results = classifier.fit(
    split,
    custom_metric="matthews_corrcoef",
    sort="accuracy",
)
```

## Default regressor metrics

| Result column | Direction when sorting |
| --- | --- |
| `mean_squared_error` | Lower first |
| `root_mean_squared_error` | Lower first |
| `r2_score` | Higher first |
| `mean_absolute_error` | Lower first |
| `median_absolute_error` | Lower first |
| `mean_squared_log_error` | Lower first |
| `explained_variance_score` | Higher first |

Supported additional regression metrics are:

- `d2_absolute_error_score`
- `d2_pinball_score`
- `d2_tweedie_score`
- `max_error`
- `mean_absolute_percentage_error`
- `mean_gamma_deviance`
- `mean_pinball_loss`
- `mean_poisson_deviance`
- `mean_tweedie_deviance`

## Training metrics

Set `show_train_score=True` to add columns ending in `_train`. A large difference
between a training and test metric can be useful evidence, but interpreting it
depends on the dataset, split, and metric.

## Missing metric values

`NaN` does not always mean the estimator failed. Common examples include:

- a classifier does not expose probabilities required by log loss or Brier score;
- a metric's mathematical domain is incompatible with the supplied target values;
- a metric is undefined for the observed labels.

Inspect {attr}`~MultiTrain.MultiClassifier.failures_` before treating an
unavailable metric as a failed fit.
