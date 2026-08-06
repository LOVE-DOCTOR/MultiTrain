# User guide

The user guide explains how each part of a MultiTrain run fits together.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Classification
:link: classification
:link-type: doc
Splitting, fitting, probability metrics, and classification-specific options.
:::

:::{grid-item-card} Regression
:link: regression
:link-type: doc
Regression targets, loss measurements, sorting, and fitted regressors.
:::

:::{grid-item-card} Data preparation
:link: data-preparation
:link-type: doc
Categorical columns, missing values, manual splits, and validation.
:::

:::{grid-item-card} Configure models
:link: custom-models
:link-type: doc
Built-in selections, estimator dictionaries, pipelines, and parameter overrides.
:::

:::{grid-item-card} Inspect fitted outputs
:link: artifacts
:link-type: doc
Models, predictions, probabilities, warnings, failures, and result tables.
:::

:::{grid-item-card} Large datasets
:link: performance
:link-type: doc
Model processes, estimator threads, shared transforms, memory, and GPUs.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

classification
regression
data-preparation
model-catalog
metrics
custom-models
artifacts
text-pca-gpu
performance
```
