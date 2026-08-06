# Performance and large datasets

## Choose the two parallelism levels deliberately

```python
train = MultiClassifier(
    n_jobs=1,
    model_workers=4,
    custom_models=[
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
    ],
)
```

`model_workers=4` permits four independent model processes. `n_jobs=1` limits supported internal estimator parallelism to one thread per process.

MultiTrain bounds its automatic worker count by available CPUs, selected model count, and estimator thread allocation. `model_workers=-1` allows one worker per available CPU allocation and can use substantial memory. If `n_jobs` is negative, MultiTrain runs models sequentially to prevent nested all-core execution.

## Shared work

The execution layer avoids repeating work that can safely be shared:

- tabular arrays are normalized once;
- scaling and PCA are fitted once when explicitly requested through `pca`;
- text is vectorized once;
- a guarded dense text representation is created only when selected models require it;
- predictions and probabilities are cached once per requested partition.

Model-specific transformations remain inside the estimator pipeline so one model's mathematical requirements do not change another model's input.

## Memory considerations

Retaining `models_`, `predictions_`, and `probabilities_` consumes memory after fitting. Probability matrices can be large for multiclass datasets. Delete the runner or the unneeded artifact entries when a long-lived process no longer needs them.

Process workers also need estimator state and working memory. Begin with one or two workers when:

- the feature matrix is dense and large;
- selected models build large ensembles;
- the dataset has many classes;
- memory is constrained;
- external boosting libraries allocate their own caches.

## Measure the real workload

Use a representative subset to estimate runtime and peak memory, but make the final comparison on the complete intended training partition. Record dataset shape, dependency versions, worker settings, and estimator parameters alongside benchmark results.
