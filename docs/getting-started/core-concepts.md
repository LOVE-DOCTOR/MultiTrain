# Core concepts

## Every selected model receives the complete training data

MultiTrain compares independent estimators. Process-level parallelism changes when models run, but it does not divide the dataset between them. Every selected estimator receives the same prepared `X_train` and `y_train` values.

## The holdout data stays separate

`split` creates the training and test partitions before learning missing-value replacements or categorical encodings. Shared PCA and scaling are fitted on training features and then applied to test features. This prevents test values from influencing the learned transformation.

## Measurements are calculated from cached outputs

Each estimator is fitted once. MultiTrain caches label predictions, available probabilities, and ROC AUC inputs, then calculates the result columns from those cached values. This prevents repeated prediction calls from producing inconsistent measurements.

## MultiTrain does not decide for you

Sorting changes the order of the table. `return_best_model` can return the row with the strongest value for a requested metric, but MultiTrain does not claim that one measurement makes a model appropriate for deployment. Validation strategy, operational cost, interpretability, fairness, and domain constraints remain part of your decision.

## Two levels of parallelism

- `model_workers` controls how many different estimators run in separate processes.
- `n_jobs` controls supported parallel work inside one estimator.

When `model_workers` is greater than one, begin with `n_jobs=1`. This avoids multiplying processes by estimator threads and exhausting the machine.

## Failures remain model-specific

One estimator can fail while others complete. Its measurement row contains missing values, and the details are recorded in `failures_`. Ordinary warnings are recorded in `warnings_` and remain visible without automatically invalidating successful output.
