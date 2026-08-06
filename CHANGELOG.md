# WHAT'S NEW

## 1.2.0

- Modernized packaging around a single `pyproject.toml` source of truth.
- Added tested Python 3.8 through 3.13 support with runtime-specific dependency constraints.
- Split runtime, notebook, and development dependencies.
- Added comprehensive validation, combinatorial, property, and mutation tests.
- Fixed data leakage, metric handling, estimator defaults, and GPU configuration.
- Aligned pandas, scikit-learn, CatBoost, LightGBM, and XGBoost usage with their documented APIs.
- Added pull-request CI, built-artifact verification, and trusted PyPI publishing.
- Removed tests and development tools from the production wheel.
- Added bounded model-level process parallelism while keeping every selected model on the full training dataset.
- Shared PCA, scaling, and text vectorization across model runs to avoid repeated preprocessing.
- Added explicit PCA component sizing, dense-text memory limits, cached predictions, and sequential GPU scheduling.
- Added the required macOS OpenMP runtime setup to CI and deployment instructions.
- Constrained Pyparsing on Python 3.8 and 3.9 to remain compatible with their Matplotlib releases.
- Added leakage-safe feature and target scaling for convergence-sensitive estimators.
- Removed premature iteration limits from libsvm estimators and selected stable MLP and LinearSVR convergence settings.
- Added Penguins and Red Wine Quality datasets to validate multiclass, categorical, missing-value, and regression workflows.
- Added early dataset and manual-split validation for invalid targets, values, schemas, indices, and class coverage.
- Stratified classification splits and added sortable train/test root mean squared error results.
- Expanded mutation testing to cover the shared execution engine and dataset-backed metric oracles.
- Routed probability-based metrics through `predict_proba`, rejected non-scalar metric APIs, and corrected custom-metric ranking directions.
- Added multiclass Brier scores and binary scoring for string or otherwise non-integer class labels.
- Made count-vectorized text floating-point at construction time so LightGBM can consume it without a second full sparse copy.
- Added model-local non-negative feature and positive-target transforms for estimators with stricter mathematical domains.
- Adapted cross-validation folds and neighbor counts to the available training data on small datasets.
- Converted invalid text-vectorization failures into actionable MultiTrain errors.

# 0.13.11
- Removed force_finite parameter in r2 score - regression
- add roc_auc metric in kfold - classification

# 0.13.10
- Fixed bug causing upgrade to fail

# 0.13.0
- Added new parameter 'select_models' that enables you to select only a few models to train with instead of using all models at once.
- Added progress bar when training models
- Added more understandable error message with fixes indicated
- Fixed key error bug in use_model when specifying metric
- Removed r2 score metric from classification

# 0.12.3
- Fixed bug that stopped models from training due to inconsistent number of columns
- Temporarily disabled using over, under or over_under sampling techniques when using kf=True
- Added a new parameter y to 'visualize' and 'show' methods to indicate the target.

# 0.12.0
- If model is unable to properly compute metrics, it's value is replaced with np.nan
- Added 'encode' parameter in split method to encode categorical columns
- Added 'missing' values parameter in split method for filling missing values for both numerical and categorical columns.
# 0.11.21 - BUG FIX
- Added missing requirement for kaleido engine
# 0.11.0 - PATCH RELEASE
- Removed the target_class parameter in the instance of the MultiClass object, the class of your target is automatically checked.
# 0.1.31
- Added support for dimensionality reduction in split method in MultiRegressor
