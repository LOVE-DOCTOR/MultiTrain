"""
Compare regressors with custom names
====================================

Estimator dictionaries let a result table use experiment-specific names while
retaining the fitted copies for later prediction.
"""

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MultiTrain import MultiRegressor


features, target = make_regression(
    n_samples=300,
    n_features=8,
    n_informative=6,
    noise=8.0,
    random_state=42,
)
frame = pd.DataFrame(
    features,
    columns=[f"feature_{number}" for number in range(features.shape[1])],
)
frame["target"] = target

train = MultiRegressor(
    custom_models={
        "scaled ridge": make_pipeline(StandardScaler(), Ridge()),
        "small forest": RandomForestRegressor(random_state=42, n_jobs=1),
    },
    model_params={
        "scaled ridge": {"ridge__alpha": 0.5},
        "small forest": {"n_estimators": 100, "max_depth": 8},
    },
    n_jobs=1,
    model_workers=1,
)
split = train.split(frame, target="target", random_state=42)
results = train.fit(split, show_train_score=True, sort="mean_absolute_error")
print(results)

fitted_forest = train.models_["small forest"]
print("\nFitted model names:", list(train.models_))
print("Forest tree count:", fitted_forest.n_estimators)
print("Original target units retained:", train.predictions_["test"]["small forest"][:3])
print("Regression probability outputs:", train.probabilities_)
