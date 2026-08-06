"""
Compare classifiers and inspect fitted artifacts
=================================================

This example creates a reproducible binary dataset, compares two classifiers,
and then reads the exact fitted output retained by MultiTrain.
"""

import pandas as pd
from sklearn.datasets import make_classification

from MultiTrain import MultiClassifier


# Build a DataFrame so the example follows the same path as a CSV-backed project.
features, target = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42,
)
frame = pd.DataFrame(
    features,
    columns=[f"feature_{number}" for number in range(features.shape[1])],
)
frame["target"] = target

# A small selection keeps the example quick. Both models still receive every
# row in the training partition.
train = MultiClassifier(
    custom_models=["LogisticRegression", "DecisionTreeClassifier"],
    model_params={"DecisionTreeClassifier": {"max_depth": 4}},
    n_jobs=1,
    model_workers=1,
    random_state=42,
)
split = train.split(frame, target="target", random_state=42)
results = train.fit(split, show_train_score=True, sort="accuracy")
print(results)

# The fitted estimators and the predictions used above are available without a
# second fit or predict call.
print("\nFitted models:", list(train.models_))
print(
    "Logistic test prediction shape:",
    train.predictions_["test"]["LogisticRegression"].shape,
)
print("Warnings recorded:", len(train.warnings_))
print("Failures recorded:", len(train.failures_))
