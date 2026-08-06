---
html_theme.sidebar_secondary.remove: true
---

<div class="multitrain-hero">

# MultiTrain

<p class="multitrain-tagline">Train and compare multiple classification or regression models through one consistent Python API.</p>

</div>

MultiTrain prepares one training and test split, fits every selected estimator on the complete training data, and returns their measurements in a pandas DataFrame. You decide which measurements matter for your problem; MultiTrain keeps the fitted estimators and their outputs available for further inspection.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/quickstart
:link-type: doc

Install MultiTrain and run a focused classification or regression comparison.
:::

:::{grid-item-card} Read the user guide
:link: user-guide/index
:link-type: doc

Learn splitting, encoding, metrics, custom estimators, artifacts, and parallel execution.
:::

:::{grid-item-card} Browse examples
:link: auto_examples/index
:link-type: doc

Run complete, downloadable examples generated and tested with the documentation.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

See the signatures and source-backed reference for every public class and exception.
:::

::::

## A small example

```python
import pandas as pd
from sklearn.datasets import make_classification

from MultiTrain import MultiClassifier

X, y = make_classification(
    n_samples=300,
    n_features=8,
    n_informative=5,
    random_state=42,
)
data = pd.DataFrame(X, columns=[f"feature_{number}" for number in range(X.shape[1])])
data["target"] = y

train = MultiClassifier(
    custom_models=["LogisticRegression", "RandomForestClassifier"],
    n_jobs=1,
    model_workers=2,
)
split = train.split(data, target="target", random_state=42)
results = train.fit(split, show_train_score=True, sort="accuracy")
```

The returned `results` table is also available as `train.results_`. Fitted estimators live in `train.models_`, while cached predictions, probabilities, warnings, and failures are exposed through the other [post-fit artifacts](user-guide/artifacts.md).

```{toctree}
:caption: Getting started
:hidden:

getting-started/installation
getting-started/quickstart
getting-started/core-concepts
```

```{toctree}
:caption: User guide
:hidden:

user-guide/index
```

```{toctree}
:caption: Examples
:hidden:

auto_examples/index
examples/notebooks
```

```{toctree}
:caption: Reference
:hidden:

api/index
troubleshooting/index
release-notes
development/index
```
