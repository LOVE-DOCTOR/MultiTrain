---
html_theme.sidebar_secondary.remove: true
---

<div class="multitrain-hero">

# MultiTrain

<p class="multitrain-tagline">Train and compare multiple classification or regression models through one consistent Python API.</p>

</div>

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/quickstart
:link-type: doc

Install MultiTrain and run a focused classification or regression comparison.
:::

:::{grid-item-card} Solve a task
:link: user-guide/index
:link-type: doc

Follow focused guides for data preparation, model configuration, fitted outputs, and performance.
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
data = pd.DataFrame(
    X,
    columns=[f"feature_{number}" for number in range(X.shape[1])],
)
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
```

```{toctree}
:caption: How-to guides
:hidden:

user-guide/index
```

```{toctree}
:caption: Explanation
:hidden:

getting-started/core-concepts
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
user-guide/model-catalog
user-guide/metrics
troubleshooting/index
```

```{toctree}
:caption: Project
:hidden:

release-notes
development/index
```
