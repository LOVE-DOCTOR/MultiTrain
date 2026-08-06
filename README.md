![PyPI](https://img.shields.io/pypi/v/MultiTrain?label=pypi%20package)
![Languages](https://img.shields.io/github/languages/top/LOVE-DOCTOR/train-with-models)
![GitHub repo size](https://img.shields.io/github/repo-size/LOVE-DOCTOR/train-with-models)
![GitHub](https://img.shields.io/github/license/LOVE-DOCTOR/train-with-models)
![GitHub Repo stars](https://img.shields.io/github/stars/love-doctor/train-with-models)
![GitHub contributors](https://img.shields.io/github/contributors/love-doctor/train-with-models)
[![Downloads](https://pepy.tech/badge/multitrain)](https://pepy.tech/project/multitrain)
[![python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Windows](https://img.shields.io/badge/Windows-0078D6?&logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?&logo=ubuntu&logoColor=white)
![macOS](https://img.shields.io/badge/mac%20os-0078D6?&logo=macos&logoColor=white)


# CONTRIBUTING
If you wish to make small changes to the codebase, your pull requests are welcome. However, for major changes or ideas on how to improve the library, please create an issue.
# LINKS
- [MultiTrain](#multitrain)
- [Requirements](#requirements)
- [Installation](#installation)
- [Deployment](#deployment)
- [Issues](#issues)
- [Usage](#usage)
    1. [Visualize training results](#visualize-training-results)
    2. [Hyperparameter Tuning](#hyperparameter-tuning)
    - [MultiClassifier(Classification)](#multiclassifier)
        1. [Classifier Model Names](#classifier-model-names)
        2. [Split](#split-classifier)
        3. [Fit](#fit-classifier)
    - [MultiRegressor](#multiregressor)
        1. [Regression Model Names](#regression-model-names)
        2. [Split](#split-regression)
        3. [Fit](#fit-regression)
# MultiTrain

MultiTrain is a python module for machine learning, built with the aim of assisting you to find the machine learning model that works best on a particular dataset.

# REQUIREMENTS

MultiTrain supports Python 3.8 through Python 3.13. Pip selects the newest compatible dependency set for the Python version being used. The Python 3.10 development environment currently uses these versions:

- numpy==2.2.6
- pandas==2.3.3
- scikit-learn==1.7.2
- xgboost==3.0.5
- catboost==1.2.10
- lightgbm==4.7.0
- joblib==1.5.3
- tqdm==4.70.0

These packages are installed automatically when you install MultiTrain. Jupyter, ipywidgets, and seaborn are available through the optional notebook dependencies if you are working through the examples in a notebook.

# INSTALLATION
On macOS, install the OpenMP runtime required by LightGBM and XGBoost first:
```commandline
brew install libomp
```

Install MultiTrain using:
```commandline
pip install MultiTrain
```

# ISSUES
If you experience issues or come across a bug while using MultiTrain, make sure to update to the latest version with
```commandline
pip install --upgrade MultiTrain
```
If that doesn't fix your bug, create an issue in the issue tracker

# USAGE

### MULTICLASSIFIER
The MultiClassifier is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics such as accuracy, balanced accuracy, f1, precision, recall, and roc auc for each of the models.
```python
# This is a code snippet showing how to import MultiClassifier and set its parameters.

from MultiTrain import MultiClassifier
train = MultiClassifier(
    n_jobs=1,           # Give each model one CPU thread
    model_workers=4,    # Train up to four different models at the same time
    random_state=42,    # Ensure reproducibility
    max_iter=1000,      # Maximum number of iterations for models that require it
    custom_models=['LogisticRegression', 'GradientBoostingClassifier']  # Leave this as None to train every available classifier.
)
```

### SPLIT CLASSIFIER
This function operates identically like the scikit-learn framework's train test split function.
However, it has some extra features.
For example, the split method is demonstrated in the code below.

```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv("nameofFile.csv")

split = train.split(
    data=df,
    target="label_column",  # Specify the name of the target column here.
    random_state=42,  # Set a random seed.
    test_size=0.3,  # 0.3 gives you 70% training data and 30% test data.
    auto_cat_encode=True,  # Automatically encode all categorical columns
    fix_nan_custom={'column1': 'ffill', 'column2': 'bfill', 'column3': 'interpolate'},  # Specify columns with the strategies to fill with 
    drop=['unnecessary_column']  # Drop columns that are not needed
)
```

The example above uses automatic encoding. If you want to choose how individual columns are encoded, leave `auto_cat_encode` as `False` and use `manual_encode` instead, as shown next. You cannot use both options in the same call.

#### Encoding categorical columns
In 'manual_encode', you are expected to pass in the type of encoding you want to perform on the columns in your dataset. The only available encoding types for now are 'label' for label encoding and 'onehot' for one hot encoding.


```python

# Automatic encoding
split = train.split(
    data=df,
    target='label_column',
    test_size=0.2,
    auto_cat_encode=True
)

# Label encoding
split = train.split(
    data=df,
    target='label_column',
    test_size=0.2,
    manual_encode={'label': ['column1', 'column2']}
)


# Onehot encoding
split = train.split(
    data=df,
    target='label_column',
    test_size=0.2,
    manual_encode={'onehot': ['column1', 'column2']}
)

# Label and onehot encoding
split = train.split(
    data=df,
    target='label_column',
    test_size=0.2,
    manual_encode={'label': ['column1', 'column2'],
                   'onehot': ['column3', 'column4']}
)
```
#### Filling missing values
With the help of the 'fix_nan_custom' argument, you may quickly fill in missing values.

You would need to supply a dictionary to the argument in order to fill in the missing values. Each preset key in the dictionary must be used as shown in the example below.


```python
# The available strategies are 'ffill', 'bfill', and 'interpolate'.
split = train.split(
    data=df,
    target='label_column',
    test_size=0.2,
    fix_nan_custom={'column1': 'ffill', 'column2': 'bfill', 'column3': 'interpolate'}
)
```


### FIT CLASSIFIER
Now that the dataset has been split using the split method, it is time to train on it using the fit method.
Instead of the standard training in scikit-learn, catboost, or xgboost, this fit method integrates almost all available machine learning algorithms and trains them all on the dataset.
It then returns a pandas dataframe containing the assessment metrics for each model. A basic code example for using the fit function is shown below.
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('file.csv')


split = train.split(data=df,
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column',
                    )

fit = train.fit(
    datasplits=split,
    sort='accuracy',  # Sort the final results by accuracy.
)

# The available metrics to pass into sort are 
# 1. accuracy 2. precision 3. recall 4. f1 5. roc_auc
```
Now, we would be looking at the various ways the fit method can be implemented. 
#### If you used the traditional train_test_split method available in scikit-learn
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from MultiTrain import MultiClassifier
train = MultiClassifier()

df = pd.read_csv('filename.csv')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

datasplits = (X_train, X_test, y_train, y_test)
fit = train.fit(
    datasplits=datasplits,
    show_train_score=True,  # Include the training scores so you can spot overfitting.
    sort='accuracy',  # Sort the resulting dataframe by the best accuracy.
    custom_metric='matthews_corrcoef',  # Add another sklearn classification metric to the table.
    imbalanced=True,  # Use micro averaging for precision, recall, and f1.
)
```

#### If you used the split method provided by the MultiClassifier
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

split = train.split(data=df,
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column',
                    )

fit = train.fit(datasplits=split,
                sort='accuracy',
                show_train_score=True)     
```

`MultiClassifier.split` stratifies by the target so every class is represented in both partitions. It rejects duplicate columns, infinite values, missing targets, continuous targets, and datasets that cannot form a valid stratified split before model training begins.

#### If you're working on an NLP problem
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier(text=True)
df = pd.read_csv('filename.csv')

split = train.split(data=df,
                    test_size=0.2,
                    target='label_column',
                    )

fit = train.fit(datasplits=split,
                sort='accuracy',
                show_train_score=True,
                vectorizer='tfidf',
                pipeline_dict={'ngram_range': (1, 2), 'encoding': 'utf-8', 'max_features': 5000, 'analyzer': 'word'},
                ) 
```

Set `text=True` when you create `MultiClassifier`, not when you call `fit`. Your feature data must contain exactly one text column for this mode.

The vectorizer is fitted once and the same feature matrix is shared by every selected model. Models such as `GaussianNB` need a dense matrix, so MultiTrain checks the allocation before creating it. The default limit is 1 GiB; change `max_dense_bytes` only when you know the machine has enough memory.

#### Returning only the best classifier
Use `return_best_model` when you only need the strongest result for one metric. Do not pass `sort` in the same call because these two options return different kinds of results.

```python
best_model = train.fit(
    datasplits=split,
    return_best_model='f1',
)
```

#### Scaling features and reducing dimensions before training
The `pca` argument keeps its original name for API compatibility. It chooses the scaler used before PCA, and that transformation is fitted once on the training data and shared by every model. The supported values are `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `QuantileTransformer`, and `PowerTransformer`.

MultiTrain also applies training-only standardization inside scale-sensitive linear, SVM, neural-network, and iterative regression models. Sparse inputs are scaled without centering, and regression predictions are returned in the target column's original unit.

```python
fit = train.fit(
    datasplits=split,
    sort='accuracy',
    pca='StandardScaler',
    n_components=20,  # Keep 20 principal components.
)
```

#### Training large datasets
MultiTrain parallelizes across models because the models are independent. `model_workers` controls how many models train in separate processes, while `n_jobs` controls the threads used inside each model. Keeping `n_jobs=1` is usually the best starting point when `model_workers` is greater than one because it prevents every model from trying to use every CPU core at the same time.

```python
train = MultiClassifier(
    n_jobs=1,
    model_workers=4,
    custom_models=[
        'LogisticRegression',
        'RandomForestClassifier',
        'LGBMClassifier',
        'XGBClassifier',
    ],
)

results = train.fit(datasplits=split, show_train_score=True)
```

Leave `model_workers=None` if you want MultiTrain to choose a conservative process count. Setting `model_workers=-1` allows one worker per available CPU allocation, so watch memory use on large datasets. If you intentionally set `n_jobs=-1`, models run one after another to avoid nested parallelism. GPU-backed CatBoost and XGBoost models also run one at a time so they do not compete for the same device.

## MULTIREGRESSOR

The MultiRegressor is a combination of many regression estimators, each of which is fitted on the training data and returns assessment metrics for each of the models.
```python
# This is a code snippet showing how to import MultiRegressor and set its parameters.

from MultiTrain import MultiRegressor
train = MultiRegressor(
    n_jobs=1,           # Give each model one CPU thread
    model_workers=4,    # Train up to four different models at the same time
    random_state=42,    # Ensure reproducibility
    max_iter=1000,      # Maximum number of iterations for models that require it
    custom_models=['LinearRegression', 'GradientBoostingRegressor']  # Leave this as None to train every available regressor.
)
```

### SPLIT REGRESSION
This function operates identically like the scikit-learn framework's train test split function.
However, it has some extra features.
For example, the split method is demonstrated in the code below.
```python
from MultiTrain import MultiRegressor
train = MultiRegressor()
df = pd.read_csv('sample_data.csv')
split = train.split(data=df,
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='target_column',
                    )

```

If you want to fill missing values using the split function
> [Fill missing values](#filling-missing-values)

If you want to encode your categorical columns using the split function
> [Encode categorical columns](#encoding-categorical-columns)

All you need to do is swap out MultiClassifier with MultiRegressor and you're good to go.

### FIT REGRESSION
Now, we would be looking at the various ways the fit method can be implemented. 
#### If you used the traditional train_test_split method available in scikit-learn
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from MultiTrain import MultiRegressor
train = MultiRegressor()

df = pd.read_csv('filename.csv')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

datasplits = (X_train, X_test, y_train, y_test)
fit = train.fit(
    datasplits=datasplits,
    show_train_score=True,  # Include the training scores so you can compare them with the test scores.
    sort='mean_squared_error',  # Lower values appear first for this metric.
    custom_metric='max_error',  # Add another sklearn regression metric to the table.
)

# The metrics available for sorting are 
# mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error,
# median_absolute_error, mean_squared_log_error, and explained_variance_score.
```

When you provide your own split, MultiTrain checks row counts, feature order, pandas index alignment, missing or infinite values, and target types before training. This prevents a malformed split from appearing as a table of failed models.
#### If you used the split method provided by the MultiRegressor
```python
import pandas as pd
from MultiTrain import MultiRegressor

train = MultiRegressor()
df = pd.read_csv('filename.csv')

split = train.split(data=df,
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='target_column',
                    )

fit = train.fit(datasplits=split,
                sort='r2_score',
                show_train_score=True)      
```

If you only want the best regression model, use `return_best_model` without `sort`:

```python
best_model = train.fit(
    datasplits=split,
    return_best_model='mean_squared_error',
)
```

# DEPLOYMENT

The release files are built from `pyproject.toml`, so there is only one source of package metadata. The commands below use Python 3.10 for the release build, while CI tests Python 3.8 through Python 3.13.

Start by creating a clean development environment:

```commandline
python3.10 -m venv .venv
```

Activate it with `.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on Ubuntu and macOS. On macOS, install the OpenMP runtime shown in the [installation instructions](#installation) before continuing.

Then install the development and notebook tools:

```commandline
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

Run the checks before building anything you intend to publish:

```commandline
python -m ruff check MultiTrain
python -m pytest
python -m build
python -m twine check dist/*
```

The test workflow repeats these checks on supported Python versions and operating systems. When a GitHub release is created with a tag matching the package version, the publish workflow builds the distributions again and uploads them to PyPI through trusted publishing. Configure the GitHub repository as a trusted publisher in PyPI before the first release; no API token needs to be stored in the repository.

To inspect a release locally without publishing it, install the wheel into a fresh environment and import the package:

```commandline
python -m pip install dist/multitrain-1.2.0-py3-none-any.whl
python -c "import MultiTrain; print(MultiTrain.__version__)"
```
