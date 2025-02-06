![PyPI](https://img.shields.io/pypi/v/MultiTrain?label=pypi%20package)
![Languages](https://img.shields.io/github/languages/top/LOVE-DOCTOR/train-with-models)
![GitHub repo size](https://img.shields.io/github/repo-size/LOVE-DOCTOR/train-with-models)
![GitHub](https://img.shields.io/github/license/LOVE-DOCTOR/train-with-models)
![GitHub Repo stars](https://img.shields.io/github/stars/love-doctor/train-with-models)
![GitHub contributors](https://img.shields.io/github/contributors/love-doctor/train-with-models)
[![Downloads](https://pepy.tech/badge/multitrain)](https://pepy.tech/project/multitrain)
[![python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![Windows](https://img.shields.io/badge/Windows-0078D6?&logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?&logo=ubuntu&logoColor=white)
![macOS](https://img.shields.io/badge/mac%20os-0078D6?&logo=macos&logoColor=white)


# CONTRIBUTING
If you wish to make small changes to the codebase, your pull requests are welcome. However, for major changes or ideas on how to improve the library, please create an issue.
# LINKS
- [MultiTrain](#multitrain)
- [Requirements](#requirements)
- [Installation](#installation)
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

MultiTrain requires:

- matplotlib==3.5.3
- numpy==1.23.3
- pandas==1.4.4
- plotly==5.10.0
- scikit-learn==1.1.2
- xgboost==1.6.2
- catboost==1.0.6
- imbalanced-learn==0.9.1
- seaborn==0.12.0
- lightgbm==3.3.2
- scikit-optimize==0.9.0

# INSTALLATION
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
The MultiClassifier is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics such as accuracy, balanced accuracy, r2 score, f1 score, precision, recall, roc auc score for each of the models.
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

from MultiTrain import MultiClassifier
train = MultiClassifier(
    n_jobs=-1,          # Use all available CPU cores
    random_state=42,    # Ensure reproducibility
    max_iter=1000,      # Maximum number of iterations for models that require it
    custom_models=['LogisticRegression', 'GradientBoostingClassifier'] # If nothing is set here, all available classifiers will be used for training
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
    target="label_column", # Specify the name of the target column here
    random_state=42, # Set a random seed
    test_size=0.3, # Set the test size to be used for splitting the dataset i.e 0.3 = 70% train, 30% test
    auto_cat_encode=True,  # Automatically encode all categorical columns
    manual_encode={'label': ['cat_feature'], 'onehot': ['city', 'country']},  # Optional manual encoding for select columns (You can't use this with auto_cat_encode)
    fix_nan_custom={'column1': 'ffill', 'column2': 'bfill', 'column3': 'interpolate'},  # Specify columns with the strategies to fill with 
    drop=['unnecessary_column']  # Drop columns that are not needed
)
```

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
# the three strategies available to fill missing values are ['ffill', 'bfill', 'interpolate']

```python
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
It then returns a pandas dataframe including information such as which algorithm is overfitting, which algorithm has the greatest accuracy, and so on. A basic code example for using the fit function is shown below.
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('file.csv')


split = train.split(data=df
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column'
                    )

fit = train.fit(
    datasplits=split,
    sort='accuracy', # The metric to sort the final results
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
fit = train.fit(datasplits=datasplit
              show_train_score=True, # Only set this to true if you want to compare train equivalent of all the metrics shown on the dataframe
              sort='accuracy', # Set a metric here to sort the resulting dataframe by the best performing model based on the metric
              custom_metric='log_loss', # If you set a custom metric here, it will be added to the list of metrics displayed on the final table
              imbalanced=True, # Only set this to true if you're working with an imbalanced dataset. It adjust metrics calculation for imbalanced data
              text=True, # Set this to true if you're working with NLP
              vectorizer= 'count', # specify either count or tfidf if you set text to True
              pipeline_dict = {'ngram_range': (1, 2), 'encoding': 'utf-8', 'max_features': 5000, 'analyzer': 'word'} # You must pass in a similar dictionary also if you set text to True
              return_best_model = 'f1' # If you set this, it will return the single best performing model based on the f1 score metric
              ) 
```
#### If you used the split method provided by the MultiClassifier
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

split = train.split(data=df
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column'
                    )

fit = train.fit(datasplits=split,
                sort='accuracy',
                show_train_score=True)     
```
#### If you're working on an NLP problem
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

split = train.split(data=df
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column'
                    )

fit = train.fit(datasplits=split,
                sort='accuracy',
                show_train_score=True,
                text=True,
                vectorizer='tfidf',
                pipeline_dict = {'ngram_range': (1, 2), 'encoding': 'utf-8', 'max_features': 5000, 'analyzer': 'word'}
                ) 
```

## MULTIREGRESSOR

The MultiRegressor is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics for each of the models.
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

from MultiTrain import MultiRegressor
train = MultiRegressor(
    n_jobs=-1,          # Use all available CPU cores
    random_state=42,    # Ensure reproducibility
    max_iter=1000,      # Maximum number of iterations for models that require it
    custom_models=['LogisticRegression', 'GradientBoostingClassifier'] # If nothing is set here, all available classifiers will be used for training
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
split = train.split(data=df
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column'
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
fit = train.fit(datasplits=datasplit
              show_train_score=True, # Only set this to true if you want to compare train equivalent of all the metrics shown on the dataframe
              sort='mean_squared_error', # Set a metric here to sort the resulting dataframe by the best performing model based on the metric
              custom_metric='r2_score', # If you set a custom metric here, it will be added to the list of metrics displayed on the final table
              return_best_model = 'mean_squared_error' # If you set this, it will return the single best performing model based on the mean squared error metric
              ) 

# The metrics available for sorting are 
# mean squared error, r2 score, mean absolute error, median absolute error, mean squared log error, explained variance score
```
#### If you used the split method provided by the MultiRegressor
```python
import pandas as pd
from MultiTrain import MultiRegressor

train = MultiRegressor()
df = pd.read_csv('filename.csv')

split = train.split(data=df
                    test_size=0.2,
                    auto_cat_encode=True,
                    target='label_column'
                    )

fit = train.fit(datasplits=split,
                sort='r2 score',
                show_train_score=True)      
```