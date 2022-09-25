![PyPI](https://img.shields.io/pypi/v/MultiTrain?label=pypi%20package)
![GitHub branch checks state](https://img.shields.io/github/checks-status/LOVE-DOCTOR/train-with-models/main?style=plastic)
![Languages](https://img.shields.io/github/languages/top/LOVE-DOCTOR/train-with-models)
![GitHub repo size](https://img.shields.io/github/repo-size/LOVE-DOCTOR/train-with-models)
![GitHub issues](https://img.shields.io/github/issues/LOVE-DOCTOR/train-with-models)
![GitHub closed issues](https://img.shields.io/github/issues-closed/LOVE-DOCTOR/train-with-models)
![GitHub pull requests](https://img.shields.io/github/issues-pr/LOVE-DOCTOR/train-with-models)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/LOVE-DOCTOR/train-with-models)
![GitHub](https://img.shields.io/github/license/LOVE-DOCTOR/train-with-models)
![GitHub Repo stars](https://img.shields.io/github/stars/love-doctor/train-with-models?style=social)
![GitHub forks](https://img.shields.io/github/forks/love-doctor/train-with-models?style=social)
![GitHub contributors](https://img.shields.io/github/contributors/love-doctor/train-with-models)
[![Downloads](https://pepy.tech/badge/multitrain)](https://pepy.tech/project/multitrain)
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
    - [Classification](#classification)
        > [MultiClassifier](#multiclassifier)
        1. [Classifier Model Names](#classifier-model-names)
        2. [Split](#split-classifier)
        3. [Fit](#fit-classifier)
    - [Regression](#regression)
        > [MultiRegressor](#multiregressor)
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
## CLASSIFICATION

### MULTICLASSIFIER
The MultiClassifier is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics such as accuracy, balanced accuracy, r2 score, 
f1 score, precision, recall, roc auc score for each of the models.
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

#Note: the parameter target_class was removed in version 0.11.0, your dataset is automatically checked
#      for binary labels or multiclass labels
from MultiTrain import MultiClassifier
train = MultiClassifier(cores=-1, #this parameter works exactly the same as setting n_jobs to -1, this uses all the cpu cores to make training faster
                        random_state=42, #setting random state here automatically sets a unified random state across function imports
                        verbose=True, #set this to True to display the name of the estimators being fitted at a particular time
                        imbalanced=True, #set this parameter to true if you are working with an imbalanced dataset
                        sampling='SMOTE', #set this parameter to any over_sampling, under_sampling or over_under_sampling methods if imbalanced is True
                        strategy='auto' #not all samplers use this parameters, the parameter is named as sampling_strategy for the samplers that support,
                                        #read more in the imbalanced learn documentation before using this parameter
                        )
```
In continuation of the code snippet above, if you're unsure about the various sampling techniques accessible after setting imbalanced to True when working on an imbalanced dataset, 
a code snippet is provided below to generate a list of all available sampling techniques.

```python
from MultiTrain import MultiClassifier
train = MultiClassifier()
print(train.strategies()) #this line of codes returns all the under sampling, over sampling and over_under sampling methods available for use
```

### CLASSIFIER MODEL NAMES
To return a list of all models available for training
```python
from MultiTrain import MultiClassifier
train = MultiClassifier()
print(train.classifier_model_names())

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

features = df.drop("nameOflabelcolumn", axis = 1)
labels = df["nameOflabelcolumn"]

split = train.split(X=features, 
                    y=labels, 
                    sizeOfTest=0.3, 
                    randomState=42)

```
If you want to run Principal Component Analysis on your dataset to reduce its dimensionality,
You can achieve this with the split function. See the code excerpt below.
#### Dimensionality Reduction
```python
import pandas as pd
from MultiTrain import MultiClassifier #import the module

train = MultiClassifier()
df = pd.read_csv('NameOfFile.csv')

features = df.drop("nameOfLabelColumn", axis=1)
labels = df['nameOfLabelColumn']
pretend_columns = ['columnA', 'columnB', 'columnC']

#It's important to note that when using the split function, it must be assigned to a variable as it returns values.
split = train.split(X=features, #the features of the dataset
                    y=labels,   #the labels of the dataset
                    sizeOfTest=0.2, #same as test_size parameter in train_test_split
                    randomState=42, #initialize the value of the random state parameter
                    dimensionality_reduction=True, #setting to True enables this function to perform PCA on both X_train and X_test automatically after splitting
                    normalize='StandardScaler', #when using dimensionality_reduction, this must be set to one of StandardScaler,MinMaxScaler or RobustScaler if feature columns aren't scaled before a split
                    n_components=2, #when using dimensionality_reduction, this parameter must be set to define the number of components to keep.
                    columns_to_scale=pretend_columns #pass in a list of the columns in your dataset that you wish to scale 
                    ) 
```
You can also encode your categorical columns with the split function
#### Categorical encoding
It is important to remember that the keys are preset when using the dictionaries in the encode parameter and cannot be modified without causing an error.
```python
# continuation from example code above

# if you want to automatically apply label encoder to all categorical columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    encode='labelencoder'
                    )

# if you want to automatically apply one hot encoder to all categorical columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    encode='onehotencoder'
                    )
# there can also be scenarios whereby you only want to apply the labelencoder on
# selected columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    encode={'labelencoder':['the', 'column', 'names']}
                    )

# if you want to apply onehot encoder on selected columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    encode={'onehotencoder': ['the', 'column', 'names']}
                    )

# you can also use both label and onehotencoder together
# this is used when there are columns you want to label encode
# and columns you want to onehotencode in the same dataset
columns_to_encode = {'labelencoder': ['column1', 'column2', 'column3'],
                     'onehotencoder': ['column4', 'column5', 'column6']}
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    encode=columns_to_encode
                    )
```
#### Filling missing values
With the help of the "missing values" argument, you may quickly fill in missing values.

You would need to supply a dictionary to the argument in order to fill in the missing values. Each preset key in the dictionary must be used as shown in the example below.

It's important to remember that the categorical columns are represented by the key "cat" and their corresponding value is the method for filling all of the categorical columns.
The method to fill all numerical columns is represented by the key "num," which stands in for all numerical columns.
```python
# the three strategies available to fill missing values are ['most_frequent', 'mean', 'median']

# only fill categorical columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    missing_values={'cat': 'most_frequent'}
                    )

# only fill numerical columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    missing_values={'num': 'mean'}
                    )

# fill both categorical and numerical columns
split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    missing_values={'cat': 'most_frequent', 'num': 'most_frequent'}
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

features = df.drop("nameOflabelcolumn", axis = 1)
labels = df["nameOflabelcolumn"]

split = train.split(X=features, 
                    y=labels, 
                    sizeOfTest=0.3, 
                    randomState=42,
                    strat=True,
                    shuffle_data=True)

fit = train.fit(splitting=True,
                split_data=split)
```
Now, we would be looking at the various ways the fit method can be implemented. 
#### If you used the traditional train_test_split method available in scikit-learn
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from MultiTrain import MultiClassifier
train = MultiClassifier()

df = pd.read_csv('filename.csv')

features = df.drop('labelName', axis=1)
labels = df['labelName']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
fit = train.fit(X_train=X_train, 
              X_test=X_test, 
              y_train=y_train, 
              y_test=y_test, 
              split_self=True, #always set this to true if you used the traditional train_test_split
              show_train_score=True, #only set this to true if you want to compare train equivalent of all the metrics shown on the dataframe
              return_best_model='Accuracy', #Set a metric here to sort the resulting dataframe by the best performing model based on the metric 
              excel=True #when this parameter is set to true, an spreadsheet report of the training is stored in your current working directory
              ) 
```
#### If you used the split method provided by the MultiClassifier
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

features = df.drop('labelName', axis=1)
labels = df['labelName']

split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    randomState=42,
                    shuffle_data=True)

fit = train.fit(splitting=True,
                split_data=split,
                show_train_score=True,
                excel=True)     
```
#### If you want to train on your dataset with KFold
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

features = df.drop('labelName', axis=1)
labels = df['labelName']

fit = train.fit(X=features,
                y=labels,
                kf=True, #set this to true if you want to train on your dataset with KFold
                fold=5, #you can adjust this to use any number of folds you want for kfold, higher numbers leads to higher training times
                show_train_score=True,
                excel=True)
```
#### If you're working on an NLP problem
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv('filename.csv')

features = df.drop('LabelName', axis=1)
labels = df['labelName']

data_split = train.split(X=features,
                         y=labels,
                         sizeOfTest=0.2,
                         randomState=42)

fit = train.fit(splitting=True,
                split_data=data_split,
                show_train_score=True,
                excel=True,
                text=True, #setting text to True lets the method know you're working on NLP
                vectorizer='count', #set this to one of 'count' or 'tfidf' when text is True
                ngrams=(1,3) #this defines the sequence of N words
 )
```
### USE BEST MODEL
After training on your dataset, it is only normal that you'd want to make use of the best algorithm based on a specific metric. 
A method is also provided for you to do this easily.
Continuing from any of the code snippets above(for the fit method) - after training, to use the best algorithm based on it's name

```python
mod=train.use_best_model(df=fit, model='LogisticRegression')
```
Or else if you want to automatically select the best algorithm based on a particular metric of your choice
```python
mod=train.use_best_model(df=fit, best='Balanced Accuracy')
```
### VISUALIZE TRAINING RESULTS
It gets interesting. After model training, it is obvious that you get a dataframe containing all algorithms and their performance.
What if you could visualize this dataframe instead and even save all the plots to your directory?
Check the code snippet below to see how

Note: In order to visualize your model training results, you must have passed the fit method into a variable.
#### If you want to visualize the plots with matplotlib
```python
#this code is a continuation of the implementations of the fit method above

#if you only want to visualize the results in your notebook, use this code
train.visualize(param=fit, #this parameter takes in the dataframe of the training results 
                t_split=True, #set t_split to true here if you split your data with the split method provided by MultiTrain
                kf=False, #set kf to True here if you used KFold split to train, note t_split and kf can't be set to True at the same time
                size=(15,8) #this sets the size of each plots to be displayed in your notebook
                )

#if you want to visualize the results in your notebook and save the plots to your system
train.visualize(param=fit,
                t_split=True,
                size=(15,8),
                file_path='C:/Users/lenovo/', #you can set your own filepath here)
                save='png', #you can choose to set this parameter to either 'png' or 'pdf'
                save_name='dir1'
                )

# the value set to save_name becomes the name of the pdf file if you set save='pdf'
# the value set to save_name becomes the name of a folder created to accommodate the png file if you set save='png'

```
#### If you want to visualize the plots with plotly
Plotly unlike matplotlib provides you with interactive plots. The code syntax is exactly the same with the visualize function.
The only exception is that you need to use train.show() instead of train.visualize()

### HYPERPARAMETER TUNING
After training the MultiClassifier on your dataset and you have selected a model you wish to work with, you can perform hyperparameter tuning
on such model. 
All parameters available to use
```python
tune_parameters(self,
                model: str = None,
                parameters: dict = None,
                tune: str = None,
                use_cpu: int = None,
                cv: int = 5,
                n_iter: any = 50,
                return_train_score: bool = False,
                refit: bool = True,
                random_state: int = None,
                factor: int = 3,
                verbose: int = 4,
                resource: any = "n_samples",
                max_resources: any = "auto",
                min_resources_grid: any = "exhaust",
                min_resources_rand: any = "smallest",
                aggressive_elimination: any = False,
                error_score: any = np.nan,
                pre_dispatch: any = "2*n_jobs",
                optimizer_kwargs: any = None,
                fit_params: any = None,
                n_points: any = 1,
                score='accuracy')
     
```
The different hyper parameter tuning methods available are listed below
```text
1. GridSearchCV represented with 'grid'
2. RandomizedSearchCV represented with 'random'
3. BayesSearchCV represented with 'bayes'
4. HalvingGridSearchCV represented with 'half-grid'
5. HalvingRandomizedSearchCV represented with 'half-random'
```
Let's train a new model and then perform hyperparameter tuning on it.
```python
import pandas as pd
from MultiTrain import MultiClassifier

df = pd.read_csv('data.csv')
features = df.drop('thelabels', axis=1)
labels = df['thelabels']

train = MultiClassifier(random_state=42,
                        verbose=True)

fit = train.fit(X=features,
                y=labels,
                kf=True,
                fold=5)

mod = train.use_best_model(df=fit, best='Balanced Accuracy')

param = {'random_state': [1, 2, 3, 4]} #remember to set your own parameters here

#using grid search
tuned_model_grid = train.tune_parameters(model=mod,
                                    parameters=param,
                                    use_cpu=-1, #uses all cores of the cpu
                                    tune='grid',
                                    cv=5)

#using random search
tuned_model_random = train.tune_parameters(model=mod,
                                           parameters=param,
                                           use_cpu=-1, #uses all cores of the cpu
                                           tune='random',
                                           cv=5)
```
Notice how you only had to had to change the value of tune to use another hyperparameter tuning algorithm. That's the simplicity MultiTrain provides you.
## REGRESSION
### MULTIREGRESSOR

The MultiRegressor is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics for each of the models.
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

from MultiTrain import MultiRegressor
train = MultiRegressor(cores=-1, #this parameter works exactly the same as setting n_jobs to -1, this uses all the cpu cores to make training faster
                       random_state=42, #setting random state here automatically sets a unified random state across function imports
                       verbose=True #set this to True to display the name of the estimators being fitted at a particular time
                      )
```
### REGRESSION MODEL NAMES
To return a list of all models available for training
```python
from MultiTrain import MultiRegressor
train = MultiRegressor()
print(train.regression_model_names())

```
### SPLIT REGRESSION
This function operates identically like the scikit-learn framework's train test split function.
However, it has some extra features.
For example, the split method is demonstrated in the code below.
```python
from MultiTrain import MultiRegressor
train = MultiRegressor()
df = pd.read_csv("FileName.csv")
X = df.drop("LabelColumn", axis = 1)
y = df["LabelColumn"]
split = train.split(X=X, 
                    y=y, 
                    sizeofTest=0.3, 
                    random_state = 42,  
                    strat = True, 
                    shuffle_data=True)
```

If you also want to perform dimensionality reduction using the split function, refer to this link 
> [Dimensionality reduction](#dimensionality-reduction)

If you want to fill missing values using the split function
> [Fill missing values](#filling-missing-values)

If you want to encode your categorical columns using the split function
> [Encode categorical columns](#categorical encoding)

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

features = df.drop('labelName', axis=1)
labels = df['labelName']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
fit = train.fit(X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test, 
                split_self=True, #always set this to true if you used the traditional train_test_split
                show_train_score=True, #only set this to true if you want to compare train equivalent of all the metrics shown on the dataframe
                return_best_model=None, #Set a metric here to sort the resulting dataframe by the best performing model based on the metric
                excel=True #when this parameter is set to true, an spreadsheet report of the training is stored in your current working directory
              ) 
```
#### If you used the split method provided by the MultiRegressor
```python
import pandas as pd
from MultiTrain import MultiRegressor

train = MultiRegressor()
df = pd.read_csv('filename.csv')

features = df.drop('labelName', axis=1)
labels = df['labelName']

split = train.split(X=features,
                    y=labels,
                    sizeOfTest=0.2,
                    randomState=42,
                    shuffle_data=True)

fit = train.fit(splitting=True,
                split_data=split,
                show_train_score=True,
                excel=True)     
```
#### If you want to train on your dataset with KFold
```python
import pandas as pd
from MultiTrain import MultiRegressor

train = MultiRegressor()
df = pd.read_csv('filename.csv')

features = df.drop('labelName', axis=1)
labels = df['labelName']

fit = train.fit(X=features,
                y=labels,
                kf=True, #set this to true if you want to train on your dataset with KFold
                fold=5, #you can adjust this to use any number of folds you want for kfold, higher numbers leads to higher training times
                show_train_score=True,
                excel=True)
```





