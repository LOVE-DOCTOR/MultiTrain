![PyPI](https://img.shields.io/pypi/v/MultiTrain?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/MultiTrain)
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

# MultiTrain

MultiTrain is a python module for machine learning, built with the aim of assisting you to find the machine learning model that works best on a particular dataset.

## REQUIREMENTS

MultiTrain requires:

- matplotlib==3.5.3
- numpy==1.23.3
- pandas==1.4.4
- plotly==5.10.0
- scikit-learn==1.1.2

## INSTALLATION
Install MultiTrain using:
```commandline
pip install MultiTrain
```

## USAGE
*CLASSIFICATION*

### MultiClassifier
The MultiClassifier is a combination of many classifier estimators, each of which is fitted on the training data and returns assessment metrics such as accuracy, balanced accuracy, r2 score, 
f1 score, precision, recall, roc auc score for each of the models.
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

from MultiTrain import MultiClassifier
train = MultiClassifier(cores=-1, #this parameter works exactly the same as setting n_jobs to -1, this uses all the cpu cores to make training faster
                        random_state=42, #setting random state here automatically sets a unified random state across function imports
                        verbose=True, #set this to True to display the name of the estimators being fitted at a particular time
                        target_class='binary', #Recommended: set this to one of binary or multiclass to allow the library to adjust to the type of classification problem
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

### Classifier Model Names
To return a list of all models available for training
```python
from MultiTrain import MultiClassifier
train = MultiClassifier()
print(train.classifier_model_names())

```
### Split
This function operates identically like the scikit-learn framework's train test split function.
However, it has some extra features.
For example, the split method is demonstrated in the code below.
```python
import pandas as pd
from MultiTrain import MultiClassifier

train = MultiClassifier()
df = pd.read_csv("nameofFile.csv")

features = df.drop("nameOflabelcolum", axis = 1)
labels = df["nameOflabelcolum"]

split = train.split(X=features, 
                    y=labels, 
                    sizeOfTest=0.3, 
                    randomState=42)

```
If you want to run Principal Component Analysis on your dataset to reduce its dimensionality,
You can achieve this with the split function. See the code excerpt below.

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
### Fit
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

fit = train.fit(X=features,
                y=labels,
                splitting=True,
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
              return_best_model=True, #setting this to True means that you'll get a dataframe containing only the best performing model
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
After training on your dataset, it is only normal that you'd want to make use of the best algorithm based on a specific metric. 
A method is also provided for you to do this easily.
Using the code snippet above - after training, to use the best algorithm based on it's name
```python

```
Or else if you want to automatically select the best algorithm based on a particular metric of your choice
```python

```

**REGRESSION**
```

```

