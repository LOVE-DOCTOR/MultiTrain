# MultiTrain

MultiTrain is a python module for machine built with the aim of assisting you to find the machine learning model that works best on a particular dataset.

## REQUIREMENTS

MultiTrain requires:

## INSTALLATION
Install MultiTrain using:
```
pip install MultiTrain
```

## USAGE
*CLASSIFICATION*

### MultiClassifier
The MultiClassifier is a combination of several classifier estimators in which each of the estimators is fitted on the training data and a pandas dataframe containing evaluation metrics such as accuracy, balanced accuracy, r2 score, f1 score, precision, recall, roc auc score are reported for each of the models. 
```python
#This is a code snippet of how to import the MultiClassifier and the parameters contained in an instance

from MultiTrain import MultiClassifier
train = MultiClassifier(cores=-1, #this parameter works exactly the same as setting n_jobs to -1, this uses all the cpu cores to make training faster
                        random_state=42, #setting random state here automatically sets a unified random state across function imports
                        verbose=True, #set this to True to display the name of the estimators being fitted at a particular time
                        target_class='binary', #Recommended: set this to one of binary or multiclass to allow the library to adjust to the type of classification problem
                        imbalanced=True, #set this parameter to true if you are working with an imbalanced dataset
                        sampling='SMOTE' #set this parameter to any over_sampling, under_sampling or over_under_sampling methods if imbalanced is True
                        )
```
In continuation of the code snippet above, incase you're confused about the different sampling methods you could use after setting imbalanced to True when working on an imalanced dataset, a code snippet is provided below to return a list of all the sampling methods available.

```python
from MultiTrain import MultiClassifier
train = MultiClassifier()
print(train.strategies()) #this line of codes returns all the under sampling, over sampling and over_under sampling methods available for use
```

### Classifier Model Names
To return a list of all models available for training
```python


```
### Split
This function works exactly like the train_test_split function in the scikit-learn framework, 
but it comes with some additional functionalities.
For example, below is a basic code snippet of how you can use the split function
```python


```
If you wish to perform Principal Component Analysis on your dataset for the purpose of dimensionality reduction,
the split function allows you to do that. Check code snippet below
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


**REGRESSION**
```

```



You can only use this code on classification problems
