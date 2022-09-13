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

#It's important to note that when using the split function, it must be assigned to a variable as it returns values.
split = train.split(X=features, #the features of the dataset
                    y=labels,   #the labels of the dataset
                    sizeOfTest=0.2, #same as test_size parameter in train_test_split
                    randomState=42, #initialize the value of the random state parameter
                    dimensionality_reduction=True, #setting to True enables this function to perform PCA on both X_train and X_test automatically after splitting
                    normalize='StandardScaler', #when using dimensionality_reduction, this must be set to one of StandardScaler,MinMaxScaler or RobustScaler if feature columns aren't scaled before a split
                    n_components=2 #when using dimensionality_reduction, this parameter must be set to define the number of components to keep.
                    ) 
```


**REGRESSION**
```

```



You can only use this code on classification problems
