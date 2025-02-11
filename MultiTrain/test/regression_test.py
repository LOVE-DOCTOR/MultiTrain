import pytest
import pandas as pd
import numpy as np
from MultiTrain.regression.regression_models import MultiRegressor, subMultiRegressor
from MultiTrain.errors.errors import (
    MultiTrainDatasetTypeError,
    MultiTrainColumnMissingError,
    MultiTrainEncodingError,
    MultiTrainError,
    MultiTrainTypeError,
    MultiTrainNaNError,
    MultiTrainPCAError,
    MultiTrainSplitError,
)
import logging


@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'target': [10.5, 12.3, 11.2, 13.4, 10.9]
    })
    target = 'target'
    regressor = MultiRegressor(custom_models=['LinearRegression'])
    return data, target, regressor


def test_split_normal(sample_data):
    data, target, regressor = sample_data
    X_train, X_test, y_train, y_test = regressor.split(
        data=data, target=target, auto_cat_encode=True
    )
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1


def test_split_with_drop(sample_data):
    data, target, regressor = sample_data
    X_train, X_test, y_train, y_test = regressor.split(
        data, target, auto_cat_encode=True, drop=['feature1']
    )
    assert 'feature1' not in X_train.columns
    assert 'feature1' not in X_test.columns


def test_split_invalid_drop_type(sample_data):
    data, target, regressor = sample_data
    with pytest.raises(MultiTrainTypeError):
        regressor.split(data, target, drop='feature1')


def test_split_missing_target(sample_data):
    data, target, regressor = sample_data
    with pytest.raises(MultiTrainColumnMissingError):
        regressor.split(data.drop(columns=[target]), target)


def test_split_auto_cat_encode(sample_data):
    data, target, regressor = sample_data
    X_train, X_test, y_train, y_test = regressor.split(
        data, target, auto_cat_encode=True
    )
    assert X_train['feature2'].dtype in [int, float]


def test_fit_normal(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits)
    assert isinstance(results, pd.DataFrame)


def test_fit_invalid_datasplits(sample_data):
    data, target, regressor = sample_data
    with pytest.raises(MultiTrainSplitError):
        regressor.fit((data, data, data))


def test_fit_custom_metric(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits, custom_metric='mean_absolute_error')
    assert 'mean_absolute_error' in results.columns


def test_split_with_nan_handling(sample_data):
    data, target, regressor = sample_data
    data.loc[0, 'feature1'] = None  # Introduce NaN
    data_split = regressor.split(
        data, target, auto_cat_encode=True, fix_nan_custom={'feature1': 'ffill'}
    )
    assert not data_split[0].isnull().any().any()


def test_split_manual_encoding(sample_data):
    data, target, regressor = sample_data
    manual_encode = {'label': ['feature2']}
    data_split = regressor.split(data, target, manual_encode=manual_encode)
    assert data_split[0]['feature2'].dtype in [int, float]


def test_split_complex_dataset():
    data = pd.DataFrame({
        'feature1': range(1, 11),
        'feature2': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'feature3': range(10, 0, -1),
        'target': [x + 0.5 for x in range(10)]
    })
    target = 'target'
    regressor = MultiRegressor()
    split = regressor.split(data, target, auto_cat_encode=True)
    assert len(split[0]) == 8  # train set
    assert len(split[1]) == 2  # test set


def test_fit_with_custom_models(sample_data):
    data, target, regressor = sample_data
    regressor.custom_models = ['LinearRegression', 'Ridge']
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits)
    assert isinstance(results, pd.DataFrame)
    assert 'LinearRegression' in results.index
    assert 'Ridge' in results.index


def test_fit_with_pca(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits, pca='StandardScaler')
    assert isinstance(results, pd.DataFrame)


def test_fit_with_invalid_pca_scaler(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    with pytest.raises(MultiTrainPCAError):
        regressor.fit(datasplits, pca='InvalidScaler')


def test_invalid_test_size(sample_data):
    data, target, regressor = sample_data
    with pytest.raises(ValueError):
        regressor.split(data, target, test_size=1.5)


def test_show_train_score(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits, show_train_score=True)
    assert 'r2_score_train' in results.columns


def test_invalid_device_type():
    with pytest.raises(MultiTrainTypeError):
        MultiRegressor(device=123)


def test_gpu_regressor():
    regressor = MultiRegressor(use_gpu=True)
    assert regressor.use_gpu == True
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [2, 4, 6, 8, 10]
    })
    datasplits = regressor.split(data, 'target')
    assert isinstance(datasplits[0], np.ndarray)

def test_return_best_model(sample_data):
    data, target, regressor = sample_data
    datasplits = regressor.split(data, target, auto_cat_encode=True)
    results = regressor.fit(datasplits, return_best_model='r2_score')
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1


def test_gpu_regressor_with_device():
    regressor = MultiRegressor(use_gpu=True, device='0')
    assert regressor.use_gpu == True
    assert regressor.device == '0'
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [2, 4, 6, 8, 10]
    })
    datasplits = regressor.split(data, 'target')
    assert isinstance(datasplits[0], np.ndarray)


def test_nan_handling_interpolate(sample_data):
    data, target, regressor = sample_data
    data.loc[0:2, 'feature1'] = None  # Introduce multiple NaNs
    data_split = regressor.split(
        data, 
        target, 
        auto_cat_encode=True, 
        fix_nan_custom={'feature1': 'interpolate'}
    )
    assert not data_split[0].isnull().any().any()


def test_invalid_nan_handling(sample_data):
    data, target, regressor = sample_data
    data.loc[0, 'feature1'] = None
    with pytest.raises(MultiTrainNaNError):
        regressor.split(data, target, auto_cat_encode=True, fix_nan_custom={'feature1': 'invalid_strategy'})

