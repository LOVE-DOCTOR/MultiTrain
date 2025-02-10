import pytest
import pandas as pd
import numpy as np
from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import (
    MultiTrainDatasetTypeError,
    MultiTrainColumnMissingError,
    MultiTrainEncodingError,
    MultiTrainError,
    MultiTrainTypeError,
    MultiTrainNaNError,
    MultiTrainMetricError,
    MultiTrainSplitError,
    MultiTrainPCAError,
    MultiTrainTextError,
)
import logging


@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["A", "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    target = "target"
    classifier = MultiClassifier(custom_models=["LogisticRegression"])
    return data, target, classifier


def test_split_normal(sample_data):
    data, target, classifier = sample_data
    X_train, X_test, y_train, y_test = classifier.split(
        data=data, target=target, auto_cat_encode=True
    )
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1


def test_split_with_drop(sample_data):
    data, target, classifier = sample_data
    X_train, X_test, y_train, y_test = classifier.split(
        data, target, auto_cat_encode=True, drop=["feature1"]
    )
    assert "feature1" not in X_train.columns
    assert "feature1" not in X_test.columns


def test_split_invalid_drop_type(sample_data):
    data, target, classifier = sample_data
    with pytest.raises(MultiTrainTypeError):
        classifier.split(data, target, drop="feature1")


def test_split_missing_target(sample_data):
    data, target, classifier = sample_data
    with pytest.raises(MultiTrainColumnMissingError):
        classifier.split(data.drop(columns=[target]), target)


def test_split_auto_cat_encode(sample_data):
    data, target, classifier = sample_data
    X_train, X_test, y_train, y_test = classifier.split(
        data, target, auto_cat_encode=True
    )
    assert X_train["feature2"].dtype in [int, float]


def test_fit_normal(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits)
    assert isinstance(results, pd.DataFrame)


def test_fit_invalid_datasplits(sample_data):
    data, target, classifier = sample_data
    with pytest.raises(MultiTrainSplitError):
        classifier.fit((data, data, data))


def test_fit_custom_metric(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    with pytest.raises(MultiTrainMetricError):
        classifier.fit(datasplits, custom_metric="invalid_metric")


def test_fit_imbalanced(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits, imbalanced=True)
    assert isinstance(results, pd.DataFrame)


def test_split_with_nan_handling(sample_data):
    data, target, classifier = sample_data
    data.loc[0, "feature1"] = None  # Introduce NaN
    data_split = classifier.split(
        data, target, auto_cat_encode=True, fix_nan_custom={"feature1": "ffill"}
    )
    assert not data_split[0].isnull().any().any()  # Ensure no NaNs in training data


def test_split_manual_encoding(sample_data):
    data, target, classifier = sample_data
    manual_encode = {"label": ["feature2"]}
    data_split = classifier.split(data, target, manual_encode=manual_encode)
    assert data_split[0]["feature2"].dtype in [
        int,
        float,
    ]  # Check if manual encoding worked


def test_split_complex_dataset():
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "feature3": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    target = "target"
    classifier = MultiClassifier()
    split = classifier.split(data, target, auto_cat_encode=True)
    assert len(split[0]) == 8
    assert len(split[1]) == 2


def test_fit_with_custom_models(sample_data):
    data, target, classifier = sample_data
    classifier.custom_models = ["RandomForestClassifier", "LogisticRegression"]
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits)
    assert isinstance(results, pd.DataFrame)
    assert "RandomForestClassifier" in results.index
    assert "LogisticRegression" in results.index


def test_fit_with_pca(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits, pca='StandardScaler')
    assert isinstance(results, pd.DataFrame)


def test_fit_with_invalid_pca_scaler(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    with pytest.raises(MultiTrainPCAError):
        classifier.fit(datasplits, pca='InvalidScaler')


def test_text_processing():
    data = pd.DataFrame({
        'text': ['This is sample text', 'Another example', 'Third sample'],
        'target': [0, 1, 0]
    })
    classifier = MultiClassifier()
    datasplits = classifier.split(data, 'target')
    pipeline_dict = {
        'ngram_range': (1, 2),
        'encoding': 'utf-8',
        'max_features': 1000,
        'analyzer': 'word'
    }
    results = classifier.fit(
        datasplits,
        text=True,
        vectorizer='tfidf',
        pipeline_dict=pipeline_dict
    )
    assert isinstance(results, pd.DataFrame)


def test_text_processing_missing_params(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target)
    with pytest.raises(MultiTrainTextError):
        classifier.fit(datasplits, text=True)


def test_return_best_model(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits, return_best_model='accuracy')
    assert isinstance(results, pd.DataFrame)
    assert 'Best Model' in results.index


def test_sort_results(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits, sort='accuracy')
    # Check if results are sorted by accuracy (descending)
    assert results['accuracy'].is_monotonic_decreasing


def test_gpu_classifier():
    classifier = MultiClassifier(use_gpu=True)
    assert classifier.use_gpu == True
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    datasplits = classifier.split(data, 'target')
    # GPU splits should return numpy arrays
    assert isinstance(datasplits[0], np.ndarray)


def test_invalid_test_size(sample_data):
    data, target, classifier = sample_data
    with pytest.raises(ValueError):
        classifier.split(data, target, test_size=1.5)


def test_show_train_score(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(datasplits, show_train_score=True)
    assert 'accuracy_train' in results.columns


def test_invalid_manual_encode(sample_data):
    data, target, classifier = sample_data
    with pytest.raises(MultiTrainError):
        # Test duplicate encoding types
        manual_encode = {'label': ['feature2'], 'label': ['feature1']}
        classifier.split(data, target, manual_encode=manual_encode)


def test_invalid_device_type():
    with pytest.raises(MultiTrainTypeError):
        MultiClassifier(device=123)  # device should be string


def test_pipeline_dict_without_text(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target)
    pipeline_dict = {
        'ngram_range': (1, 2),
        'encoding': 'utf-8'
    }
    with pytest.raises(MultiTrainTextError):
        classifier.fit(datasplits, pipeline_dict=pipeline_dict)


def test_multiple_custom_metrics(sample_data):
    data, target, classifier = sample_data
    datasplits = classifier.split(data, target, auto_cat_encode=True)
    results = classifier.fit(
        datasplits,
        custom_metric='f1_score',
        show_train_score=True,
        sort='accuracy'
    )
    assert isinstance(results, pd.DataFrame)
    assert 'f1_score' in results.columns
    assert results['accuracy'].is_monotonic_decreasing