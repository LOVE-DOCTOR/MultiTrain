import pytest
import pandas as pd
import sklearn

from MultiTrain.utils.utils import (
    _models_classifier,
    _init_metrics,
    _metrics,
    _cat_encoder,
    _manual_encoder,
    _non_auto_cat_encode_error,
    _fill_missing_values,
    _handle_missing_values,
    _display_table,
    _check_custom_models,
    _fit_pred,
    _calculate_metric,
)
from MultiTrain.errors.errors import (
    MultiTrainMetricError,
    MultiTrainEncodingError,
    MultiTrainNaNError,
    MultiTrainTypeError,
    MultiTrainColumnMissingError,
)
from sklearn.metrics import accuracy_score


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"cat_col": ["a", "b", "a"], "num_col": [1, None, 3]})


@pytest.fixture
def models():
    return _models_classifier(random_state=42, n_jobs=1, max_iter=500)


def test_models(models):
    assert "LogisticRegression" in models
    assert isinstance(
        models["LogisticRegression"], sklearn.linear_model.LogisticRegression
    )


def test_init_metrics():
    metrics = _init_metrics()
    assert "accuracy_score" in metrics


@pytest.mark.parametrize(
    "custom_metric, expected_exception",
    [(None, None), ("invalid_metric", MultiTrainMetricError)],
)
def test_metrics(custom_metric, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            _metrics(custom_metric=custom_metric, metric_type='classification') 
    else:
        metrics = _metrics(custom_metric=custom_metric, metric_type='classification') 
        assert "accuracy" in metrics


def test_cat_encoder(sample_dataframe):
    encoded_df = _cat_encoder(sample_dataframe, auto_cat_encode=True)
    assert "cat_col" in encoded_df.columns
    assert encoded_df["cat_col"].dtype in ["int64", "int32"]


def test_manual_encoder(sample_dataframe):
    manual_encode = {"label": ["cat_col"], "onehot": []}
    encoded_df = _manual_encoder(manual_encode, sample_dataframe)
    assert "cat_col" in encoded_df.columns
    assert encoded_df["cat_col"].dtype in ["int64", "int32"]


def test_non_auto_cat_encode_error(sample_dataframe):
    with pytest.raises(MultiTrainEncodingError):
        _non_auto_cat_encode_error(
            sample_dataframe, auto_cat_encode=False, manual_encode=None
        )


def test_fill_missing_values(sample_dataframe):
    filled_col = _fill_missing_values(sample_dataframe, "num_col")
    assert not filled_col.isnull().any()


def test_handle_missing_values(sample_dataframe):
    with pytest.raises(MultiTrainNaNError):
        _handle_missing_values(sample_dataframe, fix_nan_custom=False)

    fixed_df = _handle_missing_values(
        sample_dataframe, fix_nan_custom={"num_col": "ffill"}
    )
    assert not fixed_df["num_col"].isnull().any()


def test_display_table():
    results = {"model1": {"accuracy": 0.9}, "model2": {"accuracy": 0.8}}
    sorted_df = _display_table(results, sort="accuracy")
    assert sorted_df.index[0] == "model1"


def test_check_custom_models(models):
    model_names, model_list = _check_custom_models(["LogisticRegression"], models)
    assert "LogisticRegression" in model_names


def test_fit_pred(models):
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"feature": [1, 2]})
    model, prediction, time_taken = _fit_pred(
        models["LogisticRegression"],
        ["LogisticRegression"],
        0,
        X_train,
        y_train,
        X_test,
    )
    assert len(prediction) == len(X_test)


def test_calculate_metric():
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    accuracy = _calculate_metric(accuracy_score, y_true, y_pred)
    assert accuracy == pytest.approx(2 / 3)
