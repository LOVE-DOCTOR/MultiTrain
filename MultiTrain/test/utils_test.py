import pytest
import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from MultiTrain.utils.utils import (
    _models_classifier,
    _models_regressor,
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
    _fit_pred_text,
    _format_time,
    _prep_model_names_list,
    _sub_fit,
)
from MultiTrain.errors.errors import (
    MultiTrainMetricError,
    MultiTrainEncodingError,
    MultiTrainNaNError,
    MultiTrainTypeError,
    MultiTrainColumnMissingError,
    MultiTrainPCAError,
)
from sklearn.metrics import accuracy_score, mean_squared_error


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "cat_col": ["a", "b", "a"], 
        "num_col": [1, None, 3],
        "text_col": ["sample text", "another text", "more text"]
    })


@pytest.fixture
def models():
    return _models_classifier(random_state=42, n_jobs=1, max_iter=500)


def test_models_classifier(models):
    assert "LogisticRegression" in models
    assert isinstance(models["LogisticRegression"], sklearn.linear_model.LogisticRegression)


def test_models_regressor():
    models = _models_regressor(random_state=42, n_jobs=1, max_iter=500)
    assert "LinearRegression" in models
    assert isinstance(models["LinearRegression"], sklearn.linear_model.LinearRegression)


def test_init_metrics():
    metrics = _init_metrics()
    assert "accuracy_score" in metrics
    assert "mean_squared_error" in metrics


@pytest.mark.parametrize(
    "custom_metric, metric_type, expected_exception",
    [
        (None, "classification", None),
        ("invalid_metric", "classification", MultiTrainMetricError),
        (None, "regression", None),
        ("mean_squared_error", "regression", None),
    ],
)
def test_metrics(custom_metric, metric_type, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            _metrics(custom_metric=custom_metric, metric_type=metric_type)
    else:
        metrics = _metrics(custom_metric=custom_metric, metric_type=metric_type)
        assert len(metrics) > 0


def test_cat_encoder(sample_dataframe):
    encoded_df = _cat_encoder(sample_dataframe, auto_cat_encode=True)
    assert "cat_col" in encoded_df.columns
    assert encoded_df["cat_col"].dtype in ["int64", "int32"]


def test_manual_encoder(sample_dataframe):
    manual_encode = {"label": ["cat_col"]}
    encoded_df = _manual_encoder(manual_encode, sample_dataframe)
    assert "cat_col" in encoded_df.columns
    assert encoded_df["cat_col"].dtype in ["int64", "int32"]


def test_manual_encoder_onehot(sample_dataframe):
    manual_encode = {"onehot": ["cat_col"]}
    encoded_df = _manual_encoder(manual_encode, sample_dataframe)
    assert "cat_col_a" in encoded_df.columns
    assert "cat_col_b" in encoded_df.columns


def test_non_auto_cat_encode_error(sample_dataframe):
    with pytest.raises(MultiTrainEncodingError):
        _non_auto_cat_encode_error(
            sample_dataframe, auto_cat_encode=False, manual_encode=None
        )


def test_fill_missing_values(sample_dataframe):
    filled_col = _fill_missing_values(sample_dataframe, "num_col")
    assert not filled_col.isnull().any()


def test_handle_missing_values_strategies(sample_dataframe):
    # Test different strategies
    strategies = {
        "ffill": {"num_col": "ffill"},
        "bfill": {"num_col": "bfill"},
        "interpolate": {"num_col": "interpolate"}
    }
    
    for strategy_name, strategy in strategies.items():
        fixed_df = _handle_missing_values(sample_dataframe, fix_nan_custom=strategy)
        assert not fixed_df["num_col"].isnull().any()


def test_handle_missing_values_invalid_column():
    df = pd.DataFrame({"col": [1, None, 3]})
    with pytest.raises(MultiTrainColumnMissingError):
        _handle_missing_values(df, fix_nan_custom={"invalid_col": "ffill"})


def test_handle_missing_values_invalid_strategy():
    df = pd.DataFrame({"col": [1, None, 3]})
    with pytest.raises(MultiTrainNaNError):
        _handle_missing_values(df, fix_nan_custom={"col": "invalid_strategy"})


def test_display_table():
    # Test classification results
    class_results = {
        "model1": {"accuracy": 0.9, "precision": 0.8},
        "model2": {"accuracy": 0.8, "precision": 0.9}
    }
    sorted_df = _display_table(class_results, sort="accuracy", task="classification")
    assert sorted_df.index[0] == "model1"

    # Test regression results
    reg_results = {
        "model1": {"mean squared error": 0.1},
        "model2": {"mean squared error": 0.2}
    }
    sorted_df = _display_table(reg_results, sort="mean squared error", task="regression")
    assert sorted_df.index[0] == "model1"


def test_check_custom_models(models):
    model_names, model_list = _check_custom_models(["LogisticRegression"], models)
    assert "LogisticRegression" in model_names
    assert len(model_list) == 1


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
        False
    )
    assert len(prediction) == len(X_test)
    assert isinstance(time_taken, str)


def test_calculate_metric():
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    
    # Test classification metric
    accuracy = _calculate_metric(accuracy_score, y_true, y_pred)
    assert accuracy == pytest.approx(2/3)
    
    # Test regression metric
    mse = _calculate_metric(mean_squared_error, y_true, y_pred)
    assert mse == pytest.approx(1/3)


def test_fit_pred_text(sample_dataframe):
    model = _models_classifier(random_state=42, n_jobs=1, max_iter=500)["LogisticRegression"]
    pipeline_dict = {
        "ngram_range": (1, 2),
        "encoding": "utf-8",
        "max_features": 1000,
        "analyzer": "word"
    }
    X_train = sample_dataframe["text_col"][:2]
    y_train = pd.Series([0, 1])
    X_test = sample_dataframe["text_col"][2:]
    
    pipeline, predictions, time_taken = _fit_pred_text(
        "tfidf",
        pipeline_dict,
        model,
        X_train,
        y_train,
        X_test,
        pca=False
    )
    
    assert isinstance(pipeline, Pipeline)
    assert len(predictions) == 1
    assert isinstance(time_taken, str)


def test_format_time():
    assert _format_time(61) == "1m 1.00s"
    assert _format_time(3661) == "1h 1m 1.00s"
    assert _format_time(0.5) == "0.5s"


def test_prep_model_names_list():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    datasplits = (X[:3], X[3:], y[:3], y[3:])
    
    model_names, model_list, X_train, X_test, y_train, y_test = _prep_model_names_list(
        datasplits=datasplits,
        custom_metric=None,
        random_state=42,
        n_jobs=1,
        custom_models=["LogisticRegression"],
        class_type="classification",
        max_iter=500
    )
    
    assert "LogisticRegression" in model_names
    assert len(model_list) == 1
    assert len(X_train) == 3
    assert len(X_test) == 1


def test_sub_fit():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"feature": [4]})
    model = _models_classifier(random_state=42, n_jobs=1, max_iter=500)["LogisticRegression"]
    
    fitted_model, predictions = _sub_fit(model, X_train, y_train, X_test, pca_scaler=False)
    
    assert hasattr(fitted_model, "predict")
    assert len(predictions) == 1
