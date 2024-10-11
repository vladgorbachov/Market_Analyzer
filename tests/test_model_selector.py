import pytest
import pandas as pd
import numpy as np

from src.models.model_selector import ModelSelector


def test_model_selector_initialization(model_selector):
    assert isinstance(model_selector, ModelSelector)
    assert set(model_selector.models.keys()) >= {'ARIMA', 'Prophet', 'XGBoost', 'RandomForest'}


def test_model_selector_prepare_data(model_selector, sample_feature_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X_train, X_test, y_train, y_test = model_selector.prepare_data(sample_feature_data, 'close', feature_columns)

    assert len(X_train) + len(X_test) == len(sample_feature_data)
    assert set(X_train.columns) == set(feature_columns)
    assert isinstance(y_train, pd.Series)
    assert y_train.name == 'close'


def test_model_selector_train_and_evaluate(model_selector, sample_feature_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X_train, X_test, y_train, y_test = model_selector.prepare_data(sample_feature_data, 'close', feature_columns)
    results = model_selector.train_and_evaluate(X_train, X_test, y_train, y_test)

    assert isinstance(results, dict)
    assert set(results.keys()) >= {'ARIMA', 'Prophet', 'XGBoost', 'RandomForest'}
    for model_results in results.values():
        assert set(model_results.keys()) == {'MSE', 'MAE', 'R2'}
        assert all(isinstance(value, float) for value in model_results.values())


def test_model_selector_select_best_model(model_selector, sample_feature_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_feature_data, 'close', feature_columns)

    assert hasattr(model_selector, 'best_model_name')
    assert hasattr(model_selector, 'best_model')
    assert model_selector.best_model_name in model_selector.models.keys()


def test_model_selector_predict(trained_model_selector, sample_feature_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X = sample_feature_data[feature_columns]
    predictions = trained_model_selector.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()


def test_model_selector_tune_best_model(trained_model_selector, sample_feature_data):
    if trained_model_selector.best_model_name in ['XGBoost', 'RandomForest']:
        param_grid = {'n_estimators': [50, 100, 200]}
        feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
        X = sample_feature_data[feature_columns]
        y = sample_feature_data['close']

        trained_model_selector.tune_best_model(X, y, param_grid)

        assert 'n_estimators' in trained_model_selector.best_model.get_params()
    else:
        pytest.skip("Automatic tuning is not implemented for the selected model")