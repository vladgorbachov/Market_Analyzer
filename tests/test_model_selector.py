# tests/test_model_selector.py

import pytest
import pandas as pd
import numpy as np
from src.models.model_selector import ModelSelector


@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'SMA_20': np.random.randn(1000),
        'EMA_20': np.random.randn(1000),
        'RSI_14': np.random.randint(0, 100, 1000),
    })
    data.set_index('timestamp', inplace=True)
    return data


@pytest.fixture
def model_selector():
    return ModelSelector()


def test_model_selector_initialization(model_selector):
    assert isinstance(model_selector, ModelSelector)
    assert set(model_selector.models.keys()) >= {'ARIMA', 'Prophet', 'XGBoost', 'RandomForest'}
    assert model_selector.best_model is None
    assert model_selector.best_model_name is None


def test_prepare_data(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X_train, X_test, y_train, y_test = model_selector.prepare_data(sample_data, 'close', feature_columns)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) + len(X_test) == len(sample_data)
    assert set(X_train.columns) == set(feature_columns)
    assert y_train.name == 'close'


def test_train_and_evaluate(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X_train, X_test, y_train, y_test = model_selector.prepare_data(sample_data, 'close', feature_columns)
    results = model_selector.train_and_evaluate(X_train, X_test, y_train, y_test)

    assert isinstance(results, dict)
    assert set(results.keys()) >= {'ARIMA', 'Prophet', 'XGBoost', 'RandomForest'}
    for model_results in results.values():
        assert set(model_results.keys()) == {'MSE', 'MAE', 'R2'}
        assert all(isinstance(value, float) for value in model_results.values())


def test_select_best_model(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_data, 'close', feature_columns)

    assert model_selector.best_model_name is not None
    assert model_selector.best_model is not None
    assert model_selector.best_model_name in model_selector.models.keys()


def test_predict(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_data, 'close', feature_columns)

    X = sample_data[feature_columns]
    predictions = model_selector.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()


@pytest.mark.parametrize("model_name", ['XGBoost', 'RandomForest'])
def test_tune_best_model(model_selector, sample_data, model_name):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_data, 'close', feature_columns)

    # Принудительно устанавливаем лучшую модель
    model_selector.best_model_name = model_name
    model_selector.best_model = model_selector.models[model_name]

    param_grid = {'n_estimators': [50, 100]}
    X = sample_data[feature_columns]
    y = sample_data['close']

    model_selector.tune_best_model(X, y, param_grid)

    assert 'n_estimators' in model_selector.best_model.get_params()


def test_tune_best_model_unsupported(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_data, 'close', feature_columns)

    # Принудительно устанавливаем ARIMA как лучшую модель
    model_selector.best_model_name = 'ARIMA'
    model_selector.best_model = model_selector.models['ARIMA']

    param_grid = {'order': [(1, 1, 1), (2, 1, 2)]}
    X = sample_data[feature_columns]
    y = sample_data['close']

    with pytest.warns(UserWarning):
        model_selector.tune_best_model(X, y, param_grid)


def test_fit_with_invalid_data(model_selector):
    invalid_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    with pytest.raises(KeyError):
        model_selector.fit(invalid_data, 'close', ['A', 'B'])


def test_predict_without_fit(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X = sample_data[feature_columns]
    with pytest.raises(ValueError):
        model_selector.predict(X)


def test_prepare_data_with_missing_features(model_selector, sample_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume', 'non_existent_feature']
    with pytest.raises(KeyError):
        model_selector.prepare_data(sample_data, 'close', feature_columns)


def test_train_and_evaluate_with_constant_target(model_selector, sample_data):
    sample_data['constant'] = 1
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    X_train, X_test, y_train, y_test = model_selector.prepare_data(sample_data, 'constant', feature_columns)
    results = model_selector.train_and_evaluate(X_train, X_test, y_train, y_test)

    for model_results in results.values():
        assert model_results['R2'] == 0  # R2 должен быть 0 для константной целевой переменной

