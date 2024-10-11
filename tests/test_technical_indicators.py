import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.technical_indicators import sma, ema, rsi, macd, bollinger_bands, atr, add_all_indicators

@pytest.fixture(scope="module")
def sample_price_data(sample_feature_data):
    return sample_feature_data

def test_sma(sample_price_data):
    result = sma(sample_price_data['close'], 20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_data)
    assert result.isna().sum() == 19  # Первые 19 значений должны быть NaN

def test_ema(sample_price_data):
    result = ema(sample_price_data['close'], 20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_data)
    assert result.isna().sum() == 0  # EMA не должен иметь NaN значений


def test_rsi(sample_price_data):
    result = rsi(sample_price_data['close'], 14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_data)
    assert result.isna().sum() == 13  # Первые 13 значений должны быть NaN

    # Игнорируем NaN значения при проверке диапазона
    valid_values = result.dropna()
    assert (valid_values >= 0).all() and (valid_values <= 100).all(), f"RSI вышел за пределы диапазона [0, 100]. Min: {valid_values.min()}, Max: {valid_values.max()}"

    # Дополнительная проверка для отладки
    if not ((valid_values >= 0).all() and (valid_values <= 100).all()):
        problem_values = valid_values[(valid_values < 0) | (valid_values > 100)]
        print("Проблемные значения RSI:")
        print(problem_values)
        print("Минимальное значение RSI:", valid_values.min())
        print("Максимальное значение RSI:", valid_values.max())

def test_macd(sample_price_data):
    result = macd(sample_price_data['close'])
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'macd', 'signal', 'histogram'}
    assert len(result) == len(sample_price_data)

def test_bollinger_bands(sample_price_data):
    result = bollinger_bands(sample_price_data['close'], 20)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'upper', 'middle', 'lower'}
    assert len(result) == len(sample_price_data)
    valid_data = result.dropna()
    assert (valid_data['upper'] >= valid_data['middle']).all() and (valid_data['middle'] >= valid_data['lower']).all()

def test_atr(sample_price_data):
    result = atr(sample_price_data['high'], sample_price_data['low'], sample_price_data['close'], 14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_price_data)
    assert result.isna().sum() == 13  # Первые 13 значений должны быть NaN

def test_add_all_indicators(sample_price_data):
    result = add_all_indicators(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    expected_columns = {'close', 'volume', 'high', 'low', 'SMA_20', 'EMA_20', 'RSI_14', 'macd', 'signal', 'histogram',
                        'upper', 'middle', 'lower', 'ATR_14'}
    assert set(result.columns) >= expected_columns
    assert len(result) == len(sample_price_data)