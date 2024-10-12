# tests/test_technical_indicators.py

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.technical_indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, add_all_indicators,
    momentum, stochastic_oscillator
)


@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data


def test_sma(sample_data):
    result = sma(sample_data['close'], window=20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert pd.isna(result.iloc[:19]).all()
    assert not pd.isna(result.iloc[19:]).any()


def test_ema(sample_data):
    result = ema(sample_data['close'], span=20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert not pd.isna(result).any()


def test_rsi(sample_data):
    result = rsi(sample_data['close'], window=14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert pd.isna(result.iloc[:14]).all()
    assert not pd.isna(result.iloc[14:]).any()
    assert (result >= 0).all() and (result <= 100).all()


def test_macd(sample_data):
    result = macd(sample_data['close'])
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'macd', 'signal', 'histogram'}
    assert len(result) == len(sample_data)


def test_bollinger_bands(sample_data):
    result = bollinger_bands(sample_data['close'], window=20)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'upper', 'middle', 'lower'}
    assert len(result) == len(sample_data)
    assert (result['upper'] >= result['middle']).all()
    assert (result['middle'] >= result['lower']).all()


def test_atr(sample_data):
    result = atr(sample_data['high'], sample_data['low'], sample_data['close'], window=14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert pd.isna(result.iloc[:14]).all()
    assert not pd.isna(result.iloc[14:]).any()
    assert (result >= 0).all()


def test_add_all_indicators(sample_data):
    result = add_all_indicators(sample_data)
    assert isinstance(result, pd.DataFrame)
    expected_columns = {'close', 'high', 'low', 'volume', 'SMA_20', 'EMA_20', 'RSI_14',
                        'macd', 'signal', 'histogram', 'upper', 'middle', 'lower', 'ATR_14'}
    assert set(result.columns) >= expected_columns
    assert len(result) == len(sample_data)


def test_momentum(sample_data):
    result = momentum(sample_data['close'], period=14)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert pd.isna(result.iloc[:14]).all()
    assert not pd.isna(result.iloc[14:]).any()


def test_stochastic_oscillator(sample_data):
    result = stochastic_oscillator(sample_data['high'], sample_data['low'], sample_data['close'])
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'%K', '%D'}
    assert len(result) == len(sample_data)
    assert ((result >= 0) & (result <= 100)).all().all()


def test_sma_different_windows(sample_data):
    for window in [5, 10, 20, 50]:
        result = sma(sample_data['close'], window=window)
        assert pd.isna(result.iloc[:window - 1]).all()
        assert not pd.isna(result.iloc[window - 1:]).any()


def test_ema_different_spans(sample_data):
    for span in [5, 10, 20, 50]:
        result = ema(sample_data['close'], span=span)
        assert not pd.isna(result).any()


def test_rsi_different_windows(sample_data):
    for window in [7, 14, 21]:
        result = rsi(sample_data['close'], window=window)
        assert pd.isna(result.iloc[:window]).all()
        assert not pd.isna(result.iloc[window:]).any()
        assert (result >= 0).all() and (result <= 100).all()


def test_bollinger_bands_different_std(sample_data):
    for num_std in [1, 2, 3]:
        result = bollinger_bands(sample_data['close'], window=20, num_std=num_std)
        assert (result['upper'] >= result['middle']).all()
        assert (result['middle'] >= result['lower']).all()


def test_indicators_with_constant_price(sample_data):
    constant_price = pd.Series([100] * len(sample_data), index=sample_data.index)

    assert (sma(constant_price, 20) == 100).all()
    assert (ema(constant_price, 20) == 100).all()
    assert pd.isna(rsi(constant_price, 14)).all()

    macd_result = macd(constant_price)
    assert (macd_result['macd'] == 0).all()
    assert (macd_result['signal'] == 0).all()
    assert (macd_result['histogram'] == 0).all()

    bb_result = bollinger_bands(constant_price, 20)
    assert (bb_result['upper'] == 100).all()
    assert (bb_result['middle'] == 100).all()
    assert (bb_result['lower'] == 100).all()

