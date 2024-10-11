import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_selector import ModelSelector
from src.trading.signal_generator import SignalGenerator

@pytest.fixture(scope="module")
def sample_price_data():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='h')
    prices = np.random.randn(1000).cumsum() + 100
    volumes = np.random.randint(1000, 10000, 1000)
    return pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volumes,
        'high': prices + np.random.rand(1000),
        'low': prices - np.random.rand(1000)
    }).set_index('timestamp')

@pytest.fixture(scope="module")
def sample_feature_data(sample_price_data):
    data = sample_price_data.copy()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI_14'] = 50 + np.random.randn(1000) * 10  # Упрощенный RSI
    return data.dropna()

@pytest.fixture(scope="module")
def model_selector():
    return ModelSelector()

@pytest.fixture(scope="module")
def signal_generator():
    return SignalGenerator()

@pytest.fixture(scope="module")
def trained_model_selector(model_selector, sample_feature_data):
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_feature_data, 'close', feature_columns)
    return model_selector