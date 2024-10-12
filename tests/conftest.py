# tests/conftest.py

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_selector import ModelSelector
from src.trading.signal_generator import SignalGenerator

@pytest.fixture(scope="session")
def sample_price_data():
    """Создает образец ценовых данных."""
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    np.random.seed(42)
    prices = np.random.randn(1000).cumsum() + 100
    volumes = np.random.randint(1000, 10000, 1000)
    return pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volumes,
        'high': prices + np.random.rand(1000),
        'low': prices - np.random.rand(1000)
    }).set_index('timestamp')

@pytest.fixture(scope="session")
def sample_feature_data(sample_price_data):
    """Создает образец данных с техническими индикаторами."""
    data = sample_price_data.copy()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI_14'] = 50 + np.random.randn(1000) * 10  # Упрощенный RSI
    return data.dropna()

@pytest.fixture(scope="session")
def model_selector():
    """Создает экземпляр ModelSelector."""
    return ModelSelector()

@pytest.fixture(scope="session")
def signal_generator():
    """Создает экземпляр SignalGenerator."""
    return SignalGenerator()

@pytest.fixture(scope="session")
def trained_model_selector(model_selector, sample_feature_data):
    """Создает обученный экземпляр ModelSelector."""
    feature_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'volume']
    model_selector.fit(sample_feature_data, 'close', feature_columns)
    return model_selector

@pytest.fixture(scope="session")
def sample_order_book_data():
    """Создает образец данных книги ордеров."""
    return {
        'binance': {
            'bids': [['49900', '1.5'], ['49800', '2.0']],
            'asks': [['50100', '1.0'], ['50200', '2.5']]
        },
        'okx': {
            'bids': [['49950', '1.0'], ['49850', '1.5']],
            'asks': [['50050', '1.2'], ['50150', '2.0']]
        }
    }

@pytest.fixture(scope="session")
def sample_historical_data():
    """Создает образец исторических данных."""
    start_time = int(datetime(2020, 1, 1).timestamp() * 1000)
    return {
        'binance': [
            [start_time, '50000', '50100', '49900', '50050', '100'],
            [start_time + 3600000, '50050', '50200', '50000', '50150', '120']
        ],
        'okx': [
            [start_time, '50010', '50110', '49910', '50060', '110'],
            [start_time + 3600000, '50060', '50210', '50010', '50160', '130']
        ]
    }

@pytest.fixture(scope="session")
def sample_cmc_listings():
    """Создает образец данных листинга с CoinMarketCap."""
    return {
        'data': [
            {
                'id': 1,
                'name': 'Bitcoin',
                'symbol': 'BTC',
                'slug': 'bitcoin',
                'cmc_rank': 1,
                'quote': {
                    'USD': {
                        'price': 50000,
                        'volume_24h': 30000000000,
                        'percent_change_1h': 0.1,
                        'percent_change_24h': 1.5,
                        'percent_change_7d': -2.0,
                        'market_cap': 1000000000000
                    }
                },
                'last_updated': '2021-07-01T00:00:00.000Z'
            }
        ]
    }

@pytest.fixture(scope="session")
def sample_text_data():
    """Создает образец текстовых данных для анализа настроений."""
    return pd.DataFrame({
        'text': [
            "Bitcoin is showing strong bullish trends.",
            "The crypto market is facing regulatory challenges.",
            "Ethereum upgrade successfully implemented.",
            "Bearish sentiment prevails in altcoin markets."
        ]
    })

