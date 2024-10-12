# tests/test_data_harmonization.py

import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_harmonization import (
    harmonize_price_data, harmonize_order_book, harmonize_historical_data,
    handle_missing_values, normalize_data,
    aggregate_data
)

@pytest.fixture
def sample_price_data():
    return {
        'binance': {'symbol': 'BTCUSDT', 'price': '50000', 'time': 1625097600000},
        'okx': {'instId': 'BTC-USDT', 'last': '50100', 'ts': '1625097600000'}
    }

@pytest.fixture
def sample_order_book_data():
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

@pytest.fixture
def sample_historical_data():
    return {
        'binance': [
            [1625097600000, '50000', '50100', '49900', '50050', '100'],
            [1625101200000, '50050', '50200', '50000', '50150', '120']
        ],
        'okx': [
            [1625097600000, '50010', '50110', '49910', '50060', '110'],
            [1625101200000, '50060', '50210', '50010', '50160', '130']
        ]
    }

@pytest.fixture
def sample_cmc_listings():
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

def test_harmonize_price_data(sample_price_data):
    result = harmonize_price_data(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'symbol', 'price', 'timestamp'}
    assert len(result) == 2
    assert 'binance' in result['source'].values
    assert 'okx' in result['source'].values

def test_harmonize_order_book(sample_order_book_data):
    result = harmonize_order_book(sample_order_book_data)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'type', 'price', 'quantity'}
    assert len(result) == 8  # 2 bids + 2 asks for each of 2 sources
    assert set(result['type'].unique()) == {'bid', 'ask'}

def test_harmonize_historical_data(sample_historical_data):
    result = harmonize_historical_data(sample_historical_data)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    assert len(result) == 4  # 2 candles for each of 2 sources


def test_handle_missing_values():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5]
    })
    result = handle_missing_values(df, method='ffill')
    assert result['A'].isna().sum() == 0
    assert result['B'].isna().sum() == 0

def test_normalize_data():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    result = normalize_data(df, method='minmax')
    assert result['A'].min() == 0
    assert result['A'].max() == 1
    assert result['B'].min() == 0
    assert result['B'].max() == 1


def test_aggregate_data():
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2021-01-01', periods=5, freq='H'),
        'value': [1, 2, 3, 4, 5]
    })
    df.set_index('timestamp', inplace=True)
    result = aggregate_data(df, 'timestamp', {'value': 'mean'})
    assert len(result) == 1
    assert result['value'].iloc[0] == 3

def test_harmonize_price_data_with_invalid_input():
    invalid_data = {'invalid_source': {'invalid_key': 'invalid_value'}}
    result = harmonize_price_data(invalid_data)
    assert len(result) == 0

def test_harmonize_historical_data_with_invalid_input():
    invalid_data = {'invalid_source': [[1, 'a', 'b', 'c', 'd', 'e']]}
    result = harmonize_historical_data(invalid_data)
    assert len(result) == 0

def test_normalize_data_with_constant_values():
    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1],
        'B': [2, 2, 2, 2, 2]
    })
    result = normalize_data(df, method='minmax')
    assert (result['A'] == 0).all()
    assert (result['B'] == 0).all()



