import pytest
import pandas as pd
from src.data_processing.data_harmonization import harmonize_price_data, harmonize_order_book, harmonize_historical_data


def test_harmonize_price_data():
    test_data = {
        'binance': {'symbol': 'BTCUSDT', 'price': '50000', 'time': 1625097600000},
        'okx': {'instId': 'BTC-USDT', 'last': '50100', 'ts': '1625097600000'}
    }
    result = harmonize_price_data(test_data)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'symbol', 'price', 'timestamp'}
    assert len(result) == 2
    assert 'binance' in result['source'].values
    assert 'okx' in result['source'].values


def test_harmonize_order_book():
    test_data = {
        'binance': {
            'bids': [['49900', '1.5'], ['49800', '2.0']],
            'asks': [['50100', '1.0'], ['50200', '2.5']]
        },
        'okx': {
            'bids': [['49950', '1.0'], ['49850', '1.5']],
            'asks': [['50050', '1.2'], ['50150', '2.0']]
        }
    }
    result = harmonize_order_book(test_data)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'type', 'price', 'quantity'}
    assert len(result) == 8  # 2 bids + 2 asks for each of 2 sources
    assert set(result['type'].unique()) == {'bid', 'ask'}


def test_harmonize_historical_data():
    test_data = {
        'binance': [
            [1625097600000, '50000', '50100', '49900', '50050', '100'],
            [1625101200000, '50050', '50200', '50000', '50150', '120']
        ],
        'okx': [
            [1625097600000, '50010', '50110', '49910', '50060', '110'],
            [1625101200000, '50060', '50210', '50010', '50160', '130']
        ]
    }
    result = harmonize_historical_data(test_data)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'source', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    assert len(result) == 4  # 2 candles for each of 2 sources