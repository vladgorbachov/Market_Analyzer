# tests/test_api_manager.py

import pytest
from unittest.mock import patch, MagicMock
from src.data_collection.api_manager import APIManager

@pytest.fixture
def api_manager():
    return APIManager()

@patch('src.data_collection.binance_api.get_symbol_price')
@patch('src.data_collection.okx_api.get_ticker')
def test_get_price_data(mock_okx_ticker, mock_binance_price, api_manager):
    mock_binance_price.return_value = {"symbol": "BTCUSDT", "price": "50000"}
    mock_okx_ticker.return_value = {"instId": "BTC-USDT", "last": "50100"}

    result = api_manager.get_price_data("BTCUSDT")

    assert "binance" in result
    assert "okx" in result
    assert result["binance"]["price"] == "50000"
    assert result["okx"]["last"] == "50100"

@patch('src.data_collection.coinmarketcap_api.get_latest_listings')
def test_get_market_overview(mock_get_listings, api_manager):
    mock_data = {"data": [{"id": 1, "name": "Bitcoin", "symbol": "BTC"}]}
    mock_get_listings.return_value = mock_data

    result = api_manager.get_market_overview(limit=1)

    assert result == mock_data
    mock_get_listings.assert_called_once_with(1)

@patch('src.data_collection.binance_api.get_order_book')
@patch('src.data_collection.okx_api.get_order_book')
def test_get_order_book(mock_okx_order_book, mock_binance_order_book, api_manager):
    mock_binance_order_book.return_value = {"bids": [["49900", "1.5"]], "asks": [["50100", "1.0"]]}
    mock_okx_order_book.return_value = {"bids": [["49950", "1.0"]], "asks": [["50050", "1.2"]]}

    result = api_manager.get_order_book("BTCUSDT", 1)

    assert "binance" in result
    assert "okx" in result
    assert len(result["binance"]["bids"]) == 1
    assert len(result["okx"]["asks"]) == 1

@patch('src.data_collection.binance_api.get_historical_klines')
@patch('src.data_collection.okx_api.get_candlesticks')
def test_get_historical_data(mock_okx_candlesticks, mock_binance_klines, api_manager):
    mock_binance_klines.return_value = [[1625097600000, "50000", "50100", "49900", "50050", "100"]]
    mock_okx_candlesticks.return_value = [[1625097600000, "50010", "50110", "49910", "50060", "110"]]

    result = api_manager.get_historical_data("BTCUSDT", "1h", 1625097600000, 1625101200000)

    assert "binance" in result
    assert "okx" in result
    assert len(result["binance"]) == 1
    assert len(result["okx"]) == 1

@patch('src.data_collection.coinmarketcap_api.get_metadata')
def test_get_crypto_metadata(mock_get_metadata, api_manager):
    mock_data = {"data": {"BTC": {"id": 1, "name": "Bitcoin", "symbol": "BTC"}}}
    mock_get_metadata.return_value = mock_data

    result = api_manager.get_crypto_metadata("BTC")

    assert result == mock_data
    mock_get_metadata.assert_called_once_with("BTC")

@patch('src.data_collection.coinmarketcap_api.get_global_metrics')
def test_get_global_market_metrics(mock_get_global_metrics, api_manager):
    mock_data = {"data": {"total_market_cap": {"USD": 2000000000000}}}
    mock_get_global_metrics.return_value = mock_data

    result = api_manager.get_global_market_metrics()

    assert result == mock_data
    mock_get_global_metrics.assert_called_once()

@patch('src.data_collection.binance_api.get_exchange_info')
@patch('src.data_collection.coinmarketcap_api.get_exchange_listings')
def test_get_exchange_info(mock_cmc_listings, mock_binance_info, api_manager):
    mock_binance_info.return_value = {"symbols": [{"symbol": "BTCUSDT"}]}
    mock_cmc_listings.return_value = {"data": [{"id": 1, "name": "Binance"}]}

    result = api_manager.get_exchange_info()

    assert "binance" in result
    assert "coinmarketcap" in result
    assert "symbols" in result["binance"]
    assert "data" in result["coinmarketcap"]

@patch('src.data_collection.okx_api.get_funding_rate')
def test_get_funding_rates(mock_get_funding_rate, api_manager):
    mock_data = {"fundingRate": "0.0001", "nextFundingRate": "0.0002"}
    mock_get_funding_rate.return_value = mock_data

    result = api_manager.get_funding_rates("BTC-USD-SWAP")

    assert result == mock_data
    mock_get_funding_rate.assert_called_once_with("BTC-USD-SWAP")

def test_api_manager_initialization(api_manager):
    assert hasattr(api_manager, 'binance')
    assert hasattr(api_manager, 'coinmarketcap')
    assert hasattr(api_manager, 'okx')

