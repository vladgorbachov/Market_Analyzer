# src/data_collection/binance_api.py

import requests
from typing import Dict, Any, List

BASE_URL = "https://api.binance.com/api/v3"


def get_symbol_price(symbol: str) -> Dict[str, Any]:
    """Get the current price for a symbol."""
    endpoint = f"{BASE_URL}/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


def get_historical_klines(symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
    """Get historical klines/candlestick data."""
    endpoint = f"{BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()