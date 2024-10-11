import requests
from typing import Dict, Any, List
import time

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

def get_order_book(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """Get the current order book for a symbol."""
    endpoint = f"{BASE_URL}/depth"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()

def get_24h_ticker(symbol: str) -> Dict[str, Any]:
    """Get 24-hour price change statistics for a symbol."""
    endpoint = f"{BASE_URL}/ticker/24hr"
    params = {"symbol": symbol}
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()

def get_exchange_info() -> Dict[str, Any]:
    """Get exchange trading rules and symbol information."""
    endpoint = f"{BASE_URL}/exchangeInfo"
    response = requests.get(endpoint)
    response.raise_for_status()
    return response.json()

def get_recent_trades(symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Get recent trades for a symbol."""
    endpoint = f"{BASE_URL}/trades"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()