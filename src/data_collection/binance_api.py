# src/data_collection/binance_api.py

import requests
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3"
RATE_LIMIT_STATUS_CODE = 429
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

class BinanceAPIException(Exception):
    pass

def handle_response(response: requests.Response) -> Dict[str, Any]:
    if response.status_code == 200:
        return response.json()
    elif response.status_code == RATE_LIMIT_STATUS_CODE:
        raise BinanceAPIException("Rate limit exceeded")
    else:
        raise BinanceAPIException(f"API request failed with status code {response.status_code}: {response.text}")

def retry_request(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                if "Rate limit exceeded" in str(e) and attempt < MAX_RETRIES - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    return wrapper

@retry_request
def get_symbol_price(symbol: str) -> Dict[str, Any]:
    """Get the current price for a symbol."""
    logger.info(f"Fetching price for symbol: {symbol}")
    endpoint = f"{BASE_URL}/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(endpoint, params=params)
    return handle_response(response)

@retry_request
def get_historical_klines(symbol: str, interval: str, start_time: int, end_time: int) -> List[List[Any]]:
    """Get historical klines/candlestick data."""
    logger.info(f"Fetching historical klines for symbol: {symbol}, interval: {interval}")
    endpoint = f"{BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(endpoint, params=params)
    return handle_response(response)

@retry_request
def get_order_book(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """Get the current order book for a symbol."""
    logger.info(f"Fetching order book for symbol: {symbol}, limit: {limit}")
    endpoint = f"{BASE_URL}/depth"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(endpoint, params=params)
    return handle_response(response)

@retry_request
def get_24h_ticker(symbol: str) -> Dict[str, Any]:
    """Get 24-hour price change statistics for a symbol."""
    logger.info(f"Fetching 24h ticker for symbol: {symbol}")
    endpoint = f"{BASE_URL}/ticker/24hr"
    params = {"symbol": symbol}
    response = requests.get(endpoint, params=params)
    return handle_response(response)

@retry_request
def get_exchange_info() -> Dict[str, Any]:
    """Get exchange trading rules and symbol information."""
    logger.info("Fetching exchange info")
    endpoint = f"{BASE_URL}/exchangeInfo"
    response = requests.get(endpoint)
    return handle_response(response)

@retry_request
def get_recent_trades(symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Get recent trades for a symbol."""
    logger.info(f"Fetching recent trades for symbol: {symbol}, limit: {limit}")
    endpoint = f"{BASE_URL}/trades"
    params = {"symbol": symbol, "limit": limit}
    response = requests.get(endpoint, params=params)
    return handle_response(response)

def get_server_time() -> int:
    """Get the current server time."""
    logger.info("Fetching server time")
    endpoint = f"{BASE_URL}/time"
    response = requests.get(endpoint)
    return handle_response(response)["serverTime"]

def check_symbol_validity(symbol: str) -> bool:
    """Check if a symbol is valid."""
    logger.info(f"Checking validity of symbol: {symbol}")
    try:
        get_symbol_price(symbol)
        return True
    except BinanceAPIException:
        return False