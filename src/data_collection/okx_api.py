# src/data_collection/okx_api.py

import requests
import os
import time
import hmac
import base64
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

if not all([API_KEY, API_SECRET, PASSPHRASE]):
    raise ValueError("OKX API credentials not found in environment variables")

BASE_URL = "https://www.okx.com"


class OKXAPIException(Exception):
    pass


def generate_signature(timestamp: str, method: str, request_path: str, body: str = '') -> str:
    message = timestamp + method + request_path + (body or '')
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d).decode()


def get_header(method: str, request_path: str, body: str = '') -> Dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, request_path, body)

    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        'Content-Type': 'application/json'
    }


def handle_response(response: requests.Response) -> Dict[str, Any]:
    if response.status_code == 200:
        return response.json()
    else:
        raise OKXAPIException(f"API request failed with status code {response.status_code}: {response.text}")


def get_ticker(symbol: str) -> Dict[str, Any]:
    """
    Get the latest ticker information for a symbol.

    :param symbol: The trading pair symbol (e.g., 'BTC-USDT').
    :return: Dictionary containing ticker information.
    """
    logger.info(f"Fetching ticker for symbol: {symbol}")
    url = f"{BASE_URL}/api/v5/market/ticker"
    params = {"instId": symbol}
    try:
        response = requests.get(url, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
        raise OKXAPIException(f"Failed to fetch ticker for {symbol}: {str(e)}")


def get_account_balance() -> Dict[str, Any]:
    """
    Get the account balance information.

    :return: Dictionary containing account balance information.
    """
    logger.info("Fetching account balance")
    method = 'GET'
    request_path = '/api/v5/account/balance'
    url = f"{BASE_URL}{request_path}"
    headers = get_header(method, request_path)
    try:
        response = requests.get(url, headers=headers)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching account balance: {str(e)}")
        raise OKXAPIException(f"Failed to fetch account balance: {str(e)}")


def get_order_book(symbol: str, depth: int = 20) -> Dict[str, Any]:
    """
    Get the order book for a symbol.

    :param symbol: The trading pair symbol (e.g., 'BTC-USDT').
    :param depth: The depth of the order book to retrieve. Default is 20.
    :return: Dictionary containing order book information.
    """
    logger.info(f"Fetching order book for symbol: {symbol}, depth: {depth}")
    url = f"{BASE_URL}/api/v5/market/books"
    params = {"instId": symbol, "sz": depth}
    try:
        response = requests.get(url, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching order book for {symbol}: {str(e)}")
        raise OKXAPIException(f"Failed to fetch order book for {symbol}: {str(e)}")


def get_candlesticks(symbol: str, interval: str = '1m', limit: int = 100) -> Dict[str, Any]:
    """
    Get candlestick data for a symbol.

    :param symbol: The trading pair symbol (e.g., 'BTC-USDT').
    :param interval: The time interval for each candlestick. Default is '1m'.
    :param limit: The number of candlesticks to retrieve. Default is 100.
    :return: Dictionary containing candlestick data.
    """
    logger.info(f"Fetching candlesticks for symbol: {symbol}, interval: {interval}, limit: {limit}")
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}
    try:
        response = requests.get(url, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching candlesticks for {symbol}: {str(e)}")
        raise OKXAPIException(f"Failed to fetch candlesticks for {symbol}: {str(e)}")


def get_funding_rate(symbol: str) -> Dict[str, Any]:
    """
    Get the funding rate for a perpetual swap.

    :param symbol: The trading pair symbol (e.g., 'BTC-USD-SWAP').
    :return: Dictionary containing funding rate information.
    """
    logger.info(f"Fetching funding rate for symbol: {symbol}")
    url = f"{BASE_URL}/api/v5/public/funding-rate"
    params = {"instId": symbol}
    try:
        response = requests.get(url, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching funding rate for {symbol}: {str(e)}")
        raise OKXAPIException(f"Failed to fetch funding rate for {symbol}: {str(e)}")


def place_order(symbol: str, side: str, order_type: str, size: float, price: Optional[float] = None) -> Dict[str, Any]:
    """
    Place a new order.

    :param symbol: The trading pair symbol (e.g., 'BTC-USDT').
    :param side: The side of the order ('buy' or 'sell').
    :param order_type: The type of the order ('market' or 'limit').
    :param size: The amount of the asset to buy or sell.
    :param price: The price for limit orders. Optional for market orders.
    :return: Dictionary containing order information.
    """
    logger.info(f"Placing {order_type} {side} order for {symbol}, size: {size}, price: {price}")
    method = 'POST'
    request_path = '/api/v5/trade/order'
    url = f"{BASE_URL}{request_path}"
    body = {
        "instId": symbol,
        "tdMode": "cash",
        "side": side,
        "ordType": order_type,
        "sz": str(size)
    }
    if price is not None:
        body["px"] = str(price)

    headers = get_header(method, request_path, str(body))
    try:
        response = requests.post(url, headers=headers, json=body)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error placing order for {symbol}: {str(e)}")
        raise OKXAPIException(f"Failed to place order for {symbol}: {str(e)}")