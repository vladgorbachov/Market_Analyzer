# src/data_collection/coinmarketcap_api.py

import requests
import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("COINMARKETCAP_API_KEY")
if not API_KEY:
    raise ValueError("COINMARKETCAP_API_KEY not found in environment variables")

BASE_URL = "https://pro-api.coinmarketcap.com/v1"

headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY,
}


class CoinMarketCapAPIException(Exception):
    pass


def handle_response(response: requests.Response) -> Dict[str, Any]:
    if response.status_code == 200:
        return response.json()
    else:
        raise CoinMarketCapAPIException(f"API request failed with status code {response.status_code}: {response.text}")


def get_latest_listings(limit: int = 100) -> Dict[str, Any]:
    """
    Get the latest listings from CoinMarketCap.

    :param limit: Number of results to return. Default is 100.
    :return: Dictionary containing the latest listings data.
    """
    logger.info(f"Fetching latest listings with limit: {limit}")
    url = f"{BASE_URL}/cryptocurrency/listings/latest"
    params = {"limit": limit}
    try:
        response = requests.get(url, headers=headers, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching latest listings: {str(e)}")
        raise CoinMarketCapAPIException(f"Failed to fetch latest listings: {str(e)}")


def get_metadata(symbol: str) -> Dict[str, Any]:
    """
    Get metadata for a specific cryptocurrency.

    :param symbol: The symbol of the cryptocurrency (e.g., 'BTC').
    :return: Dictionary containing the metadata for the specified cryptocurrency.
    """
    logger.info(f"Fetching metadata for symbol: {symbol}")
    url = f"{BASE_URL}/cryptocurrency/info"
    params = {"symbol": symbol}
    try:
        response = requests.get(url, headers=headers, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching metadata for {symbol}: {str(e)}")
        raise CoinMarketCapAPIException(f"Failed to fetch metadata for {symbol}: {str(e)}")


def get_global_metrics() -> Dict[str, Any]:
    """
    Get global cryptocurrency market metrics.

    :return: Dictionary containing global market metrics.
    """
    logger.info("Fetching global market metrics")
    url = f"{BASE_URL}/global-metrics/quotes/latest"
    try:
        response = requests.get(url, headers=headers)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching global market metrics: {str(e)}")
        raise CoinMarketCapAPIException(f"Failed to fetch global market metrics: {str(e)}")


def get_exchange_listings(limit: int = 100) -> Dict[str, Any]:
    """
    Get the latest exchange listings from CoinMarketCap.

    :param limit: Number of results to return. Default is 100.
    :return: Dictionary containing the latest exchange listings data.
    """
    logger.info(f"Fetching exchange listings with limit: {limit}")
    url = f"{BASE_URL}/exchange/listings/latest"
    params = {"limit": limit}
    try:
        response = requests.get(url, headers=headers, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error fetching exchange listings: {str(e)}")
        raise CoinMarketCapAPIException(f"Failed to fetch exchange listings: {str(e)}")


def get_price_conversion(amount: float, symbol: str, convert_to: str = 'USD') -> Dict[str, Any]:
    """
    Convert an amount of one cryptocurrency to another currency.

    :param amount: The amount to convert.
    :param symbol: The symbol of the cryptocurrency to convert from.
    :param convert_to: The symbol of the currency to convert to. Default is 'USD'.
    :return: Dictionary containing the conversion data.
    """
    logger.info(f"Converting {amount} {symbol} to {convert_to}")
    url = f"{BASE_URL}/tools/price-conversion"
    params = {
        "amount": amount,
        "symbol": symbol,
        "convert": convert_to
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        return handle_response(response)
    except requests.RequestException as e:
        logger.error(f"Error converting {amount} {symbol} to {convert_to}: {str(e)}")
        raise CoinMarketCapAPIException(f"Failed to convert {amount} {symbol} to {convert_to}: {str(e)}")