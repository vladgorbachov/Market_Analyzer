# src/data_collection/api_manager.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from . import binance_api, coinmarketcap_api, okx_api

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIManager:
    def __init__(self):
        self.binance = binance_api
        self.coinmarketcap = coinmarketcap_api
        self.okx = okx_api

    def get_price_data(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"Fetching price data for {symbol}")
        try:
            binance_data = self.binance.get_symbol_price(symbol)
            okx_data = self.okx.get_ticker(symbol)
            return {
                "binance": binance_data,
                "okx": okx_data
            }
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {str(e)}")
            raise

    def get_market_overview(self, limit: int = 100) -> Dict[str, Any]:
        logger.info(f"Fetching market overview with limit {limit}")
        try:
            return self.coinmarketcap.get_latest_listings(limit)
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            raise

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        logger.info(f"Fetching order book for {symbol} with limit {limit}")
        try:
            binance_order_book = self.binance.get_order_book(symbol, limit)
            okx_order_book = self.okx.get_order_book(symbol, limit)
            return {
                "binance": binance_order_book,
                "okx": okx_order_book
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise

    def get_historical_data(self, symbol: str, interval: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> Dict[str, List[List[Any]]]:
        logger.info(f"Fetching historical data for {symbol} with interval {interval}")
        try:
            if start_time is None:
                start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            if end_time is None:
                end_time = int(datetime.now().timestamp() * 1000)

            binance_data = self.binance.get_historical_klines(symbol, interval, start_time, end_time)
            okx_data = self.okx.get_candlesticks(symbol, interval)
            return {
                "binance": binance_data,
                "okx": okx_data
            }
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    def get_crypto_metadata(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"Fetching metadata for {symbol}")
        try:
            return self.coinmarketcap.get_metadata(symbol)
        except Exception as e:
            logger.error(f"Error fetching metadata for {symbol}: {str(e)}")
            raise

    def get_global_market_metrics(self) -> Dict[str, Any]:
        logger.info("Fetching global market metrics")
        try:
            return self.coinmarketcap.get_global_metrics()
        except Exception as e:
            logger.error(f"Error fetching global market metrics: {str(e)}")
            raise

    def get_exchange_info(self) -> Dict[str, Any]:
        logger.info("Fetching exchange info")
        try:
            binance_info = self.binance.get_exchange_info()
            cmc_exchange_listings = self.coinmarketcap.get_exchange_listings()
            return {
                "binance": binance_info,
                "coinmarketcap": cmc_exchange_listings
            }
        except Exception as e:
            logger.error(f"Error fetching exchange info: {str(e)}")
            raise

    def get_funding_rates(self, symbol: str) -> Dict[str, Any]:
        logger.info(f"Fetching funding rates for {symbol}")
        try:
            return self.okx.get_funding_rate(symbol)
        except Exception as e:
            logger.error(f"Error fetching funding rates for {symbol}: {str(e)}")
            raise