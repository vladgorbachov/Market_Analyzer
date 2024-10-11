from . import binance_api, coinmarketcap_api, okx_api
from typing import Dict, Any, List

class APIManager:
    def __init__(self):
        self.binance = binance_api
        self.coinmarketcap = coinmarketcap_api
        self.okx = okx_api

    def get_price_data(self, symbol: str) -> Dict[str, Any]:
        binance_data = self.binance.get_symbol_price(symbol)
        okx_data = self.okx.get_ticker(symbol)
        return {
            "binance": binance_data,
            "okx": okx_data
        }

    def get_market_overview(self, limit: int = 100) -> Dict[str, Any]:
        return self.coinmarketcap.get_latest_listings(limit)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        binance_order_book = self.binance.get_order_book(symbol, limit)
        okx_order_book = self.okx.get_order_book(symbol, limit)
        return {
            "binance": binance_order_book,
            "okx": okx_order_book
        }

    def get_historical_data(self, symbol: str, interval: str, start_time: int, end_time: int) -> Dict[str, List[List]]:
        binance_data = self.binance.get_historical_klines(symbol, interval, start_time, end_time)
        okx_data = self.okx.get_candlesticks(symbol, interval)
        return {
            "binance": binance_data,
            "okx": okx_data
        }

    def get_crypto_metadata(self, symbol: str) -> Dict[str, Any]:
        return self.coinmarketcap.get_metadata(symbol)

    def get_global_market_metrics(self) -> Dict[str, Any]:
        return self.coinmarketcap.get_global_metrics()

    def get_exchange_info(self) -> Dict[str, Any]:
        binance_info = self.binance.get_exchange_info()
        cmc_exchange_listings = self.coinmarketcap.get_exchange_listings()
        return {
            "binance": binance_info,
            "coinmarketcap": cmc_exchange_listings
        }

    def get_funding_rates(self, symbol: str) -> Dict[str, Any]:
        return self.okx.get_funding_rate(symbol)