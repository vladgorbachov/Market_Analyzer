# src/data_collection/api_manager.py
from . import binance_api, coinmarketcap_api, okx_api


class APIManager:
    def __init__(self):
        self.binance = binance_api
        self.coinmarketcap = coinmarketcap_api
        self.okx = okx_api

    def get_price_data(self, symbol):
        binance_data = self.binance.get_symbol_price(symbol)
        okx_data = self.okx.get_ticker(symbol)
        return {
            "binance": binance_data,
            "okx": okx_data
        }

    def get_market_overview(self, limit=100):
        return self.coinmarketcap.get_latest_listings(limit)

    # Добавьте другие методы для агрегации данных из разных источников

