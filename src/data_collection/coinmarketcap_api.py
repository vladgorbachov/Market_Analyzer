# src/data_collection/coinmarketcap_api.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COINMARKETCAP_API_KEY")
BASE_URL = "https://pro-api.coinmarketcap.com/v1"


def get_latest_listings(limit=100):
    url = f"{BASE_URL}/cryptocurrency/listings/latest"
    headers = {
        "X-CMC_PRO_API_KEY": API_KEY,
    }
    params = {
        "limit": limit,
        "convert": "USD"
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


# Добавьте другие необходимые функции для работы с CoinMarketCap API

