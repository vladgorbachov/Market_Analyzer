# src/data_collection/okx_api.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
BASE_URL = "https://www.okx.com"


def get_ticker(symbol):
    url = f"{BASE_URL}/api/v5/market/ticker"
    params = {"instId": symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Добавьте другие необходимые функции для работы с OKX API

