import requests
import os
import time
import hmac
import base64
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")
BASE_URL = "https://www.okx.com"


def generate_signature(timestamp, method, request_path, body=''):
    message = timestamp + method + request_path + (body or '')
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d).decode()


def get_header(method, request_path, body=''):
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, request_path, body)

    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        'Content-Type': 'application/json'
    }


def get_ticker(symbol):
    url = f"{BASE_URL}/api/v5/market/ticker"
    params = {"instId": symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_account_balance():
    method = 'GET'
    request_path = '/api/v5/account/balance'
    url = f"{BASE_URL}{request_path}"
    headers = get_header(method, request_path)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_order_book(symbol, depth=20):
    url = f"{BASE_URL}/api/v5/market/books"
    params = {"instId": symbol, "sz": depth}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_candlesticks(symbol, interval='1m', limit=100):
    url = f"{BASE_URL}/api/v5/market/candles"
    params = {"instId": symbol, "bar": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_funding_rate(symbol):
    url = f"{BASE_URL}/api/v5/public/funding-rate"
    params = {"instId": symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()