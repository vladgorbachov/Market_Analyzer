import pandas as pd
import numpy as np
from typing import Dict, Any, List

def harmonize_price_data(data: Dict[str, Any]) -> pd.DataFrame:
    harmonized_data = []
    for source, price_info in data.items():
        if source == 'binance':
            harmonized_data.append({
                'source': source,
                'symbol': price_info.get('symbol', 'N/A'),
                'price': float(price_info.get('price', 0.0)),
                'timestamp': pd.to_datetime(pd.to_numeric(price_info.get('time', pd.Timestamp.now().timestamp() * 1000)), unit='ms')
            })
        elif source == 'okx':
            harmonized_data.append({
                'source': source,
                'symbol': price_info.get('instId', 'N/A'),
                'price': float(price_info.get('last', 0.0)),
                'timestamp': pd.to_datetime(pd.to_numeric(price_info.get('ts', pd.Timestamp.now().timestamp() * 1000)), unit='ms')
            })
    return pd.DataFrame(harmonized_data)



def harmonize_order_book(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Приводит данные ордербука из разных источников к единому формату.
    """
    harmonized_data = []
    for source, order_book in data.items():
        if source == 'binance':
            for bid in order_book['bids']:
                harmonized_data.append({
                    'source': source,
                    'type': 'bid',
                    'price': float(bid[0]),
                    'quantity': float(bid[1])
                })
            for ask in order_book['asks']:
                harmonized_data.append({
                    'source': source,
                    'type': 'ask',
                    'price': float(ask[0]),
                    'quantity': float(ask[1])
                })
        elif source == 'okx':
            for bid in order_book['bids']:
                harmonized_data.append({
                    'source': source,
                    'type': 'bid',
                    'price': float(bid[0]),
                    'quantity': float(bid[1])
                })
            for ask in order_book['asks']:
                harmonized_data.append({
                    'source': source,
                    'type': 'ask',
                    'price': float(ask[0]),
                    'quantity': float(ask[1])
                })
        # Добавьте обработку данных из других источников при необходимости

    return pd.DataFrame(harmonized_data)


def harmonize_historical_data(data: Dict[str, List[List]]) -> pd.DataFrame:
    """
    Приводит исторические данные из разных источников к единому формату.
    """
    harmonized_data = []
    for source, candles in data.items():
        if source == 'binance':
            for candle in candles:
                try:
                    # Проверка на корректность данных перед преобразованием
                    timestamp = pd.to_datetime(candle[0], unit='ms')
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                except (ValueError, TypeError):
                    # Если данные некорректны, пропускаем эту свечу
                    continue

                harmonized_data.append({
                    'source': source,
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        elif source == 'okx':
            for candle in candles:
                try:
                    # Проверка на корректность данных перед преобразованием
                    timestamp = pd.to_datetime(candle[0], unit='ms')
                    open_price = float(candle[1])
                    high_price = float(candle[2])
                    low_price = float(candle[3])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                except (ValueError, TypeError):
                    # Если данные некорректны, пропускаем эту свечу
                    continue

                harmonized_data.append({
                    'source': source,
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        # Добавьте обработку данных из других источников при необходимости

    return pd.DataFrame(harmonized_data)


def harmonize_cmc_listings(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Гармонизирует данные о листингах криптовалют с CoinMarketCap.
    """
    listings = data.get('data', [])
    harmonized_data = []
    for listing in listings:
        harmonized_data.append({
            'id': listing['id'],
            'name': listing['name'],
            'symbol': listing['symbol'],
            'slug': listing['slug'],
            'cmc_rank': listing['cmc_rank'],
            'market_cap': listing['quote']['USD']['market_cap'],
            'price': listing['quote']['USD']['price'],
            'volume_24h': listing['quote']['USD']['volume_24h'],
            'percent_change_1h': listing['quote']['USD']['percent_change_1h'],
            'percent_change_24h': listing['quote']['USD']['percent_change_24h'],
            'percent_change_7d': listing['quote']['USD']['percent_change_7d'],
            'last_updated': pd.to_datetime(listing['last_updated'])
        })
    return pd.DataFrame(harmonized_data)

def harmonize_cmc_metadata(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Гармонизирует метаданные криптовалют с CoinMarketCap.
    """
    metadata = data.get('data', {})
    harmonized_data = []
    for symbol, info in metadata.items():
        harmonized_data.append({
            'id': info['id'],
            'name': info['name'],
            'symbol': info['symbol'],
            'category': info['category'],
            'description': info['description'],
            'slug': info['slug'],
            'logo': info['logo'],
            'subreddit': info['subreddit'],
            'notice': info['notice'],
            'tags': ', '.join(info['tags']),
            'platform': info['platform']['name'] if info['platform'] else None,
            'date_added': pd.to_datetime(info['date_added']),
            'twitter_username': info['twitter_username'],
            'is_hidden': info['is_hidden']
        })
    return pd.DataFrame(harmonized_data)

def harmonize_cmc_global_metrics(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Гармонизирует глобальные метрики рынка с CoinMarketCap.
    """
    metrics = data.get('data', {})
    return pd.DataFrame([{
        'total_cryptocurrencies': metrics['total_cryptocurrencies'],
        'active_cryptocurrencies': metrics['active_cryptocurrencies'],
        'total_exchanges': metrics['total_exchanges'],
        'active_exchanges': metrics['active_exchanges'],
        'eth_dominance': metrics['eth_dominance'],
        'btc_dominance': metrics['btc_dominance'],
        'total_market_cap': metrics['quote']['USD']['total_market_cap'],
        'total_volume_24h': metrics['quote']['USD']['total_volume_24h'],
        'altcoin_volume_24h': metrics['quote']['USD']['altcoin_volume_24h'],
        'altcoin_market_cap': metrics['quote']['USD']['altcoin_market_cap'],
        'last_updated': pd.to_datetime(metrics['last_updated'])
    }])

def combine_market_data(price_data: pd.DataFrame, cmc_listings: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет данные о ценах с бирж и листинги с CoinMarketCap.
    """
    # Предполагаем, что в price_data есть колонки 'symbol' и 'price'
    combined_data = price_data.merge(cmc_listings[['symbol', 'market_cap', 'volume_24h', 'percent_change_24h']],
                                     on='symbol', how='left')
    return combined_data

def enrich_data_with_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Обогащает данные метаданными с CoinMarketCap.
    """
    # Предполагаем, что в data есть колонка 'symbol'
    enriched_data = data.merge(metadata[['symbol', 'category', 'tags', 'platform']],
                               on='symbol', how='left')
    return enriched_data

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    if method == 'ffill':
        return df.ffill()
    elif method == 'bfill':
        return df.bfill()
    elif method == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError("Unsupported method for handling missing values")


def normalize_data(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """
    Нормализует числовые данные.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if method == 'minmax':
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
    elif method == 'zscore':
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    else:
        raise ValueError("Unsupported normalization method")
    return df

def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Удаляет выбросы из данных.
    """
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        else:
            raise ValueError("Unsupported method for removing outliers")
    return df

def aggregate_data(df: pd.DataFrame, group_by: str, agg_func: Dict[str, str]) -> pd.DataFrame:
    """
    Агрегирует данные по заданному временному интервалу.
    """
    return df.groupby(group_by).agg(agg_func).reset_index()