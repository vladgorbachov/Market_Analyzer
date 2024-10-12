# src/data_processing/data_harmonization.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def harmonize_price_data(data: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Harmonizing price data")
    harmonized_data = []
    for source, price_info in data.items():
        try:
            if source == 'binance':
                harmonized_data.append({
                    'source': source,
                    'symbol': price_info.get('symbol', 'N/A'),
                    'price': float(price_info.get('price', 0.0)),
                    'timestamp': pd.to_datetime(price_info.get('time', pd.Timestamp.now().timestamp() * 1000),
                                                unit='ms')
                })
            elif source == 'okx':
                harmonized_data.append({
                    'source': source,
                    'symbol': price_info.get('instId', 'N/A'),
                    'price': float(price_info.get('last', 0.0)),
                    'timestamp': pd.to_datetime(price_info.get('ts', pd.Timestamp.now().timestamp() * 1000), unit='ms')
                })
            # Добавьте обработку данных от других источников по мере необходимости
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing {source} data: {str(e)}")

    result = pd.DataFrame(harmonized_data)
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    return result

def harmonize_order_book(data: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Harmonizing order book data")
    harmonized_data = []
    for source, order_book in data.items():
        try:
            if source in ['binance', 'okx']:
                for bid in order_book.get('bids', []):
                    harmonized_data.append({
                        'source': source,
                        'type': 'bid',
                        'price': float(bid[0]),
                        'quantity': float(bid[1])
                    })
                for ask in order_book.get('asks', []):
                    harmonized_data.append({
                        'source': source,
                        'type': 'ask',
                        'price': float(ask[0]),
                        'quantity': float(ask[1])
                    })
            # Добавьте обработку данных от других источников по мере необходимости
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Error processing {source} order book data: {str(e)}")

    return pd.DataFrame(harmonized_data)

def harmonize_historical_data(data: Dict[str, List[List]]) -> pd.DataFrame:
    logger.info("Harmonizing historical data")
    harmonized_data = []
    for source, candles in data.items():
        for candle in candles:
            try:
                if source == 'binance' or source == 'okx':
                    harmonized_data.append({
                        'source': source,
                        'timestamp': pd.to_datetime(candle[0], unit='ms'),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                # Добавьте обработку данных от других источников по мере необходимости
            except (ValueError, IndexError) as e:
                logger.error(f"Error processing {source} historical data: {str(e)}")

    result = pd.DataFrame(harmonized_data)
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    return result

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    logger.info(f"Handling missing values using method: {method}")
    if method == 'ffill':
        return df.ffill().bfill()  # Чтобы полностью заполнить пропуски
    elif method == 'bfill':
        return df.bfill().ffill()
    elif method == 'interpolate':
        return df.interpolate()
    else:
        logger.warning(f"Unsupported method for handling missing values: {method}. Using 'ffill' instead.")
        return df.ffill().bfill()

def normalize_data(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    logger.info(f"Normalizing data using method: {method}")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if method == 'minmax':
        for column in numeric_columns:
            if df[column].nunique() > 1:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
            else:
                df[column] = 0  # Константные значения нормализуются в 0
    elif method == 'zscore':
        for column in numeric_columns:
            if df[column].nunique() > 1:
                df[column] = (df[column] - df[column].mean()) / df[column].std()
            else:
                df[column] = 0
    else:
        logger.warning(f"Unsupported normalization method: {method}. Data not normalized.")
    return df

def aggregate_data(df: pd.DataFrame, group_by: str, agg_func: Dict[str, str]) -> pd.DataFrame:
    logger.info(f"Aggregating data by {group_by}")
    if group_by in df.columns:
        return df.groupby(group_by).agg(agg_func).reset_index()
    else:
        logger.error(f"Group by column {group_by} not found in DataFrame.")
        return df

