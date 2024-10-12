# src/feature_engineering/technical_indicators.py

import pandas as pd
import numpy as np
from typing import Union
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sma(data: pd.Series, window: int) -> pd.Series:
    logger.info(f"Calculating SMA with window {window}")
    return data.rolling(window=window).mean()

def ema(data: pd.Series, span: int) -> pd.Series:
    logger.info(f"Calculating EMA with span {span}")
    return data.ewm(span=span, adjust=False).mean()

def rsi(data: pd.Series, window: int) -> pd.Series:
    logger.info(f"Calculating RSI with window {window}")
    delta = data.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))

    rsi[:window] = np.nan  # Устанавливаем первые window значений как NaN
    return rsi.clip(0, 100)

def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    logger.info(f"Calculating MACD with fast={fast}, slow={slow}, signal={signal}")
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def bollinger_bands(data: pd.Series, window: int, num_std: float = 2) -> pd.DataFrame:
    logger.info(f"Calculating Bollinger Bands with window={window}, num_std={num_std}")
    sma_line = sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma_line + (std * num_std)
    lower_band = sma_line - (std * num_std)
    return pd.DataFrame({
        'upper': upper_band,
        'middle': sma_line,
        'lower': lower_band
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    logger.info(f"Calculating ATR with window {window}")
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.rolling(window=window).mean()
    atr_values[:window] = np.nan  # Устанавливаем первые window значений как NaN
    return atr_values

def add_all_indicators(df: pd.DataFrame, close_column: str = 'close', high_column: str = 'high',
                       low_column: str = 'low') -> pd.DataFrame:
    logger.info("Adding all technical indicators")
    try:
        df = df.copy()  # Создаем копию, чтобы не изменять оригинальный DataFrame

        df['SMA_20'] = sma(df[close_column], 20)
        df['EMA_20'] = ema(df[close_column], 20)
        df['RSI_14'] = rsi(df[close_column], 14)

        macd_data = macd(df[close_column])
        df = pd.concat([df, macd_data], axis=1)

        bb_data = bollinger_bands(df[close_column], 20)
        df = pd.concat([df, bb_data], axis=1)

        if high_column in df.columns and low_column in df.columns:
            df['ATR_14'] = atr(df[high_column], df[low_column], df[close_column], 14)
        else:
            logger.warning(f"Columns {high_column} or {low_column} not found. ATR not calculated.")

        return df
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        raise

def momentum(data: pd.Series, period: int = 14) -> pd.Series:
    logger.info(f"Calculating Momentum with period {period}")
    return data.diff(period)

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                          k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    logger.info(f"Calculating Stochastic Oscillator with k_window={k_window}, d_window={d_window}")
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k = k.clip(0, 100)  # Ограничиваем значения в пределах [0, 100]
    d = k.rolling(window=d_window).mean()
    d = d.clip(0, 100)  # Ограничиваем значения в пределах [0, 100]

    return pd.DataFrame({'%K': k, '%D': d})