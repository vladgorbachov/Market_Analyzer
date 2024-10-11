import pandas as pd
import numpy as np
from typing import Union

def sma(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def ema(data: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=span, adjust=False).mean()


def rsi(data: pd.Series, window: int) -> pd.Series:
    delta = data.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)

    rsi = 100 - (100 / (1 + rs))

    rsi[:window - 1] = np.nan

    return rsi.clip(0, 100)

def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
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
    """Bollinger Bands"""
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
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def add_all_indicators(df: pd.DataFrame, close_column: str = 'close', high_column: str = 'high',
                       low_column: str = 'low') -> pd.DataFrame:
    """Add all technical indicators to the dataframe"""
    df['SMA_20'] = sma(df[close_column], 20)
    df['EMA_20'] = ema(df[close_column], 20)
    df['RSI_14'] = rsi(df[close_column], 14)

    macd_data = macd(df[close_column])
    df = pd.concat([df, macd_data], axis=1)

    bb_data = bollinger_bands(df[close_column], 20)
    df = pd.concat([df, bb_data], axis=1)

    if high_column in df.columns and low_column in df.columns:
        df['ATR_14'] = atr(df[high_column], df[low_column], df[close_column], 14)

    return df