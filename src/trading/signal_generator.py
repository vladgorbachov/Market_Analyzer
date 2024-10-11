import pandas as pd
import numpy as np
from typing import List, Dict, Callable
from enum import Enum

class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class SignalGenerator:
    def __init__(self):
        self.strategies = {}

    def add_strategy(self, name: str, strategy: Callable):
        self.strategies[name] = strategy

    def generate_signals(self, data: pd.DataFrame, strategy_name: str) -> pd.Series:
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        return self.strategies[strategy_name](data)

    def combine_signals(self, signals: List[pd.Series], method: str = 'majority') -> pd.Series:
        combined = pd.concat(signals, axis=1).fillna(Signal.HOLD.value)  # Заполняем NaN значениями HOLD
        if method == 'majority':
            mode = combined.mode(axis=1)[0].fillna(Signal.HOLD.value)
            return pd.Series(mode, index=combined.index).map(lambda x: Signal(x).value)
        elif method == 'unanimous':
            return (combined == Signal.BUY.value).all(axis=1).map({True: Signal.BUY.value, False: Signal.HOLD.value})
        elif method == 'any':
            return (combined == Signal.BUY.value).any(axis=1).map({True: Signal.BUY.value, False: Signal.HOLD.value})
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def backtest_signals(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
        if len(data) != len(signals):
            raise ValueError("Data and signals must have the same length")

        initial_capital = 10000.0
        position = 0
        capital = initial_capital
        trades = 0

        for i in range(1, len(data)):
            if signals.iloc[i-1] == Signal.BUY.value and position == 0:
                position = capital / data['close'].iloc[i]
                capital = 0
                trades += 1
            elif signals.iloc[i-1] == Signal.SELL.value and position > 0:
                capital = position * data['close'].iloc[i]
                position = 0
                trades += 1

        if position > 0:
            capital = position * data['close'].iloc[-1]

        total_return = (capital - initial_capital) / initial_capital
        std_dev = np.std(data['close'].pct_change()) * np.sqrt(252)
        sharpe_ratio = 0 if std_dev == 0 else np.sqrt(252) * total_return / std_dev

        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Final Capital': capital,
            'Number of Trades': trades
        }

    @staticmethod
    def trend_following_strategy(data: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.Series:
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        signals = pd.Series(index=data.index, data=Signal.HOLD.value)
        signals[short_ma > long_ma] = Signal.BUY.value
        signals[short_ma < long_ma] = Signal.SELL.value
        return signals.fillna(Signal.HOLD.value)

    @staticmethod
    def mean_reversion_strategy(data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.Series:
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        signals = pd.Series(index=data.index, data=Signal.HOLD.value)
        signals[data['close'] > upper_band] = Signal.SELL.value
        signals[data['close'] < lower_band] = Signal.BUY.value
        return signals.fillna(Signal.HOLD.value)

    @staticmethod
    def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 70, oversold: int = 30) -> pd.Series:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals = pd.Series(index=data.index, data=Signal.HOLD.value)
        signals[rsi > overbought] = Signal.SELL.value
        signals[rsi < oversold] = Signal.BUY.value
        return signals.fillna(Signal.HOLD.value)

    @staticmethod
    def ml_model_strategy(data: pd.DataFrame, predictions: pd.Series, threshold: float = 0.01) -> pd.Series:
        signals = pd.Series(index=data.index, data=Signal.HOLD.value)
        signals[predictions > threshold] = Signal.BUY.value
        signals[predictions < -threshold] = Signal.SELL.value
        return signals.fillna(Signal.HOLD.value)
