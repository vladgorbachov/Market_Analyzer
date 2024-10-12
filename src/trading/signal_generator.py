# src/trading/signal_generator.py

import pandas as pd
import numpy as np
from typing import List, Dict, Callable
from enum import Enum
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalGenerator:
    def __init__(self):
        self.strategies = {}

    def add_strategy(self, name: str, strategy: Callable):
        """
        Добавляет новую торговую стратегию.

        :param name: Название стратегии.
        :param strategy: Функция стратегии.
        """
        logger.info(f"Adding new strategy: {name}")
        self.strategies[name] = strategy

    def generate_signals(self, data: pd.DataFrame, strategy_name: str) -> pd.Series:
        """
        Генерирует торговые сигналы, используя указанную стратегию.

        :param data: DataFrame с данными для генерации сигналов.
        :param strategy_name: Название стратегии для использования.
        :return: Series с сгенерированными сигналами.
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy '{strategy_name}' not found.")
            raise ValueError(f"Strategy '{strategy_name}' not found.")

        logger.info(f"Generating signals using strategy: {strategy_name}")
        try:
            return self.strategies[strategy_name](data)
        except Exception as e:
            logger.error(f"Error generating signals with strategy {strategy_name}: {str(e)}")
            raise

    def combine_signals(self, signals: List[pd.Series], method: str = 'majority') -> pd.Series:
        """
        Комбинирует сигналы от нескольких стратегий.

        :param signals: Список Series с сигналами от разных стратегий.
        :param method: Метод комбинирования ('majority', 'unanimous', или 'any').
        :return: Series с комбинированными сигналами.
        """
        logger.info(f"Combining signals using method: {method}")
        try:
            combined = pd.concat(signals, axis=1).fillna(Signal.HOLD.value)
            if method == 'majority':
                mode = combined.mode(axis=1)[0].fillna(Signal.HOLD.value)
                return pd.Series(mode, index=combined.index).map(lambda x: Signal(x).value)
            elif method == 'unanimous':
                return (combined == Signal.BUY.value).all(axis=1).map(
                    {True: Signal.BUY.value, False: Signal.HOLD.value})
            elif method == 'any':
                return (combined == Signal.BUY.value).any(axis=1).map(
                    {True: Signal.BUY.value, False: Signal.HOLD.value})
            else:
                logger.error(f"Unknown combination method: {method}")
                raise ValueError(f"Unknown combination method: {method}")
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            raise

    def backtest_signals(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
        """
        Проводит бэктестинг на основе сгенерированных сигналов.

        :param data: DataFrame с ценовыми данными.
        :param signals: Series с торговыми сигналами.
        :return: Словарь с результатами бэктестинга.
        """
        if len(data) != len(signals):
            logger.error("Data and signals must have the same length")
            raise ValueError("Data and signals must have the same length")

        logger.info("Performing backtest on generated signals")
        try:
            initial_capital = 10000.0
            position = 0
            capital = initial_capital
            trades = 0

            for i in range(1, len(data)):
                if signals.iloc[i - 1] == Signal.BUY.value and position == 0:
                    position = capital / data['close'].iloc[i]
                    capital = 0
                    trades += 1
                elif signals.iloc[i - 1] == Signal.SELL.value and position > 0:
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
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    @staticmethod
    def trend_following_strategy(data: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.Series:
        """
        Стратегия следования за трендом.

        :param data: DataFrame с ценовыми данными.
        :param short_window: Короткое окно для скользящей средней.
        :param long_window: Длинное окно для скользящей средней.
        :return: Series с сигналами.
        """
        logger.info(f"Applying trend following strategy with short window {short_window} and long window {long_window}")
        try:
            short_ma = data['close'].rolling(window=short_window).mean()
            long_ma = data['close'].rolling(window=long_window).mean()
            signals = pd.Series(index=data.index, data=Signal.HOLD.value)
            signals[short_ma > long_ma] = Signal.BUY.value
            signals[short_ma < long_ma] = Signal.SELL.value
            return signals
        except Exception as e:
            logger.error(f"Error in trend following strategy: {str(e)}")
            raise

    @staticmethod
    def mean_reversion_strategy(data: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.Series:
        """
        Стратегия возврата к среднему.

        :param data: DataFrame с ценовыми данными.
        :param window: Окно для расчета скользящей средней.
        :param std_dev: Количество стандартных отклонений для генерации сигналов.
        :return: Series с сигналами.
        """
        logger.info(f"Applying mean reversion strategy with window {window} and std_dev {std_dev}")
        try:
            rolling_mean = data['close'].rolling(window=window).mean()
            rolling_std = data['close'].rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            signals = pd.Series(index=data.index, data=Signal.HOLD.value)
            signals[data['close'] > upper_band] = Signal.SELL.value
            signals[data['close'] < lower_band] = Signal.BUY.value
            return signals
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {str(e)}")
            raise

    @staticmethod
    def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: int = 70, oversold: int = 30) -> pd.Series:
        """
        Стратегия на основе индикатора RSI.

        :param data: DataFrame с ценовыми данными.
        :param window: Окно для расчета RSI.
        :param overbought: Уровень перекупленности.
        :param oversold: Уровень перепроданности.
        :return: Series с сигналами.
        """
        logger.info(f"Applying RSI strategy with window {window}, overbought {overbought}, oversold {oversold}")
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            signals = pd.Series(index=data.index, data=Signal.HOLD.value)
            signals[rsi > overbought] = Signal.SELL.value
            signals[rsi < oversold] = Signal.BUY.value
            return signals
        except Exception as e:
            logger.error(f"Error in RSI strategy: {str(e)}")
            raise

    @staticmethod
    def ml_model_strategy(data: pd.DataFrame, predictions: pd.Series, threshold: float = 0.01) -> pd.Series:
        """
        Стратегия на основе предсказаний машинного обучения.

        :param data: DataFrame с ценовыми данными.
        :param predictions: Series с предсказаниями модели.
        :param threshold: Пороговое значение для генерации сигналов.
        :return: Series с сигналами.
        """
        logger.info(f"Applying ML model strategy with threshold {threshold}")
        try:
            signals = pd.Series(index=data.index, data=Signal.HOLD.value)
            signals[predictions > threshold] = Signal.BUY.value
            signals[predictions < -threshold] = Signal.SELL.value
            return signals
        except Exception as e:
            logger.error(f"Error in ML model strategy: {str(e)}")
            raise

