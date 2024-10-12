# tests/test_signal_generator.py

import pytest
import pandas as pd
import numpy as np
from src.trading.signal_generator import SignalGenerator, Signal


@pytest.fixture
def signal_generator():
    return SignalGenerator()


@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    close_prices = np.random.randn(100).cumsum() + 100
    return pd.DataFrame({
        'timestamp': dates,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    }).set_index('timestamp')


def test_add_strategy(signal_generator):
    def dummy_strategy(data):
        return pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(data)),
                         index=data.index)

    signal_generator.add_strategy('dummy', dummy_strategy)
    assert 'dummy' in signal_generator.strategies


def test_generate_signals(signal_generator, sample_data):
    signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
    signals = signal_generator.generate_signals(sample_data, 'trend_following')

    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_data)


def test_generate_signals_unknown_strategy(signal_generator, sample_data):
    with pytest.raises(ValueError):
        signal_generator.generate_signals(sample_data, 'unknown_strategy')


def test_combine_signals(signal_generator, sample_data):
    signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
    signal_generator.add_strategy('mean_reversion', SignalGenerator.mean_reversion_strategy)

    signals1 = signal_generator.generate_signals(sample_data, 'trend_following')
    signals2 = signal_generator.generate_signals(sample_data, 'mean_reversion')

    combined_signals = signal_generator.combine_signals([signals1, signals2])
    assert isinstance(combined_signals, pd.Series)
    assert combined_signals.isin([signal.value for signal in Signal]).all()
    assert len(combined_signals) == len(sample_data)


@pytest.mark.parametrize("method", ['majority', 'unanimous', 'any'])
def test_combine_signals_methods(signal_generator, sample_data, method):
    signals1 = pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data)),
                         index=sample_data.index)
    signals2 = pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data)),
                         index=sample_data.index)

    combined_signals = signal_generator.combine_signals([signals1, signals2], method=method)
    assert isinstance(combined_signals, pd.Series)
    assert combined_signals.isin([signal.value for signal in Signal]).all()
    assert len(combined_signals) == len(sample_data)


def test_combine_signals_invalid_method(signal_generator, sample_data):
    signals1 = pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data)),
                         index=sample_data.index)
    signals2 = pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data)),
                         index=sample_data.index)

    with pytest.raises(ValueError):
        signal_generator.combine_signals([signals1, signals2], method='invalid_method')


def test_backtest_signals(signal_generator, sample_data):
    signals = pd.Series(np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data)),
                        index=sample_data.index)
    backtest_results = signal_generator.backtest_signals(sample_data, signals)

    assert isinstance(backtest_results, dict)
    assert set(backtest_results.keys()) == {'Total Return', 'Sharpe Ratio', 'Final Capital', 'Number of Trades'}
    assert all(isinstance(value, (int, float)) for value in backtest_results.values())


def test_backtest_signals_mismatched_length(signal_generator, sample_data):
    signals = pd.Series(
        np.random.choice([Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value], len(sample_data) - 1))
    with pytest.raises(ValueError):
        signal_generator.backtest_signals(sample_data, signals)


def test_trend_following_strategy(sample_data):
    signals = SignalGenerator.trend_following_strategy(sample_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_data)


def test_mean_reversion_strategy(sample_data):
    signals = SignalGenerator.mean_reversion_strategy(sample_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_data)


def test_rsi_strategy(sample_data):
    signals = SignalGenerator.rsi_strategy(sample_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_data)


def test_ml_model_strategy(sample_data):
    predictions = pd.Series(np.random.randn(len(sample_data)), index=sample_data.index)
    signals = SignalGenerator.ml_model_strategy(sample_data, predictions)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_data)


@pytest.mark.parametrize("strategy", [
    SignalGenerator.trend_following_strategy,
    SignalGenerator.mean_reversion_strategy,
    SignalGenerator.rsi_strategy
])
def test_strategies_with_constant_price(strategy):
    constant_data = pd.DataFrame({
        'close': [100] * 100,
        'volume': [1000] * 100
    }, index=pd.date_range(start='2020-01-01', periods=100))

    signals = strategy(constant_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(constant_data)


def test_backtest_signals_all_buy(signal_generator, sample_data):
    all_buy_signals = pd.Series([Signal.BUY.value] * len(sample_data), index=sample_data.index)
    results = signal_generator.backtest_signals(sample_data, all_buy_signals)
    assert results['Number of Trades'] == 1  # Должна быть только одна сделка (покупка в начале)


def test_backtest_signals_alternating(signal_generator, sample_data):
    alternating_signals = pd.Series([Signal.BUY.value, Signal.SELL.value] * (len(sample_data) // 2),
                                    index=sample_data.index)
    results = signal_generator.backtest_signals(sample_data, alternating_signals)
    assert results['Number of Trades'] > 1  # Должно быть несколько сделок

