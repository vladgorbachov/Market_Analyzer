import pytest
import pandas as pd
import numpy as np
from src.trading.signal_generator import SignalGenerator, Signal

def test_generate_signals(signal_generator, sample_feature_data):
    signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
    signals = signal_generator.generate_signals(sample_feature_data, 'trend_following')

    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_feature_data)

def test_combine_signals(signal_generator, sample_feature_data):
    signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
    signal_generator.add_strategy('mean_reversion', SignalGenerator.mean_reversion_strategy)

    signals1 = signal_generator.generate_signals(sample_feature_data, 'trend_following')
    signals2 = signal_generator.generate_signals(sample_feature_data, 'mean_reversion')

    combined_signals = signal_generator.combine_signals([signals1, signals2])
    assert isinstance(combined_signals, pd.Series)
    assert combined_signals.isin([signal.value for signal in Signal]).all()
    assert len(combined_signals) == len(sample_feature_data)

def test_backtest_signals(signal_generator, sample_feature_data):
    signals = pd.Series(index=sample_feature_data.index,
                        data=np.random.choice([signal.value for signal in Signal], len(sample_feature_data)))
    backtest_results = signal_generator.backtest_signals(sample_feature_data, signals)

    assert isinstance(backtest_results, dict)
    assert set(backtest_results.keys()) == {'Total Return', 'Sharpe Ratio', 'Final Capital', 'Number of Trades'}

def test_trend_following_strategy(sample_feature_data):
    signals = SignalGenerator.trend_following_strategy(sample_feature_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_feature_data)

def test_mean_reversion_strategy(sample_feature_data):
    signals = SignalGenerator.mean_reversion_strategy(sample_feature_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_feature_data)

def test_rsi_strategy(sample_feature_data):
    signals = SignalGenerator.rsi_strategy(sample_feature_data)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_feature_data)

def test_ml_model_strategy(sample_feature_data):
    predictions = pd.Series(np.random.randn(len(sample_feature_data)), index=sample_feature_data.index)
    signals = SignalGenerator.ml_model_strategy(sample_feature_data, predictions)
    assert isinstance(signals, pd.Series)
    assert signals.isin([signal.value for signal in Signal]).all()
    assert len(signals) == len(sample_feature_data)