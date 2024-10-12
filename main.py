# main.py

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.data_collection.api_manager import APIManager
from src.data_processing.data_harmonization import harmonize_price_data, harmonize_historical_data
from src.feature_engineering.technical_indicators import add_all_indicators
from src.feature_engineering.text_features import add_text_features
from src.models.model_selector import ModelSelector
from src.evaluation.model_evaluation import ModelEvaluator
from src.trading.signal_generator import SignalGenerator
import logging
import time
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_environment_variables():
    """Загрузка переменных окружения"""
    load_dotenv()
    logger.info("Environment variables loaded")


def initialize_components() -> Dict[str, Any]:
    """Инициализация основных компонентов"""
    try:
        api_manager = APIManager()
        model_selector = ModelSelector()
        model_evaluator = ModelEvaluator(model_selector)
        signal_generator = SignalGenerator()

        return {
            'api_manager': api_manager,
            'model_selector': model_selector,
            'model_evaluator': model_evaluator,
            'signal_generator': signal_generator
        }
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise


def fetch_data(api_manager: APIManager, symbol: str) -> Dict[str, Any]:
    """Получение данных через API"""
    try:
        logger.info(f"Fetching data for symbol: {symbol}")
        price_data = api_manager.get_price_data(symbol)
        start_time = int(time.time() * 1000) - (1000 * 60 * 60 * 24 * 30)  # 30 дней назад
        end_time = int(time.time() * 1000)
        historical_data = api_manager.get_historical_data(symbol, interval="1h", start_time=start_time,
                                                          end_time=end_time)
        market_overview = api_manager.get_market_overview(limit=10)

        return {
            'price_data': price_data,
            'historical_data': historical_data,
            'market_overview': market_overview
        }
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise


def process_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Обработка и подготовка данных"""
    try:
        logger.info("Processing and preparing data")
        harmonized_price_data = harmonize_price_data(data['price_data'])
        harmonized_historical_data = harmonize_historical_data(data['historical_data'])

        data_with_indicators = add_all_indicators(harmonized_historical_data)

        market_overview_df = pd.DataFrame(data['market_overview'].get('data', []))
        if 'description' in market_overview_df.columns:
            data_with_features = add_text_features(data_with_indicators, 'description')
        else:
            data_with_features = data_with_indicators

        return data_with_features
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise


def train_and_evaluate_model(model_selector: ModelSelector, model_evaluator: ModelEvaluator,
                             data: pd.DataFrame, target_column: str, feature_columns: List[str]) -> Dict[str, Any]:
    """Обучение и оценка модели"""
    try:
        logger.info("Training and evaluating model")
        model_selector.fit(data, target_column, feature_columns)
        evaluation_results = model_evaluator.run_full_evaluation(data, target_column, feature_columns)
        return evaluation_results
    except Exception as e:
        logger.error(f"Error training and evaluating model: {str(e)}")
        raise


def generate_trading_signals(signal_generator: SignalGenerator, data: pd.DataFrame,
                             model_selector: ModelSelector, feature_columns: List[str]) -> pd.Series:
    """Генерация торговых сигналов"""
    try:
        logger.info("Generating trading signals")
        signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
        signal_generator.add_strategy('mean_reversion', SignalGenerator.mean_reversion_strategy)
        signal_generator.add_strategy('rsi', SignalGenerator.rsi_strategy)

        trend_following_signals = signal_generator.generate_signals(data, 'trend_following')
        mean_reversion_signals = signal_generator.generate_signals(data, 'mean_reversion')
        rsi_signals = signal_generator.generate_signals(data, 'rsi')

        predictions = model_selector.predict(data[feature_columns])
        ml_model_signals = SignalGenerator.ml_model_strategy(data, predictions)

        combined_signals = signal_generator.combine_signals(
            [trend_following_signals, mean_reversion_signals, rsi_signals, ml_model_signals],
            method='majority'
        )
        return combined_signals
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        raise


def backtest_signals(signal_generator: SignalGenerator, data: pd.DataFrame, signals: pd.Series) -> Dict[str, float]:
    """Проведение бэктестинга сигналов"""
    try:
        logger.info("Backtesting signals")
        return signal_generator.backtest_signals(data, signals)
    except Exception as e:
        logger.error(f"Error backtesting signals: {str(e)}")
        raise


def main():
    try:
        load_environment_variables()
        components = initialize_components()

        symbol = "BTCUSDT"
        raw_data = fetch_data(components['api_manager'], symbol)
        processed_data = process_data(raw_data)

        target_column = 'close'
        feature_columns = [col for col in processed_data.columns if
                           col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        evaluation_results = train_and_evaluate_model(components['model_selector'], components['model_evaluator'],
                                                      processed_data, target_column, feature_columns)

        logger.info("Model Evaluation Results:")
        logger.info(f"Overall Metrics: {evaluation_results['Overall Metrics']}")
        logger.info(f"Trading Performance: {evaluation_results['Trading Performance']}")

        combined_signals = generate_trading_signals(components['signal_generator'], processed_data,
                                                    components['model_selector'], feature_columns)

        backtest_results = backtest_signals(components['signal_generator'], processed_data, combined_signals)
        logger.info("\nBacktest Results:")
        for key, value in backtest_results.items():
            logger.info(f"{key}: {value}")

        latest_data = processed_data.iloc[-1:]
        next_prediction = components['model_selector'].predict(latest_data[feature_columns])
        logger.info(f"\nPredicted price for next period: {next_prediction[0]}")

        current_signal = combined_signals.iloc[-1]
        logger.info(f"Current trading signal: {current_signal}")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")


if __name__ == "__main__":
    main()