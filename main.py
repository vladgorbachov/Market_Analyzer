import os
from dotenv import load_dotenv
from src.data_collection.api_manager import APIManager
from src.data_processing.data_harmonization import harmonize_price_data, harmonize_historical_data
from src.feature_engineering.technical_indicators import add_all_indicators
from src.feature_engineering.text_features import add_text_features
from src.models.model_selector import ModelSelector
from src.evaluation.model_evaluation import ModelEvaluator
from src.trading.signal_generator import SignalGenerator
import warnings
import time
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google._upb._message")

def main():
    # Загрузка переменных окружения
    load_dotenv()

    # Инициализация компонентов
    api_manager = APIManager()
    model_selector = ModelSelector()
    model_evaluator = ModelEvaluator(model_selector)
    signal_generator = SignalGenerator()

    # Получение данных
    symbol = "BTCUSDT"
    price_data = api_manager.get_price_data(symbol)
    start_time = int(time.time() * 1000) - (1000 * 60 * 60 * 24)  # Например, 24 часа назад
    end_time = int(time.time() * 1000)
    historical_data = api_manager.get_historical_data(symbol, interval="1h", start_time=start_time, end_time=end_time)
    market_overview = api_manager.get_market_overview(limit=10)

    # Преобразование market_overview в DataFrame
    market_overview_df = pd.DataFrame(market_overview.get('data', []))

    # Обработка данных
    harmonized_price_data = harmonize_price_data(price_data)
    harmonized_historical_data = harmonize_historical_data(historical_data)

    # Извлечение признаков
    data_with_indicators = add_all_indicators(harmonized_historical_data)

    # Добавление текстовых признаков (если есть текстовые данные)
    if 'description' in market_overview_df.columns:
        data_with_features = add_text_features(data_with_indicators, 'description')
    else:
        data_with_features = data_with_indicators

    # Подготовка данных для модели
    target_column = 'close'
    feature_columns = [col for col in data_with_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Обучение модели
    model_selector.fit(data_with_features, target_column, feature_columns)

    # Оценка модели
    evaluation_results = model_evaluator.run_full_evaluation(data_with_features, target_column, feature_columns)
    print("Model Evaluation Results:")
    print(f"Overall Metrics: {evaluation_results['Overall Metrics']}")
    print(f"Trading Performance: {evaluation_results['Trading Performance']}")

    # Генерация торговых сигналов
    # Добавляем стратегии в генератор сигналов
    signal_generator.add_strategy('trend_following', SignalGenerator.trend_following_strategy)
    signal_generator.add_strategy('mean_reversion', SignalGenerator.mean_reversion_strategy)
    signal_generator.add_strategy('rsi', SignalGenerator.rsi_strategy)

    # Генерация сигналов для каждой стратегии
    trend_following_signals = signal_generator.generate_signals(data_with_features, 'trend_following')
    mean_reversion_signals = signal_generator.generate_signals(data_with_features, 'mean_reversion')
    rsi_signals = signal_generator.generate_signals(data_with_features, 'rsi')

    # Сгенерированные модели предсказания
    predictions = model_selector.predict(data_with_features[feature_columns])
    ml_model_signals = SignalGenerator.ml_model_strategy(data_with_features, predictions)

    # Объединение сигналов
    combined_signals = signal_generator.combine_signals(
        [trend_following_signals, mean_reversion_signals, rsi_signals, ml_model_signals],
        method='majority'
    )

    # Бэктестинг сигналов
    backtest_results = signal_generator.backtest_signals(data_with_features, combined_signals)
    print("\nBacktest Results:")
    for key, value in backtest_results.items():
        print(f"{key}: {value}")

    # Прогноз на следующий период
    latest_data = data_with_features.iloc[-1:]
    next_prediction = model_selector.predict(latest_data[feature_columns])
    print(f"\nPredicted price for next period: {next_prediction[0]}")

    # Генерация сигнала для текущего момента
    current_signal = combined_signals.iloc[-1]
    print(f"Current trading signal: {current_signal}")

if __name__ == "__main__":
    main()
