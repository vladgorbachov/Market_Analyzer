from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from src.data_collection.api_manager import APIManager
from src.data_processing import data_harmonization
from src.feature_engineering import feature_extractor
from src.models import price_predictor

# Загрузка переменных из .env файла
load_dotenv()

# Теперь вы можете использовать переменные окружения
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")


api_manager = APIManager()

# Получение данных о цене BTC/USDT
price_data = api_manager.get_price_data("BTCUSDT")
print(price_data)

# Получение обзора рынка
market_overview = api_manager.get_market_overview(limit=10)
print(market_overview)


def main():
    load_dotenv()  # Загрузка переменных окружения

    # Инициализация компонентов
    api_manager = APIManager()

    # Получение данных
    price_data = api_manager.get_price_data("BTCUSDT")
    market_overview = api_manager.get_market_overview(limit=10)

    # Обработка данных
    harmonized_data = data_harmonization.harmonize_price_data(price_data)

    # Извлечение признаков
    features = feature_extractor.extract_features(harmonized_data)

    # Прогнозирование
    prediction = price_predictor.predict(features)

    # Вывод результатов
    print(f"Predicted price movement: {prediction}")


if __name__ == "__main__":
    main()