# src/models/arima_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error
from math import sqrt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Инициализация модели ARIMA.

        :param order: Кортеж (p, d, q) для модели ARIMA.
        """
        self.order = order
        self.model = None
        self.model_fit = None

    def fit(self, data: pd.Series):
        """
        Обучение модели ARIMA.

        :param data: Временной ряд для обучения модели.
        """
        logger.info(f"Fitting ARIMA model with order {self.order}")
        try:
            # Преобразование индекса в DatetimeIndex, если это еще не сделано
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Установка частоты временного ряда, если она не задана
            if data.index.freq is None:
                data = data.asfreq('D')  # Предполагаем дневную частоту, измените при необходимости

            self.model = ARIMA(data, order=self.order)
            self.model_fit = self.model.fit()
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise

    def predict(self, steps: int) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели ARIMA.

        :param steps: Количество шагов для прогнозирования.
        :return: Массив прогнозов.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info(f"Predicting {steps} steps ahead with ARIMA model")
        try:
            forecast = self.model_fit.forecast(steps)
            return forecast.values
        except Exception as e:
            logger.error(f"Error predicting with ARIMA model: {str(e)}")
            raise

    def get_params(self) -> dict:
        """
        Получение параметров модели.

        :return: Словарь с параметрами модели.
        """
        return {'order': self.order}

    def set_params(self, **params):
        """
        Установка параметров модели.

        :param params: Словарь с новыми параметрами.
        """
        if 'order' in params:
            self.order = params['order']
            logger.info(f"ARIMA model order updated to {self.order}")

    def evaluate(self, test_data: pd.Series) -> dict:
        """
        Оценка производительности модели на тестовых данных.

        :param test_data: Временной ряд для оценки модели.
        :return: Словарь с метриками производительности.
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Evaluating ARIMA model performance")
        try:
            predictions = self.predict(len(test_data))
            mse = mean_squared_error(test_data, predictions)
            rmse = sqrt(mse)
            return {
                'MSE': mse,
                'RMSE': rmse
            }
        except Exception as e:
            logger.error(f"Error evaluating ARIMA model: {str(e)}")
            raise

    def summary(self) -> Optional[str]:
        """
        Получение сводки модели.

        :return: Строка с сводкой модели или None, если модель не обучена.
        """
        if self.model_fit is None:
            logger.warning("Model has not been fitted. No summary available.")
            return None

        return str(self.model_fit.summary())

    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 12)):
        """
        Построение диагностических графиков модели.

        :param figsize: Размер фигуры для графиков.
        """
        if self.model_fit is None:
            logger.warning("Model has not been fitted. Cannot plot diagnostics.")
            return

        try:
            self.model_fit.plot_diagnostics(figsize=figsize)
            logger.info("Diagnostic plots generated")
        except Exception as e:
            logger.error(f"Error generating diagnostic plots: {str(e)}")

