# src/models/prophet_model.py

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Optional, Dict, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProphetModel:
    def __init__(self, seasonality_mode: str = 'multiplicative',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True):
        """
        Инициализация модели Prophet.

        :param seasonality_mode: Режим сезонности ('multiplicative' или 'additive').
        :param yearly_seasonality: Учитывать ли годовую сезонность.
        :param weekly_seasonality: Учитывать ли недельную сезонность.
        :param daily_seasonality: Учитывать ли дневную сезонность.
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None

    def fit(self, data: pd.Series):
        """
        Обучение модели Prophet.

        :param data: Временной ряд для обучения модели (индекс - даты, значения - целевая переменная).
        """
        logger.info("Fitting Prophet model")
        try:
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            self.model.fit(df)
            logger.info("Prophet model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise

    def predict(self, future_dates: int) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели Prophet.

        :param future_dates: Количество дат для прогнозирования вперед.
        :return: Массив прогнозов.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info(f"Predicting {future_dates} steps ahead with Prophet model")
        try:
            future = self.model.make_future_dataframe(periods=future_dates)
            forecast = self.model.predict(future)
            return forecast['yhat'].values[-future_dates:]
        except Exception as e:
            logger.error(f"Error predicting with Prophet model: {str(e)}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров модели.

        :return: Словарь с параметрами модели.
        """
        return {
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality
        }

    def set_params(self, **params):
        """
        Установка параметров модели.

        :param params: Словарь с новыми параметрами.
        """
        if 'seasonality_mode' in params:
            self.seasonality_mode = params['seasonality_mode']
        if 'yearly_seasonality' in params:
            self.yearly_seasonality = params['yearly_seasonality']
        if 'weekly_seasonality' in params:
            self.weekly_seasonality = params['weekly_seasonality']
        if 'daily_seasonality' in params:
            self.daily_seasonality = params['daily_seasonality']
        self.model = None  # Сбрасываем модель, так как параметры изменились
        logger.info("Prophet model parameters updated")

    def evaluate(self, data: pd.Series, horizon: str = '30 days', period: str = '180 days',
                 initial: str = '365 days') -> Dict[str, float]:
        """
        Оценка производительности модели с использованием кросс-валидации.

        :param data: Временной ряд для оценки модели.
        :param horizon: Горизонт прогнозирования для кросс-валидации.
        :param period: Период между наборами для кросс-валидации.
        :param initial: Начальный размер обучающего набора.
        :return: Словарь с метриками производительности.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Evaluating Prophet model performance")
        try:
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
            cv_results = cross_validation(self.model, horizon=horizon, period=period, initial=initial)
            metrics = performance_metrics(cv_results)
            return {
                'mae': metrics['mae'].mean(),
                'mape': metrics['mape'].mean(),
                'mse': metrics['mse'].mean(),
                'rmse': metrics['rmse'].mean()
            }
        except Exception as e:
            logger.error(f"Error evaluating Prophet model: {str(e)}")
            raise

    def plot_components(self, figsize: tuple = (10, 8)):
        """
        Построение графиков компонентов модели.

        :param figsize: Размер фигуры для графиков.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Plotting Prophet model components")
        try:
            future = self.model.make_future_dataframe(periods=365)
            forecast = self.model.predict(future)
            self.model.plot_components(forecast, figsize=figsize)
        except Exception as e:
            logger.error(f"Error plotting Prophet model components: {str(e)}")
            raise

    def add_country_holidays(self, country_name: str):
        """
        Добавление праздников страны в модель.

        :param country_name: Название страны (например, 'US' для США).
        """
        if self.model is None:
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )

        logger.info(f"Adding {country_name} holidays to Prophet model")
        try:
            self.model.add_country_holidays(country_name=country_name)
            logger.info(f"{country_name} holidays added successfully")
        except Exception as e:
            logger.error(f"Error adding {country_name} holidays to Prophet model: {str(e)}")
            raise

