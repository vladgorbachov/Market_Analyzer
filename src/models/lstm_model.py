# src/models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, units: int = 50, input_shape: Tuple[int, int] = None, epochs: int = 100, batch_size: int = 32):
        """
        Инициализация модели LSTM.

        :param units: Количество нейронов в LSTM слоях.
        :param input_shape: Форма входных данных (временные шаги, признаки).
        :param epochs: Количество эпох для обучения.
        :param batch_size: Размер батча для обучения.
        """
        self.units = units
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()

        if tf is not None:
            try:
                self._build_model()
            except Exception as e:
                logger.error(f"Error building LSTM model: {str(e)}")

    def _build_model(self):
        """
        Построение архитектуры модели LSTM.
        """
        if self.input_shape is None:
            raise ValueError("input_shape must be specified")

        logger.info(f"Building LSTM model with {self.units} units and input shape {self.input_shape}")
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape),
            tf.keras.layers.LSTM(units=self.units),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model built successfully")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучение модели LSTM.

        :param X: DataFrame с признаками для обучения.
        :param y: Series с целевой переменной.
        """
        if self.model is None:
            logger.info("LSTM model not initialized. Initializing model.")
            self.input_shape = (X.shape[1], 1)
            self._build_model()

        logger.info(f"Fitting LSTM model with {self.epochs} epochs and batch size {self.batch_size}")
        try:
            X_scaled = self.scaler.fit_transform(X.values)
            reshaped_X = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))

            self.model.fit(reshaped_X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
            logger.info("LSTM model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели LSTM.

        :param X: Массив с признаками для прогнозирования.
        :return: Массив прогнозов.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Making predictions with LSTM model")
        try:
            X_scaled = self.scaler.transform(X)
            reshaped_X = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
            return self.model.predict(reshaped_X, verbose=0).flatten()
        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {str(e)}")
            raise

    def get_params(self) -> dict:
        """
        Получение параметров модели.

        :return: Словарь с параметрами модели.
        """
        return {
            'units': self.units,
            'input_shape': self.input_shape,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        """
        Установка параметров модели.

        :param params: Словарь с новыми параметрами.
        """
        if 'units' in params:
            self.units = params['units']
        if 'input_shape' in params:
            self.input_shape = params['input_shape']
        if 'epochs' in params:
            self.epochs = params['epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']

        # Пересоздаем модель с новыми параметрами
        self._build_model()
        logger.info("LSTM model parameters updated")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Оценка производительности модели на тестовых данных.

        :param X: Массив с признаками для оценки.
        :param y: Массив с истинными значениями.
        :return: Словарь с метриками производительности.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Evaluating LSTM model performance")
        try:
            X_scaled = self.scaler.transform(X)
            reshaped_X = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
            loss = self.model.evaluate(reshaped_X, y, verbose=0)
            return {'loss': loss}
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """
        Сохранение модели в файл.

        :param filepath: Путь для сохранения модели.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Cannot save.")

        logger.info(f"Saving LSTM model to {filepath}")
        try:
            self.model.save(filepath)
            logger.info("LSTM model saved successfully")
        except Exception as e:
            logger.error(f"Error saving LSTM model: {str(e)}")
            raise

    def load_model(self, filepath: str):
        """
        Загрузка модели из файла.

        :param filepath: Путь к файлу модели.
        """
        logger.info(f"Loading LSTM model from {filepath}")
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            raise

