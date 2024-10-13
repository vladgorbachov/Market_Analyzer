# src/models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from typing import Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, units: int = 50, input_shape: Optional[Tuple[int, int]] = None, epochs: int = 100, batch_size: int = 32):
        self.units = units
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def _build_model(self):
        if self.input_shape is None:
            raise ValueError("input_shape must be specified")

        logger.info(f"Building LSTM model with {self.units} units and input shape {self.input_shape}")
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(units=self.units, return_sequences=True, input_shape=self.input_shape),
                tf.keras.layers.LSTM(units=self.units),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            logger.info("LSTM model built successfully")
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.model is None:
            logger.info("LSTM model not initialized. Initializing model.")
            self._preprocess_data(X)
            self._build_model()

        logger.info(f"Fitting LSTM model with {self.epochs} epochs and batch size {self.batch_size}")
        try:
            X_processed = self._preprocess_data(X)
            self.model.fit(X_processed, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
            logger.info("LSTM model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Making predictions with LSTM model")
        try:
            X_processed = self._preprocess_data(X)
            return self.model.predict(X_processed, verbose=0).flatten()
        except Exception as e:
            logger.error(f"Error predicting with LSTM model: {str(e)}")
            raise

    def _preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        X_numeric = self.scaler.fit_transform(X[numeric_features])
        X_categorical = self.encoder.fit_transform(X[categorical_features])

        X_processed = np.hstack((X_numeric, X_categorical))
        return X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))

    def get_params(self) -> dict:
        return {
            'units': self.units,
            'input_shape': self.input_shape,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        if 'units' in params:
            self.units = params['units']
        if 'input_shape' in params:
            self.input_shape = params['input_shape']
        if 'epochs' in params:
            self.epochs = params['epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']

        if self.input_shape is not None:
            self._build_model()
            logger.info("LSTM model parameters updated")
        else:
            logger.warning("Cannot build model: input_shape is not set")

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info("Evaluating LSTM model performance")
        try:
            X_processed = self._preprocess_data(X)
            loss = self.model.evaluate(X_processed, y, verbose=0)
            return {'loss': loss}
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {str(e)}")
            raise

    def save_model(self, filepath: str):
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
        logger.info(f"Loading LSTM model from {filepath}")
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            raise