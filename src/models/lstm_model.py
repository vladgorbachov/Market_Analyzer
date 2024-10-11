import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class LSTMModel:
    def __init__(self, units: int = 50, input_shape: tuple = None):
        self.units = units
        self.input_shape = input_shape
        self.model = None
        self.scaler = MinMaxScaler()

        if tf is not None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(units=units, return_sequences=True, input_shape=input_shape),
                tf.keras.layers.LSTM(units=units),
                tf.keras.layers.Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 100, batch_size: int = 32):
        if self.model is None:
            print("LSTM модель не инициализирована. Пропуск обучения.")
            return
        X_scaled = self.scaler.fit_transform(X.values)
        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1)
        reshaped_X = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
        self.model.fit(reshaped_X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            print("LSTM модель не инициализирована. Возврат случайных предсказаний.")
            return np.random.rand(X.shape[0])
        X_scaled = self.scaler.transform(X)
        reshaped_X = X_scaled.reshape((X_scaled.shape[0], self.input_shape[0], self.input_shape[1]))
        return self.model.predict(reshaped_X).flatten()

    def get_params(self) -> dict:
        return {'units': self.units, 'input_shape': self.input_shape}

    def set_params(self, **params):
        self.__init__(**params)