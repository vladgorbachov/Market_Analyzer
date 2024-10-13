# src/models/arima_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None

    def fit(self, data: pd.Series):
        logger.info(f"Fitting ARIMA model with order {self.order}")
        try:
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    logger.warning("Multiple columns detected. Using the first column for ARIMA model.")
                data = data.iloc[:, 0]

            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            if data.index.freq is None:
                data = data.asfreq('D')

            # Проверка на достаточное количество наблюдений
            if len(data) <= max(self.order):
                raise ValueError("Not enough observations for the specified ARIMA order.")

            self.model = ARIMA(data, order=self.order)
            self.model_fit = self.model.fit()
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise

    def predict(self, steps: int) -> pd.Series:
        if self.model_fit is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        logger.info(f"Predicting {steps} steps ahead with ARIMA model")
        try:
            forecast = self.model_fit.forecast(steps)
            return forecast
        except Exception as e:
            logger.error(f"Error predicting with ARIMA model: {str(e)}")
            raise

    def get_params(self) -> dict:
        return {'order': self.order}

    def set_params(self, **params):
        if 'order' in params:
            self.order = params['order']
            logger.info(f"ARIMA model order updated to {self.order}")

    def evaluate(self, test_data: pd.Series) -> dict:
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
        if self.model_fit is None:
            logger.warning("Model has not been fitted. No summary available.")
            return None

        return str(self.model_fit.summary())

    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 12)):
        if self.model_fit is None:
            logger.warning("Model has not been fitted. Cannot plot diagnostics.")
            return

        try:
            self.model_fit.plot_diagnostics(figsize=figsize)
            plt.show()
            logger.info("Diagnostic plots generated")
        except Exception as e:
            logger.error(f"Error generating diagnostic plots: {str(e)}")

    def plot_forecast(self, steps: int, actual_data: Optional[pd.Series] = None, figsize: Tuple[int, int] = (12, 6)):
        if self.model_fit is None:
            logger.warning("Model has not been fitted. Cannot plot forecast.")
            return

        try:
            forecast = self.predict(steps)
            plt.figure(figsize=figsize)
            plt.plot(forecast.index, forecast.values, label='Forecast')
            if actual_data is not None:
                plt.plot(actual_data.index, actual_data.values, label='Actual')
            plt.title('ARIMA Forecast')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
            logger.info("Forecast plot generated")
        except Exception as e:
            logger.error(f"Error generating forecast plot: {str(e)}")