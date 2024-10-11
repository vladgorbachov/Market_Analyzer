import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple

class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, data: pd.Series):
        data.index = pd.DatetimeIndex(data.index).to_period('D').to_timestamp()
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps: int) -> np.ndarray:
        return self.model_fit.forecast(steps)

    def get_params(self) -> dict:
        return {'order': self.order}

    def set_params(self, **params):
        self.order = params.get('order', self.order)