import numpy as np
import pandas as pd
from prophet import Prophet

class ProphetModel:
    def __init__(self, seasonality_mode: str = 'multiplicative'):
        self.seasonality_mode = seasonality_mode
        self.model = None

    def fit(self, data: pd.Series):
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        self.model = Prophet(seasonality_mode=self.seasonality_mode)
        self.model.fit(df)

    def predict(self, future_dates: int) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        future = self.model.make_future_dataframe(periods=future_dates)
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-future_dates:]

    def get_params(self) -> dict:
        return {'seasonality_mode': self.seasonality_mode}

    def set_params(self, **params):
        self.seasonality_mode = params.get('seasonality_mode', self.seasonality_mode)
        self.model = None