import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class XGBoostModel:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self.model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)