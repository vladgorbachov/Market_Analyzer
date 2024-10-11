import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)