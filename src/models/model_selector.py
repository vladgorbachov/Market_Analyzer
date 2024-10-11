import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

from .arima_model import ARIMAModel
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel

class ModelSelector:
    def __init__(self):
        self.models = {
            'ARIMA': ARIMAModel(),
            'Prophet': ProphetModel(),
            'XGBoost': XGBoostModel(),
            'RandomForest': RandomForestModel()
        }
        try:
            lstm_model = LSTMModel()
            if lstm_model.model is not None:
                self.models['LSTM'] = lstm_model
        except:
            print("Не удалось инициализировать LSTM модель")

    def prepare_data(self, data: pd.DataFrame, target_column: str, feature_columns: list) -> tuple:
        X = data[feature_columns]
        y = data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, model in self.models.items():
            if name in ['ARIMA', 'Prophet']:
                model.fit(y_train)
                y_pred = model.predict(len(y_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

        return results

    def select_best_model(self, results: Dict[str, Dict[str, float]]) -> str:
        best_score = float('inf')
        best_model_name = ''
        for name, metrics in results.items():
            if metrics['MSE'] < best_score:
                best_score = metrics['MSE']
                best_model_name = name

        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        return best_model_name

    def tune_best_model(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, Any]):
        if self.best_model_name in ['ARIMA', 'Prophet', 'LSTM']:
            print(f"Automatic tuning is not implemented for {self.best_model_name}. Please tune manually.")
            return

        grid_search = GridSearchCV(self.best_model.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        self.best_model.set_params(**grid_search.best_params_)
        print(f"Best parameters for {self.best_model_name}: {grid_search.best_params_}")

    def fit(self, data: pd.DataFrame, target_column: str, feature_columns: list):
        X_train, X_test, y_train, y_test = self.prepare_data(data, target_column, feature_columns)
        results = self.train_and_evaluate(X_train, X_test, y_train, y_test)
        best_model = self.select_best_model(results)
        print(f"Best model: {best_model}")
        print("Results:")
        for name, metrics in results.items():
            print(f"{name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.best_model.predict(X)