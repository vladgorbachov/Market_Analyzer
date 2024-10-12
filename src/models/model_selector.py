# src/models/model_selector.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple
import logging
import warnings

from .arima_model import ARIMAModel
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        except Exception as e:
            logger.warning(f"Failed to initialize LSTM model: {str(e)}")

        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, data: pd.DataFrame, target_column: str, feature_columns: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info("Preparing data for model training and testing")
        try:
            X = data[feature_columns]
            y = data[target_column]
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        logger.info("Training and evaluating models")
        results = {}
        for name, model in self.models.items():
            try:
                logger.info(f"Training and evaluating {name} model")
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
                logger.info(f"{name} model evaluation results: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            except Exception as e:
                logger.error(f"Error in training and evaluating {name} model: {str(e)}")
                results[name] = {'MSE': float('inf'), 'MAE': float('inf'), 'R2': float('-inf')}

        return results

    def select_best_model(self, results: Dict[str, Dict[str, float]]) -> str:
        logger.info("Selecting the best model")
        best_score = float('inf')
        best_model_name = ''
        for name, metrics in results.items():
            if metrics['MSE'] < best_score:
                best_score = metrics['MSE']
                best_model_name = name

        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model selected: {best_model_name}")
        return best_model_name

    def tune_best_model(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, Any]):
        if self.best_model_name in ['ARIMA', 'Prophet', 'LSTM']:
            warnings.warn("Automatic tuning is not implemented for {self.best_model_name}. Please tune manually.", UserWarning)
            return

        logger.info(f"Tuning hyperparameters for {self.best_model_name}")
        try:
            grid_search = GridSearchCV(self.best_model.model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)

            self.best_model.set_params(**grid_search.best_params_)
            logger.info(f"Best parameters for {self.best_model_name}: {grid_search.best_params_}")
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")

    def fit(self, data: pd.DataFrame, target_column: str, feature_columns: List[str]):
        logger.info("Starting model fitting process")
        try:
            X_train, X_test, y_train, y_test = self.prepare_data(data, target_column, feature_columns)
            results = self.train_and_evaluate(X_train, X_test, y_train, y_test)
            best_model = self.select_best_model(results)
            logger.info(f"Best model: {best_model}")
            logger.info("Results:")
            for name, metrics in results.items():
                logger.info(f"{name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")
        except Exception as e:
            logger.error(f"Error in model fitting process: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        logger.info(f"Making predictions using {self.best_model_name}")
        try:
            return self.best_model.predict(X)
        except Exception as e:
            logger.error(f"Error in making predictions: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        if self.best_model_name in ['XGBoost', 'RandomForest']:
            try:
                importance = self.best_model.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.best_model.model.feature_names_in_,
                    'importance': importance
                })
                return feature_importance.sort_values('importance', ascending=False)
            except Exception as e:
                logger.error(f"Error in getting feature importance: {str(e)}")
                return pd.DataFrame()
        else:
            logger.warning(f"Feature importance is not available for {self.best_model_name}")
            return pd.DataFrame()

