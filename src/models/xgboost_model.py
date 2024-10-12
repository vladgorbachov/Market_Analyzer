# src/models/xgboost_model.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, List, Optional
import logging
import matplotlib.pyplot as plt
from joblib import dump, load

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, random_state: int = 42):
        """
        Инициализация модели XGBoost.

        :param n_estimators: Количество деревьев.
        :param learning_rate: Скорость обучения.
        :param max_depth: Максимальная глубина деревьев.
        :param subsample: Доля образцов, используемых для обучения каждого дерева.
        :param colsample_bytree: Доля признаков, используемых для обучения каждого дерева.
        :param random_state: Seed для генератора случайных чисел.
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучение модели XGBoost.

        :param X: DataFrame с признаками для обучения.
        :param y: Series с целевой переменной.
        """
        logger.info("Fitting XGBoost model")
        try:
            self.feature_names = X.columns.tolist()
            self.model.fit(X, y)
            logger.info("XGBoost model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели XGBoost.

        :param X: DataFrame с признаками для прогнозирования.
        :return: Массив прогнозов.
        """
        logger.info("Making predictions with XGBoost model")
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with XGBoost model: {str(e)}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров модели.

        :return: Словарь с параметрами модели.
        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        Установка параметров модели.

        :param params: Словарь с новыми параметрами.
        """
        logger.info("Setting new parameters for XGBoost model")
        try:
            self.model.set_params(**params)
            logger.info("Parameters updated successfully")
        except Exception as e:
            logger.error(f"Error setting parameters for XGBoost model: {str(e)}")
            raise

    def feature_importances(self) -> pd.DataFrame:
        """
        Получение важности признаков.

        :return: DataFrame с важностью признаков.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted with feature names. Call fit() first.")

        logger.info("Calculating feature importances")
        try:
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
            return feature_importance_df.sort_values('importance', ascending=False)
        except Exception as e:
            logger.error(f"Error calculating feature importances: {str(e)}")
            raise

    def grid_search(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, Any], cv: int = 5):
        """
        Выполнение поиска по сетке для оптимизации гиперпараметров.

        :param X: DataFrame с признаками.
        :param y: Series с целевой переменной.
        :param param_grid: Словарь с параметрами для поиска.
        :param cv: Количество фолдов для кросс-валидации.
        """
        logger.info("Performing grid search for XGBoost model")
        try:
            grid_search = GridSearchCV(self.model, param_grid, cv=cv, n_jobs=-1, verbose=1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(f"Best score: {grid_search.best_score_}")
        except Exception as e:
            logger.error(f"Error during grid search: {str(e)}")
            raise

    def save_model(self, filepath: str):
        """
        Сохранение модели в файл.

        :param filepath: Путь для сохранения модели.
        """
        logger.info(f"Saving XGBoost model to {filepath}")
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            dump(model_data, filepath)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: str):
        """
        Загрузка модели из файла.

        :param filepath: Путь к файлу модели.
        """
        logger.info(f"Loading XGBoost model from {filepath}")
        try:
            model_data = load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def plot_feature_importance(self, top_n: int = 10):
        """
        Построение графика важности признаков.

        :param top_n: Количество наиболее важных признаков для отображения.
        """
        logger.info(f"Plotting top {top_n} feature importances")
        try:
            feature_importance = self.feature_importances()
            top_features = feature_importance.head(top_n)

            plt.figure(figsize=(10, 6))
            plt.bar(top_features['feature'], top_features['importance'])
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            logger.info("Feature importance plot generated successfully")
        except Exception as e:
            logger.error(f"Error plotting feature importances: {str(e)}")
            raise

    def plot_learning_curve(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                            train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5)):
        """
        Построение кривой обучения.

        :param X: DataFrame с признаками.
        :param y: Series с целевой переменной.
        :param cv: Количество фолдов для кросс-валидации.
        :param train_sizes: Массив с размерами обучающей выборки.
        """
        from sklearn.model_selection import learning_curve

        logger.info("Generating learning curve")
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.title("Learning Curve")
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

            plt.legend(loc="best")
            plt.show()
            logger.info("Learning curve generated successfully")
        except Exception as e:
            logger.error(f"Error generating learning curve: {str(e)}")
            raise

