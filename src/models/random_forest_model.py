# src/models/random_forest_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Optional
import logging
from joblib import dump, load

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Инициализация модели Random Forest.

        :param n_estimators: Количество деревьев в лесу.
        :param max_depth: Максимальная глубина деревьев.
        :param min_samples_split: Минимальное количество образцов для разделения внутреннего узла.
        :param min_samples_leaf: Минимальное количество образцов в конечных узлах.
        :param random_state: Seed для генератора случайных чисел.
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.feature_names: Optional[List[str]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирование категориальных переменных.

        :param X: DataFrame с признаками.
        :return: DataFrame с закодированными признаками.
        """
        X_encoded = X.copy()
        for column in X.select_dtypes(include=['object', 'category']).columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                X_encoded[column] = self.label_encoders[column].fit_transform(X[column])
            else:
                X_encoded[column] = self.label_encoders[column].transform(X[column])
        return X_encoded

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучение модели Random Forest.

        :param X: DataFrame с признаками для обучения.
        :param y: Series с целевой переменной.
        """
        logger.info("Fitting Random Forest model")
        try:
            self.feature_names = X.columns.tolist()
            X_encoded = self._encode_categorical(X)
            self.model.fit(X_encoded, y)
            logger.info("Random Forest model fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting Random Forest model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Прогнозирование с использованием обученной модели Random Forest.

        :param X: DataFrame с признаками для прогнозирования.
        :return: Массив прогнозов.
        """
        logger.info("Making predictions with Random Forest model")
        try:
            X_encoded = self._encode_categorical(X)
            return self.model.predict(X_encoded)
        except Exception as e:
            logger.error(f"Error predicting with Random Forest model: {str(e)}")
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
        logger.info("Setting new parameters for Random Forest model")
        try:
            self.model.set_params(**params)
            logger.info("Parameters updated successfully")
        except Exception as e:
            logger.error(f"Error setting parameters for Random Forest model: {str(e)}")
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
        logger.info("Performing grid search for Random Forest model")
        try:
            X_encoded = self._encode_categorical(X)
            grid_search = GridSearchCV(self.model, param_grid, cv=cv, n_jobs=-1, verbose=1)
            grid_search.fit(X_encoded, y)
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
        logger.info(f"Saving Random Forest model to {filepath}")
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders
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
        logger.info(f"Loading Random Forest model from {filepath}")
        try:
            model_data = load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.label_encoders = model_data['label_encoders']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def partial_dependence_plot(self, X: pd.DataFrame, features: List[str], grid_resolution: int = 100):
        """
        Построение графика частичной зависимости.

        :param X: DataFrame с признаками.
        :param features: Список признаков для анализа.
        :param grid_resolution: Разрешение сетки для построения графика.
        """
        from sklearn.inspection import partial_dependence
        import matplotlib.pyplot as plt

        logger.info(f"Generating partial dependence plot for features: {features}")
        try:
            X_encoded = self._encode_categorical(X)
            pdp = partial_dependence(self.model, X_encoded, features, grid_resolution=grid_resolution)

            fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
            if len(features) == 1:
                axes = [axes]

            for i, (ax, feature) in enumerate(zip(axes, features)):
                ax.plot(pdp['values'][i], pdp['average'][i])
                ax.set_xlabel(feature)
                ax.set_ylabel('Partial dependence')
                ax.set_title(f'Partial Dependence Plot for {feature}')

            plt.tight_layout()
            plt.show()
            logger.info("Partial dependence plot generated successfully")
        except Exception as e:
            logger.error(f"Error generating partial dependence plot: {str(e)}")
            raise