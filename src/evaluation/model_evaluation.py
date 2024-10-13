# src/evaluation/model_evaluation.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from src.models.model_selector import ModelSelector
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model_selector: ModelSelector):
        """
        Инициализация оценщика моделей.

        :param model_selector: Экземпляр класса ModelSelector с обученной моделью.
        """
        self.model_selector = model_selector

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Рассчитывает различные метрики для оценки качества модели.

        :param y_true: Истинные значения целевой переменной.
        :param y_pred: Предсказанные значения.
        :return: Словарь с рассчитанными метриками.
        """
        logger.info("Calculating evaluation metrics")
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Оценивает модель на заданном наборе данных.

        :param X: DataFrame с признаками.
        :param y: Series с истинными значениями целевой переменной.
        :return: Словарь с метриками оценки.
        """
        logger.info("Evaluating model on given dataset")
        try:
            y_pred = self.model_selector.predict(X)
            return self.calculate_metrics(y.values, y_pred)
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def rolling_window_backtest(self, data: pd.DataFrame, target_column: str, feature_columns: List[str],
                                window_size: int, step_size: int) -> pd.DataFrame:
        """
        Проводит бэктестинг модели с использованием метода скользящего окна.

        :param data: DataFrame с данными.
        :param target_column: Название целевой колонки.
        :param feature_columns: Список названий колонок с признаками.
        :param window_size: Размер окна для обучения.
        :param step_size: Шаг смещения окна.
        :return: DataFrame с результатами бэктестинга.
        """
        logger.info(f"Performing rolling window backtest with window size {window_size} and step size {step_size}")
        results = []
        try:
            for i in range(0, len(data) - window_size - step_size + 1, step_size):
                train_data = data.iloc[i:i + window_size]
                test_data = data.iloc[i + window_size:i + window_size + step_size]

                self.model_selector.fit(train_data, target_column, feature_columns)

                X_test = test_data[feature_columns]
                y_test = test_data[target_column]
                y_pred = self.model_selector.predict(X_test)

                metrics = self.calculate_metrics(y_test.values, y_pred)
                metrics['start_date'] = test_data.index[0]
                metrics['end_date'] = test_data.index[-1]
                results.append(metrics)

            return pd.DataFrame(results)
        except Exception as e:
            logger.error(f"Error in rolling window backtest: {str(e)}")
            raise

    def plot_backtest_results(self, backtest_results: pd.DataFrame, metric: str = 'RMSE'):
        """
        Визуализирует результаты бэктестинга.

        :param backtest_results: DataFrame с результатами бэктестинга.
        :param metric: Метрика для визуализации.
        """
        logger.info(f"Plotting backtest results for metric: {metric}")
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(backtest_results['start_date'], backtest_results[metric])
            plt.title(f'{metric} over time')
            plt.xlabel('Date')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting backtest results: {str(e)}")
            raise

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Рассчитывает коэффициент Шарпа для заданных доходностей.

        :param returns: Массив доходностей.
        :param risk_free_rate: Безрисковая ставка доходности.
        :return: Коэффициент Шарпа.
        """
        logger.info("Calculating Sharpe ratio")
        try:
            excess_returns = returns - risk_free_rate
            if excess_returns.std() == 0:
                return 0
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            raise

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Рассчитывает максимальную просадку для заданных кумулятивных доходностей.

        :param cumulative_returns: Массив кумулятивных доходностей.
        :return: Максимальная просадка.
        """
        logger.info("Calculating maximum drawdown")
        try:
            peak = cumulative_returns.max()
            trough = cumulative_returns[cumulative_returns.argmax():].min()
            return (trough - peak) / peak if peak != 0 else 0
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {str(e)}")
            raise

    def evaluate_trading_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     initial_capital: float = 10000.0) -> Dict[str, float]:
        """
        Оценивает торговую производительность модели.

        :param y_true: Истинные значения целевой переменной.
        :param y_pred: Предсказанные значения.
        :param initial_capital: Начальный капитал.
        :return: Словарь с метриками торговой производительности.
        """
        logger.info("Evaluating trading performance")
        try:
            returns = np.diff(y_true) / y_true[:-1]
            predicted_returns = np.diff(y_pred) / y_pred[:-1]

            # Простая стратегия: длинная позиция, если предсказанная доходность положительна
            position = np.where(predicted_returns > 0, 1, -1)
            strategy_returns = position[:-1] * returns[1:]

            cumulative_returns = (1 + strategy_returns).cumprod()
            total_return = cumulative_returns[-1] - 1
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self.calculate_max_drawdown(cumulative_returns)

            return {
                'Total Return': total_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Final Portfolio Value': initial_capital * (1 + total_return)
            }
        except Exception as e:
            logger.error(f"Error evaluating trading performance: {str(e)}")
            raise

    def run_full_evaluation(self, data: pd.DataFrame, target_column: str, feature_columns: List[str],
                            window_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
        """
        Проводит полную оценку модели, включая бэктестинг и оценку торговой производительности.

        :param data: DataFrame с данными.
        :param target_column: Название целевой колонки.
        :param feature_columns: Список названий колонок с признаками.
        :param window_size: Размер окна для бэктестинга.
        :param step_size: Шаг смещения окна для бэктестинга.
        :return: Словарь с результатами оценки.
        """
        logger.info("Running full model evaluation")
        try:
            backtest_results = self.rolling_window_backtest(data, target_column, feature_columns, window_size,
                                                            step_size)

            X = data[feature_columns]
            y = data[target_column]
            y_pred = self.model_selector.predict(X)

            overall_metrics = self.evaluate_model(X, y)
            trading_performance = self.evaluate_trading_performance(y.values, y_pred)

            return {
                'Backtest Results': backtest_results,
                'Overall Metrics': overall_metrics,
                'Trading Performance': trading_performance
            }
        except Exception as e:
            logger.error(f"Error in full evaluation: {str(e)}")
            raise

    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Визуализирует предсказанные значения в сравнении с фактическими.

        :param y_true: Series с истинными значениями целевой переменной.
        :param y_pred: Массив с предсказанными значениями.
        """
        logger.info("Plotting predictions vs actual values")
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true.index, y_true.values, label='Actual')
            plt.plot(y_true.index, y_pred, label='Predicted')
            plt.title('Predicted vs Actual Values')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting predictions vs actual: {str(e)}")
            raise