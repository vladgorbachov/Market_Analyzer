import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from src.models.model_selector import ModelSelector


class ModelEvaluator:
    def __init__(self, model_selector: ModelSelector):
        self.model_selector = model_selector

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Рассчитывает различные метрики для оценки качества модели.
        """
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Оценивает модель на заданном наборе данных.
        """
        y_pred = self.model_selector.predict(X)
        return self.calculate_metrics(y, y_pred)

    def rolling_window_backtest(self, data: pd.DataFrame, target_column: str, feature_columns: List[str],
                                window_size: int, step_size: int) -> pd.DataFrame:
        """
        Проводит бэктестинг модели с использованием метода скользящего окна.
        """
        results = []
        for i in range(0, len(data) - window_size, step_size):
            train_data = data.iloc[i:i + window_size]
            test_data = data.iloc[i + window_size:i + window_size + step_size]

            self.model_selector.fit(train_data, target_column, feature_columns)

            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            y_pred = self.model_selector.predict(X_test)

            metrics = self.calculate_metrics(y_test, y_pred)
            metrics['start_date'] = test_data.index[0]
            metrics['end_date'] = test_data.index[-1]
            results.append(metrics)

        return pd.DataFrame(results)

    def plot_backtest_results(self, backtest_results: pd.DataFrame, metric: str = 'RMSE'):
        """
        Визуализирует результаты бэктестинга.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['start_date'], backtest_results[metric])
        plt.title(f'{metric} over time')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Рассчитывает коэффициент Шарпа для заданных доходностей.
        """
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Рассчитывает максимальную просадку для заданных кумулятивных доходностей.
        """
        peak = cumulative_returns.max()
        trough = cumulative_returns[cumulative_returns.argmax():].min()
        return (trough - peak) / peak

    def evaluate_trading_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     initial_capital: float = 10000.0) -> Dict[str, float]:
        """
        Оценивает торговую производительность модели.
        """
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

    def run_full_evaluation(self, data: pd.DataFrame, target_column: str, feature_columns: List[str],
                            window_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
        """
        Проводит полную оценку модели, включая бэктестинг и оценку торговой производительности.
        """
        backtest_results = self.rolling_window_backtest(data, target_column, feature_columns, window_size, step_size)

        # Оценка на всем наборе данных
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

    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Визуализирует предсказанные значения в сравнении с фактическими.
        """
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