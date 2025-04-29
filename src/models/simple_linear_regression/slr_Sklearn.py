# src/models/simple_linear_regression/slr_Sklearn.py
"""
Implements Simple Linear Regression using Scikit-learn's interface.

Supports:
- normal_equation
- batch, stochastic, mini-batch gradient descent

Designed to mirror the from-scratch interface for comparison
"""

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor

from src.core.metrics.metrics import (
    calculate_adjusted_r_squated,
    calculate_mae,
    calculate_median_ae,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from src.core.registry import register_model
from src.models.ground_up_ml_base_model import GroundUpMLBaseModel
from src.utils.utils import format_duration


@register_model("sklearn", task_type="regression", group="simple_linear")
class SimpleLinearRegressionSklearn(GroundUpMLBaseModel):
    """
    A wrapper around sklearn's SGDRegressor that mimics the interface of
    SimpleLinearRegressionFromScratch for consistent visualization and benchmarking.

    The interface of SimpleLinearRegressionSklearn matches
    SimpleLinearRegressionFromScratch and SimpleLinearRegressionPyTorch.
    """

    ALL_METHODS = [
        "normal_equation",
        "gradient_descent_batch",
        "gradient_descent_stochastic",
        "gradient_descent_mini_batch",
    ]

    def __init__(self, x: pd.Series, y: pd.Series):
        """
        Initialize the model with feature and target data.

        Args:
            x (pd.Series): Feature values.
            y (pd.Series): Target values.
        """
        self.x: np.ndarray = x.values.reshape(-1, 1)
        self.y: np.ndarray = y.to_numpy()

        # Method to calculate coefficient estimations
        self.method: str = None

        self.model: Optional[Union[LinearRegression, SGDRegressor]] = None

        # Coefficient Estimations
        self.beta_0_hat: Optional[float] = None
        self.beta_1_hat: Optional[float] = None

        # Performance metrics
        self.mse: Optional[float] = None
        self.rmse: Optional[float] = None
        self.mae: Optional[float] = None
        self.median_ae: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.adjusted_r_squared: Optional[float] = None

        self.duration_seconds: float = 0.0

        self.diagnostics: dict = {}

    @property
    def name(self):
        return "SimpleLinearRegression-Sklearn"

    def fit(self, X, y, method: str = None) -> None:
        """
        Routing function to call specific coefficient estimator methods

        Args:
            method (str, optional): Method to calculate coefficients.
                Defaults to None.

        Raises:
            ValueError: Raise error if value in 'method' is not a known one.
        """
        self.method = method

        fit_methods = {
            "normal_equation": self._fit_normal_equation,
            "gradient_descent_batch": self._fit_gradient_descent_batch,
            "gradient_descent_stochastic": self._fit_gradient_descent_stochastic,
            "gradient_descent_mini_batch": self._fit_gradient_descent_mini_batch,
        }

        # Raise error if value in 'method' is not a known one
        if method not in fit_methods:
            raise ValueError(
                f"Unknown method '{method}' for SimpleLinearRegressionSklearn."
            )

        # Call respective coefficent estimator
        fit_methods[method]()

    def _fit_normal_equation(self) -> None:
        """
        Fits the model using the Normal Equation
        """
        self.model = LinearRegression().fit(self.x, self.y)

        self.beta_1_hat = self.model.coef_[0]
        self.beta_0_hat = self.model.intercept_

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(
        self,
        n_epochs: int = 50,
    ) -> None:
        """
        Fits the model using batch gradient descent.
        sklearn.SGDRegressor

        Parameters are updated after each epoch.

        Args:
            n_epochs (int, optional): Number of iterations over the data.
                Defaults to 50.

        """
        cost_history = []

        self.model = SGDRegressor(
            loss="squared_error",
            learning_rate="invscaling",
            shuffle=False,
            random_state=42,
            tol=None,
            warm_start=True,
        )

        for epoch in range(n_epochs):
            self.model.partial_fit(self.x, self.y)

            y_pred = self.model.predict(self.x)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        self.beta_1_hat = self.model.coef_[0]
        self.beta_0_hat = self.model.intercept_[0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_stochastic(
        self,
        n_epochs: int = 50,
    ) -> None:
        """
        Fits the model using stochastic gradient descent.
        sklearn.SGDRegressor

        Parameters are updated after each training example, using one
        randomly shuffled sample per step.

        Args:
            n_epochs (int, optional): Number of iterations over the data.
                Defaults to 50.

        """

        cost_history = []

        self.model = SGDRegressor(
            loss="squared_error",
            learning_rate="invscaling",
            shuffle=False,
            random_state=42,
            tol=None,
            warm_start=True,
        )

        for epoch in range(n_epochs):
            indices = np.random.permutation(len(self.x))
            for i in indices:
                x_i = np.reshape(self.x[i], (1, -1))
                y_i = [self.y[i]]
                self.model.partial_fit(x_i, y_i)

            y_pred = self.model.predict(self.x)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        self.beta_1_hat = self.model.coef_[0]
        self.beta_0_hat = self.model.intercept_[0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_mini_batch(
        self, n_epochs: int = 50, batch_size: int = 32
    ) -> None:
        """
        Fits the model using mini-batch gradient descent.
        sklearn.SGDRegressor

        Parameters are updated after each training batch, using a
        randomly shuffled batch per step.

        Args:
            n_epochs (int, optional): Number of iterations over the data.
                Defaults to 50.

        """

        cost_history = []

        self.model = SGDRegressor(
            loss="squared_error",
            learning_rate="invscaling",
            shuffle=False,
            random_state=42,
            tol=None,
            warm_start=True,
        )

        for epoch in range(n_epochs):
            indices = np.random.permutation(len(self.x))
            for i in range(0, len(self.x), batch_size):
                batch_idx = indices[i : i + batch_size]
                x_batch = self.x[batch_idx]
                y_batch = self.y[batch_idx]
                self.model.partial_fit(x_batch, y_batch)

            y_pred = self.model.predict(self.x)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        self.beta_1_hat = self.model.coef_[0]
        self.beta_0_hat = self.model.intercept_[0]

        self.diagnostics = {"cost_history": cost_history}

    def predict(
        self, x_new: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Predict target values for new data.

        Args:
            x_new (Optional): Input features to predict on.

        Returns:
            np.ndarray: Predicted target values.
        """
        if x_new is None:
            x_new = self.x
        return self.model.predict(x_new)

    def residuals(self) -> np.ndarray:
        """

        Returns:
            np.ndarray: Difference between actual 'y' value and predicted 'y' value.
        """
        return self.y - self.predict()

    def fitted(self) -> np.ndarray:
        """
        Return the predicted y values for the training data.

        Returns:
            np.ndarray: Predicted y values for the original training data.
        """
        return self.predict()

    def calculate_metrics(self) -> None:
        y_pred = self.predict()
        self.mse = calculate_mse(self.y, y_pred)
        self.rmse = calculate_rmse(self.y, y_pred)
        self.mae = calculate_mae(self.y, y_pred)
        self.median_ae = calculate_median_ae(self.y, y_pred)
        self.r_squared = calculate_r_squared(self.y, y_pred)
        self.adjusted_r_squared = calculate_adjusted_r_squated(self.y, y_pred)

    def _coefficient_estimators(
        self, x: pd.Series, y: pd.Series, n: int, methods: str = None
    ) -> list:
        """
        Run each coefficient estimation method on the current dataset.

        Returns:
            list: A list of dictionaries containing model diagnostics per method.
        """

        coeff_results = []
        methods = methods or SimpleLinearRegressionSklearn.ALL_METHODS

        for method in methods:
            model = SimpleLinearRegressionSklearn(x, y)
            try:
                start_time = time.perf_counter()
                model.fit(method=method)
                duration = time.perf_counter() - start_time
                formatted_time = format_duration(duration)

                """
                # Performance metrics
                # RMSE: Root Mean Squared Error - Square root of the average of the
                #   squared differences between the predicted values and the actual
                # MAE: Mean Absolute Error - Average absolute difference between the
                #   predicted values and the actual
                # R_squared:
                """
                model.calculate_metrics()

                # MLflow logging
                # with start_run(run_name=f"{method}_{n}_samples"):
                #     log_params({"method": method, "n_samples": n})
                #     log_metrics({
                # "rmse": self.rmse,
                # "mae": self.mae,
                # "r_squared": self.r_squared,
                # "duration_seconds": duration,
                # "duration_pretty": formatted_time,
                #     })

                coeff_results.append(
                    {
                        "n_samples": n,
                        "method": method,
                        "rmse": model.rmse,
                        "mae": model.mae,
                        "r_squared": model.r_squared,
                        "beta_0": model.beta_0_hat,
                        "beta_1": model.beta_1_hat,
                        "duration_seconds": duration,
                        "duration_pretty": formatted_time,
                    }
                )

            except Exception as e:
                print(f"[!] Method {method} failed at n={n}: {str(e)}")

        return coeff_results

    def simulate(
        self,
        data: pd.DataFrame = None,
        n_samples_list: list = [10, 100, 1000],
        noise: float = 1.0,
        seed: int = 42,
        methods=None,
    ) -> pd.DataFrame:
        """
        Run all coefficient estimation methods on datasets of varying sample sizes.

        Returns:
            pd.DataFrame: A dataframe of results from all methods and sample sizes.
        """

        results = []

        if data is not None and "x" in data and "y" in data:
            x = data["x"]
            y = data["y"]

            results.extend(
                self._coefficient_estimators(x, y, n=x.size, methods=methods)
            )

        else:
            np.random.seed(seed)
            for n in n_samples_list:
                # Generate synthetic data
                x = pd.Series(2 * np.random.rand(n))
                y = 4 + 3 * x + np.random.randn(n) * noise

                results.extend(self._coefficient_estimators(x, y, n, methods))

        return pd.DataFrame(results).sort_values(
            ["n_samples", "method", "duration_seconds"]
        )

    def summary(self) -> None:
        """
        Print a summary of the fitted model coefficients and metrics.
        """
        print(f"Method: {self.method}")
        print(f"Intercept (β₀): {self.beta_0_hat:.4f}")
        print(f"Slope (β₁): {self.beta_1_hat:.4f}")
        if self.r_squared is not None:
            print(f"R²: {self.r_squared:.4f}")
        print()

    def benchmark_summary(self) -> pd.DataFrame:
        """
        Run a simulation and print the results.

        Returns:
            pd.DataFrame: Benchmark results across methods and sample sizes.
        """
        df = self.simulate()
        print(df)
        print()
        return df


if __name__ == "__main__":

    x_test = pd.Series(2 * np.random.rand(100))
    y_test = pd.Series(4 + 3 * x_test + np.random.randn(100))

    benchmark_model = SimpleLinearRegressionSklearn(x_test, y_test)
    benchmark_model.benchmark_summary()

    model_normal = SimpleLinearRegressionSklearn(x_test, y_test)
    model_normal.fit("normal_equation")
    model_normal.summary()

    model_gd_batch = SimpleLinearRegressionSklearn(x_test, y_test)
    model_gd_batch.fit("gradient_descent_batch")
    model_gd_batch.summary()

    model_gd_stochastic = SimpleLinearRegressionSklearn(x_test, y_test)
    model_gd_stochastic.fit("gradient_descent_stochastic")
    model_gd_stochastic.summary()

    model_gd_mini_batch = SimpleLinearRegressionSklearn(x_test, y_test)
    model_gd_mini_batch.fit("gradient_descent_mini_batch")
    model_gd_mini_batch.summary()
