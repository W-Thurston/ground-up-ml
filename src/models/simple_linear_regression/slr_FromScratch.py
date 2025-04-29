# src/models/simple_linear_regression/slr_FromScratch.py
"""
Implements Simple Linear Regression using from-scratch math and logic.

Supports:
- beta_estimations (ISLR)
- normal_equation (matrix inverse)
- batch, stochastic, mini-batch gradient descent
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

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

# from src.utils.mlflow_logger import log_metrics, log_params, start_run


@register_model("from_scratch", task_type="regression", group="simple_linear")
class SimpleLinearRegressionFromScratch(GroundUpMLBaseModel):
    """
    A Python implementation of Simple Linear Regression.
    The interface of SimpleLinearRegressionFromScratch matches
    SimpleLinearRegressionSklearn and SimpleLinearRegressionPyTorch.
    """

    ALL_METHODS = [
        "beta_estimations",
        "normal_equation",
        "gradient_descent_batch",
        "gradient_descent_stochastic",
        "gradient_descent_mini_batch",
    ]

    def __init__(self, x: pd.Series, y: pd.Series):
        self.x: np.ndarray = x.to_numpy()
        self.y: np.ndarray = y.to_numpy()

        # Method to calculate coefficient estimations
        self.method: str = None

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

        self.diagnostics = {}

    @property
    def name(self):
        return "SimpleLinearRegression-FromScratch"

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
            "beta_estimations": self._fit_beta_estimations,
            "normal_equation": self._fit_normal_equation,
            "gradient_descent_batch": self._fit_gradient_descent_batch,
            "gradient_descent_stochastic": self._fit_gradient_descent_stochastic,
            "gradient_descent_mini_batch": self._fit_gradient_descent_mini_batch,
        }

        # Raise error if value in 'method' is not a known one
        if method not in fit_methods:
            raise ValueError(
                f"Unknown method '{method}' for SimpleLinearRegressionFromScratch."
            )

        # Call respective coefficient estimator
        fit_methods[method]()

    def _fit_beta_estimations(self) -> None:
        """
        Fits the model using Beta estimations:
            β_1_hat = (∑(x_i - x_bar)*(y_i - y_bar)) / ∑(x_i - x_bar)^2
            β_0_hat = y_bar - β_1_hat * x

        """

        # Intitialize feature and target means
        x_bar = np.mean(self.x)
        y_bar = np.mean(self.y)

        # Compute coefficient estimates
        self.beta_1_hat = np.sum((self.x - x_bar) * (self.y - y_bar)) / np.sum(
            (self.x - x_bar) ** 2
        )
        self.beta_0_hat = y_bar - self.beta_1_hat * x_bar

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_normal_equation(self):
        """
        Fits the model using the Normal Equation:
            Theta_hat = ((X.T ⋅ X)^-1) ⋅ X.T ⋅ y

        """

        # Add intercept column
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]

        # Normal Equation calculation
        theta_hat = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ self.y)

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0]
        self.beta_1_hat = theta_hat[1]

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(self) -> None:
        """
        Fits the model using batch gradient descent.
        Minimizes the MSE cost function by iteratively updating θ (theta).

        Gradient: ∂/∂θ J(θ) = 2/m * Xᵀ(Xθ - y)
        """

        # Add intercept column
        m = self.x.shape[0]
        X_b = np.c_[np.ones((m, 1)), self.x]
        y = self.y.reshape(-1, 1)  # ensure shape (m, 1)

        eta_0 = 0.1  # Initial learning rate
        decay_rate = 0.01  # Controls how fast learning rate decreases
        n_iterations = 1000
        tolerance = 1e-6

        # Initialize parameters randomly
        theta_hat = np.random.randn(2, 1) * 0.01
        prev_theta = theta_hat.copy()

        cost_history = []

        for iteration in range(n_iterations):
            # Learning rate decay
            eta_t = eta_0 / (1 + decay_rate * iteration)

            # Calculate gradient
            gradients = (2 / m) * X_b.T @ ((X_b @ theta_hat) - y)

            # Home much to update theta_hat by
            update = eta_t * gradients

            # Update theta_hat
            theta_hat -= update

            # Keep track of our 'cost'. We are aiming to minimize this
            cost = np.mean((X_b @ theta_hat - y) ** 2)
            cost_history.append(cost)

            # Convergence check
            if np.linalg.norm(theta_hat - prev_theta) < tolerance:
                print(f"[✔] Converged at iteration {iteration}")
                break

            # Update previous theta for convergence checks
            prev_theta = theta_hat.copy()

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_stochastic(
        self, t0: int = 5, t1: int = 50, n_epochs: int = 50, tolerance: float = 1e-6
    ) -> None:
        """
        Fits the model using stochastic gradient descent (SGD) with a
        decaying learning rate.

        Parameters are updated after each training example, using one
        randomly shuffled sample per step.

        Args:
            t0 (int, optional): _description_. Defaults to 5.
            t1 (int, optional): _description_. Defaults to 50.
            n_epochs (int, optional): _description_. Defaults to 50.
            tolerance (float, optional): _description_. Defaults to 1e-6.

        """

        # Add intercept column
        m = self.x.shape[0]
        X_b = np.c_[np.ones((m, 1)), self.x]
        y = self.y.reshape(-1, 1)  # ensure shape (m, 1)

        # Random initialization
        theta_hat = np.random.randn(2, 1) * 0.01
        t = 0
        cost_history = []

        def learning_schedule(t):
            return t0 / (t + t1)

        for epoch in range(n_epochs):

            # Shuffle the order of observations each epoch
            indices = np.random.permutation(m)
            for i in indices:
                # Pull out our single observation to update gradients
                xi = X_b[i : i + 1]
                yi = y[i : i + 1]

                # Update gradients
                gradients = 2 * xi.T @ ((xi @ theta_hat) - yi)

                # Learning rate decay
                eta = learning_schedule(t)

                # Home much to update theta_hat by
                update = eta * gradients

                # Update theta_hat
                theta_hat -= update

                # Keep track of our 'cost'. We are aiming to minimize this
                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                # Convergence check
                if np.linalg.norm(update) < tolerance:
                    # print(f"[✔] SGD converged at epoch {epoch}, sample {i}, t={t}")
                    break
                t += 1  # count each update step for learning rate decay

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_mini_batch(
        self,
        batch_size: int = 16,
        eta_0: float = 0.1,
        decay_rate: float = 0.01,
        n_epochs: int = 50,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Fits the model using mini-batch gradient descent with a decaying learning rate.
        Each epoch processes randomly shuffled mini-batches of the training data.

        Args:
            batch_size (int, optional): Number of samples per mini-batch.
                Defaults to 16.
            eta_0 (float, optional): Initial learning rate.
                Defaults to 0.1.
            decay_rate (float, optional): Controls how quickly eta decays.
                Defaults to 0.01.
            n_epochs (int, optional): Number of passes through the data.
                Defaults to 50.
            tolerance (float, optional): Threshold for stopping based on
                parameter stability. Defaults to 1e-6.
        """

        # Add intercept column
        m = self.x.shape[0]
        X_b = np.c_[np.ones((m, 1)), self.x]
        y = self.y.reshape(-1, 1)

        # Random initialization
        theta_hat = np.random.randn(2, 1) * 0.01
        t = 0
        cost_history = []

        for epoch in range(n_epochs):

            # Shuffle order of observations for each epoch
            indices = np.random.permutation(m)
            X_b_shuffled, y_shuffled = X_b[indices], y[indices]

            for i in range(0, m, batch_size):
                X_mini = X_b_shuffled[i : i + batch_size]
                y_mini = y_shuffled[i : i + batch_size]

                # Calculate gradients
                gradients = (2 / len(X_mini)) * X_mini.T @ (X_mini @ theta_hat - y_mini)

                # Learning decay rate
                eta_t = eta_0 / (1 + decay_rate * t)

                # How much to update theta_hat by
                update = eta_t * gradients

                # Update theta_hat
                theta_hat -= update

                # Keep track of our 'cost'. We are aiming to minimize this
                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                # Convergence check
                if np.linalg.norm(update) < tolerance:
                    print(
                        f"[✔] Mini-batch GD converged at epoch {epoch}, "
                        f"batch {i // batch_size}, t={t}"
                    )
                    break
                t += 1  # count each update step for learning rate decay

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def predict(self, x_new: pd.Series = None):
        """
        Calculate predicted values based on learned Simple Linear Regression training

        Args:
            x_new (pd.Series, optional): New observations to predict target for.
                Defaults to None.

        Returns:
            float: Predicted value of x_new's 'y' value
        """
        if x_new is None:
            x_new = self.x
        return self.beta_0_hat + self.beta_1_hat * x_new

    def residuals(self) -> float:
        """
        Returns:
            float: Difference between actual 'y' value and predicted 'y' value.
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
        methods = methods or SimpleLinearRegressionFromScratch.ALL_METHODS

        for method in methods:
            model = SimpleLinearRegressionFromScratch(x, y)
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

    # benchmark_model = SimpleLinearRegressionFromScratch(x_test, y_test)
    # benchmark_model.benchmark_summary()

    model_beta = SimpleLinearRegressionFromScratch(x_test, y_test)
    model_beta.fit("beta_estimations")
    model_beta.summary()

    # model_normal = SimpleLinearRegressionFromScratch(x_test, y_test)
    # model_normal.fit("normal_equation")
    # model_normal.summary()

    # model_gd_batch = SimpleLinearRegressionFromScratch(x_test, y_test)
    # model_gd_batch.fit("gradient_descent_batch")
    # model_gd_batch.summary()

    # model_gd_stochastic = SimpleLinearRegressionFromScratch(x_test, y_test)
    # model_gd_stochastic.fit("gradient_descent_stochastic")
    # model_gd_stochastic.summary()

    # model_gd_mini_batch = SimpleLinearRegressionFromScratch(x_test, y_test)
    # model_gd_mini_batch.fit("gradient_descent_mini_batch")
    # model_gd_mini_batch.summary()
