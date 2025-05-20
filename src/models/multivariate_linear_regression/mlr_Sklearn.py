# src/models/mulivariatve_linear_regression/mlr_Sklearn.py
"""
Implements Multivariate Linear Regression using Scikit-learn's interface.

Supports:
- normal_equation (matrix inverse)
- batch, stochastic, mini-batch gradient descent
"""

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor

from src.config.defaults import DEFAULT_TRAINING_KWARGS_SKLEARN
from src.core.metrics.metrics import (
    calculate_adjusted_r_squared,
    calculate_mae,
    calculate_median_ae,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from src.core.registry import register_model
from src.data.generate_data import generate_multivariate_synthetic_data_regression
from src.models.ground_up_ml_base_model import GroundUpMLBaseModel
from src.utils.config import get_config, safe_kwargs
from src.utils.utils import format_duration


@register_model(
    name="SK:LinReg-Multi",
    learning_type="supervised",
    task_type="regression",
    data_shape="multivariate",
    implementation="sklearn",
    model_type="linear regression",
)
class MultivariateLinearRegressionSklearn(GroundUpMLBaseModel):
    """
    A wrapper around sklearn's LinearRegression and SGDRegressor
    that mimics the interface of MultivariateLinearRegressionFromScratch
    for consistent visualization and benchmarking.

    The interface of MultivariateLinearRegressionSklearn matches
    MultivariateLinearRegressionFromScratch and MultivariateLinearRegressionPyTorch.
    """

    ALL_METHODS = [
        "normal_equation",
        "gradient_descent_batch",
        "gradient_descent_stochastic",
        "gradient_descent_mini_batch",
    ]

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the model with feature and target data.

        Args:
            X (pd.DataFrame): Feature values.
            y (pd.Series): Target values.
        """
        # Store original feature names
        self.feature_names = ["Intercept"] + list(X.columns)

        self.X: np.ndarray = X.to_numpy()
        self.y: np.ndarray = y.to_numpy()

        # Method to calculate coefficient estimations
        self.method: str = None

        self.model: Optional[Union[LinearRegression, SGDRegressor]] = None

        # Coefficient Estimations
        self.theta_hat: Optional[float] = None

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
        return "MultivariateLinearRegression-Sklearn"

    def fit(
        self,
        method: str = None,
        schedule: str = None,
        schedule_kwargs: dict = {},
        training_kwargs: dict = {},
    ) -> None:
        """
        Routing function to call specific coefficient estimator methods

        Args:
            method (str): Method to calculate coefficients.
            schedule (str, optional): Learning rate schedule name.
            schedule_kwargs (dict, optional): Schedule hyperparameters.
            training_kwargs (dict, optional): Training loop hyperparameters.

        Raises:
            ValueError: Raise error if value in 'method' is not a known one.
        """
        self.method = method

        FIT_METHODS = {
            "normal_equation": (self._fit_normal_equation, False),
            "gradient_descent_batch": (self._fit_gradient_descent_batch, True),
            "gradient_descent_stochastic": (
                self._fit_gradient_descent_stochastic,
                True,
            ),
            "gradient_descent_mini_batch": (
                self._fit_gradient_descent_mini_batch,
                True,
            ),
        }

        # Raise error if value in 'method' is not a known one
        if method not in FIT_METHODS:
            raise ValueError(
                f"Unknown method '{method}' for MultivariateLinearRegressionSklearn"
                f"Choose one of: {list(FIT_METHODS.keys())}"
            )

        # Call respective coefficient estimator
        fit_method, accepts_schedules = FIT_METHODS[method]
        if accepts_schedules:
            fit_method(
                training_kwargs=training_kwargs,
            )
        else:
            fit_method()

    def _fit_normal_equation(self):
        """
        Fits the model using the Normal Equation:
            Theta_hat = ((X.T ⋅ X)^-1) ⋅ X.T ⋅ y
        """
        self.model = LinearRegression().fit(self.X, self.y)

        intercept = np.atleast_1d(self.model.intercept_).ravel()
        coef = self.model.coef_.ravel()

        # Handle shape consistency (reshape to column vector)
        self.theta_hat = np.concatenate([intercept, coef]).reshape(-1, 1)

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using batch gradient descent.
        Minimizes the MSE cost function by iteratively updating θ (theta).

        Gradient: ∂J(θ)/∂θ = (2/m) ⋅ Xᵀ(Xθ - y)

        Args:
            training_kwargs (dict): Hyperparameters for training loop
                Defaults to None.
        """
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_SKLEARN)

        # Pull out values you use directly
        max_epochs = training_kwargs.get("max_epochs", 50)

        # Clean kwargs to only pass what SGDRegressor accepts
        sgd_kwargs = safe_kwargs(SGDRegressor, training_kwargs)

        self.model = SGDRegressor(**sgd_kwargs)

        cost_history = []

        for epoch in range(max_epochs):
            self.model.partial_fit(self.X, self.y)
            y_pred = self.model.predict(self.X)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        intercept = np.atleast_1d(self.model.intercept_).ravel()
        coef = self.model.coef_.ravel()

        # Handle shape consistency (reshape to column vector)
        self.theta_hat = np.concatenate([intercept, coef]).reshape(-1, 1)

        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def _fit_gradient_descent_stochastic(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using stochastic gradient descent (SGD) with a
        decaying learning rate.

        Parameters are updated after each training example, using one
        randomly shuffled sample per step.

        Args:
            training_kwargs (dict): Hyperparameters for training loop
                Defaults to None.
        """
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_SKLEARN)

        # pull out values you use directly
        max_epochs = training_kwargs.get("max_epochs", 50)
        n = self.X.shape[0]

        # Clean kwatgs to only pass what SGDRegressor accepts
        sgd_kwargs = safe_kwargs(SGDRegressor, training_kwargs)

        self.model = SGDRegressor(**sgd_kwargs)

        cost_history = []

        for epoch in range(max_epochs):
            indices = np.random.permutation(n)
            for i in indices:
                X_i = np.reshape(self.X[i], (1, -1))
                y_i = [self.y[i]]
                self.model.partial_fit(X_i, y_i)

            y_pred = self.model.predict(self.X)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        intercept = np.atleast_1d(self.model.intercept_).ravel()
        coef = self.model.coef_.ravel()

        # Handle shape consistency (reshape to column vector)
        self.theta_hat = np.concatenate([intercept, coef]).reshape(-1, 1)

        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def _fit_gradient_descent_mini_batch(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using mini-batch gradient descent with a decaying learning rate.
        Each epoch processes randomly shuffled mini-batches of the training data.

        Args:
            training_kwargs (dict): Hyperparameters for training loop
                Defaults to None.
        """
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_SKLEARN)

        # Pull out values you use dirrectly
        max_epochs = training_kwargs.get("max_epochs", 50)
        batch_size = training_kwargs.get("batch_size", 50)
        n = self.X.shape[0]

        # Clean kwargs to only pass what SGDRegressor accepts
        sgd_kwargs = safe_kwargs(SGDRegressor, training_kwargs)

        self.model = SGDRegressor(**sgd_kwargs)

        cost_history = []

        for epoch in range(max_epochs):

            indices = np.random.permutation(n)
            X_shuffled, y_shuffled = self.X[indices], self.y[indices]

            for i in range(0, n, batch_size):
                X_mini = X_shuffled[i : i + batch_size]
                y_mini = y_shuffled[i : i + batch_size]
                self.model.partial_fit(X_mini, y_mini)

            y_pred = self.model.predict(self.X)
            cost = calculate_mse(self.y, y_pred)
            cost_history.append(cost)

        # Unpack final parameters
        intercept = np.atleast_1d(self.model.intercept_).ravel()
        coef = self.model.coef_.ravel()

        # Handle shape consistency (reshape to column vector)
        self.theta_hat = np.concatenate([intercept, coef]).reshape(-1, 1)

        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def predict(
        self, x_new: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Predict target values for new data.

        Args:
            x_new (Optional): Input features to predict on.

        Returns:
            np.ndarray: Predicted target values.
        """
        if x_new is None:
            x_new = self.X
        elif isinstance(x_new, pd.DataFrame):
            x_new = x_new.to_numpy()

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
        self.adjusted_r_squared = calculate_adjusted_r_squared(self.y, y_pred)

    def _coefficient_estimators(
        self, X: pd.DataFrame, y: pd.Series, n: int, methods: str = None
    ) -> list:
        """
        Run each coefficient estimation method on the current dataset.

        Args:
            x (pd.DataFrame): _description_
            y (pd.Series): _description_
            n (int): _description_
            methods (str, optional): _description_. Defaults to None.

        Returns:
            list: A list of dictionaries containing model diagnostics per method.
        """
        coeff_results = []

        for method in methods:
            model = MultivariateLinearRegressionSklearn(X, y)
            try:
                accepts_schedule = "gradient_descent" in method
                start_time = time.perf_counter()

                if accepts_schedule:
                    model.fit(
                        method=method,
                        schedule="time_decay",
                        schedule_kwargs={},
                        training_kwargs={},
                    )
                else:
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
                        "mse": model.mse,
                        "rmse": model.rmse,
                        "mae": model.mae,
                        "median_ae": model.median_ae,
                        "r_squared": model.r_squared,
                        "adjusted_r_squared": model.adjusted_r_squared,
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
        methods: list = None,
    ) -> pd.DataFrame:
        """
        Run all coefficient estimation methods on datasets of varying sample sizes.

        Returns:
            pd.DataFrame: A dataframe of results from all methods and sample sizes.
        """

        results = []
        methods = methods or self.ALL_METHODS

        if data is not None and "x" in data and "y" in data:
            X = data.drop("y", axis=1)
            y = data["y"]

            results.extend(
                self._coefficient_estimators(X, y, n=X.shape[0], methods=methods)
            )

        else:
            for n in n_samples_list:
                # Generate synthetic data
                X, y = generate_multivariate_synthetic_data_regression(
                    n=n, noise=noise, seed=seed
                )

                results.extend(self._coefficient_estimators(X, y, n, methods))

        df = pd.DataFrame(results)
        if not df.empty and all(
            col in df.columns for col in ["n_samples", "method", "duration_seconds"]
        ):
            df = df.sort_values(["n_samples", "method", "duration_seconds"])

        return df

    def summary(self) -> None:
        """
        Print a summary of the fitted model coefficients and metrics.
        """
        print("\n[Model Summary]")
        print(f"Method: {self.method}\n")

        try:
            coefficients = self.theta_hat.ravel()
            for name, val in zip(self.feature_names, coefficients):
                print(f"{name}: {val:.4f}")
        except Exception as e:
            print("[!] Error printing coefficients — theta_hat shape mismatch?")
            print(f"theta_hat: {self.theta_hat}")
            raise e  # Re-raise so you still get traceback

        print("\n[Metrics]")
        print(f"MSE: {self.mse:.4f}")
        print(f"RMSE: {self.rmse:.4f}")
        print(f"MAE: {self.mae:.4f}")
        print(f"Median AE: {self.median_ae:.4f}")
        print(f"R²: {self.r_squared:.4f}")
        print(f"Adjusted R²: {self.adjusted_r_squared:.4f}")
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
