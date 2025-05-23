# src/models/multivariate_linear_regression/mlr_FromScratch.py
"""
Implements Multivariate Linear Regression using from-scratch math and logic.

Supports:
- nomal_equation (matrix inverse)
- batch, stochastic, mini-batch gradient descent
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

from src.config.defaults import (
    DEFAULT_SCHEDULE_KWARGS,
    DEFAULT_TRAINING_KWARGS_FROM_SCRATCH,
)
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
from src.utils.config import get_config
from src.utils.learning_rate import get_learning_rate_schedule
from src.utils.utils import format_duration

# from src.utils.mlflow_logger import log_metrics, log_params, start_run


@register_model(
    name="FS:LinReg-Multi",
    learning_type="supervised",
    task_type="regression",
    data_shape="multivariate",
    implementation="from-scratch",
    model_type="linear regression",
)
class MultivariateLinearRegressionFromScratch(GroundUpMLBaseModel):
    """
    A Python implementation of Multivariate Linear Regression.
    The interface of MultivariateLinearRegressionFromScratch derives from
    GroundUpBaseModel and matches the Sklearn and Pytorch implementations
    in this folder.
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

        # Store X with bias term baked in
        m = X.shape[0]
        self.X_b: np.ndarray = np.c_[np.ones((m, 1)), X.to_numpy()]
        self.y: np.ndarray = y.to_numpy().reshape(-1, 1)

        # Method to calculate coefficient estimations
        self.method: str = None

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
        return "MultivariateLinearRegression-FromScratch"

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
                f"Unknown method '{method}' for MultivariateLinearRegressionFromScratch"
                f"Choose one of: {list(FIT_METHODS.keys())}"
            )

        # Call respective coefficient estimator
        fit_method, accepts_schedules = FIT_METHODS[method]
        if accepts_schedules:
            schedule_name = schedule or "time_decay"
            fit_method(
                schedule_name=schedule_name,
                schedule_kwargs=schedule_kwargs,
                training_kwargs=training_kwargs,
            )
        else:
            fit_method()

    def _fit_normal_equation(self):
        """
        Fits the model using the Normal Equation:
            Theta_hat = ((X.T ⋅ X)^-1) ⋅ X.T ⋅ y
        """

        # Normal Equation calculation
        self.theta_hat = np.linalg.pinv(self.X_b.T @ self.X_b) @ self.X_b.T @ self.y

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(
        self,
        schedule_name="time_decay",
        schedule_kwargs=None,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using batch gradient descent.
        Minimizes the MSE cost function by iteratively updating θ (theta).

        Gradient: ∂J(θ)/∂θ = (2/m) ⋅ Xᵀ(Xθ - y)

        Args:
            schedule_name (str): Name of the learning rate schedule.
            schedule_kwargs (dict): Hyperparameters for the learning rate scheduler.
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - convergence_tol (float): Threshold for stopping early
                - theta_init_scale (float): Variance scale for initial theta
                - verbose (bool): Whether to print convergence updates
                - batch_size (int, optional): Only for mini-batch
        """

        # Step 1: Pull X_b and y from self
        m = self.X_b.shape[0]
        X_b = self.X_b
        y = self.y

        # Step 2: Set hyperparameters
        schedule_kwargs = get_config(
            schedule_kwargs, DEFAULT_SCHEDULE_KWARGS.get(schedule_name, {})
        )
        schedule = get_learning_rate_schedule(schedule_name, **schedule_kwargs)

        training_kwargs = get_config(
            training_kwargs, DEFAULT_TRAINING_KWARGS_FROM_SCRATCH
        )
        max_epochs = training_kwargs["max_epochs"]
        convergence_tol = training_kwargs["convergence_tol"]
        theta_init_scale = training_kwargs["theta_init_scale"]
        verbose = training_kwargs["verbose"]

        # Step 3: Initialize theta_hat randomly
        n_features = X_b.shape[1]
        theta_hat = np.random.randn(n_features, 1) * theta_init_scale

        # Step 4: Create storage
        cost_history = []

        # Step 5: Loop over iterations
        for epoch in range(max_epochs):
            # Step 5a: Learning rate decay
            eta = schedule(epoch)

            # Step 5b: Compute Gradients
            gradients = (2 / m) * X_b.T @ (X_b @ theta_hat - y)

            # Step 5c: How much to update theta_hat by
            update = eta * gradients

            # Step 5d: Update theta_hat
            theta_hat -= update

            # Step 5e: Compute cost and save it
            cost = np.mean((X_b @ theta_hat - y) ** 2)
            cost_history.append(cost)

            # Step 5f: Convergence check
            if np.linalg.norm(update) < convergence_tol:
                if verbose:
                    print(f"[✔] Converged at epoch {epoch}")
                    print(
                        f"[ℹ️] Using schedule '{schedule_name}' with: {schedule_kwargs}"
                    )
                break

        # Step 6: Store:
        # Final theta_hat
        self.theta_hat = theta_hat

        # Diagnostics dictionary
        self.diagnostics = {
            "cost_history": cost_history,
            "epochs_run": epoch + 1,
            "schedule": schedule_name,
            "schedule_kwargs": schedule_kwargs,
            "training_kwargs": training_kwargs,
            "final_eta": eta,
            "converged": epoch < max_epochs - 1,
            "theta_hat": theta_hat.ravel(),
        }

    def _fit_gradient_descent_stochastic(
        self,
        schedule_name="time_decay",
        schedule_kwargs=None,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using stochastic gradient descent (SGD) with a
        decaying learning rate.

        Parameters are updated after each training example, using one
        randomly shuffled sample per step.

        Args:
            schedule_name (str): Name of the learning rate schedule.
            schedule_kwargs (dict): Hyperparameters for the learning rate scheduler.
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - convergence_tol (float): Threshold for stopping early
                - theta_init_scale (float): Variance scale for initial theta
                - verbose (bool): Whether to print convergence updates
                - batch_size (int, optional): Only for mini-batch
        """
        # Step 1: Pull X_b and y from self
        m = self.X_b.shape[0]
        X_b = self.X_b
        y = self.y

        # Step 2: Set hyperparameters
        schedule_kwargs = get_config(
            schedule_kwargs, DEFAULT_SCHEDULE_KWARGS.get(schedule_name, {})
        )
        schedule = get_learning_rate_schedule(schedule_name, **schedule_kwargs)

        training_kwargs = get_config(
            training_kwargs, DEFAULT_TRAINING_KWARGS_FROM_SCRATCH
        )
        max_epochs = training_kwargs["max_epochs"]
        convergence_tol = training_kwargs["convergence_tol"]
        theta_init_scale = training_kwargs["theta_init_scale"]
        verbose = training_kwargs["verbose"]

        # Step 3: Initialize theta_hat randomly
        n_features = X_b.shape[1]
        theta_hat = np.random.randn(n_features, 1) * theta_init_scale
        t = 0

        # Step 4: Create storage
        cost_history = []

        # Step 5: Loop over iterations
        for epoch in range(max_epochs):

            # Step 5a: Shuffle the order of observations each epoch
            indices = np.random.permutation(m)
            for i in indices:
                # Step 5b: Pull out our single observation to update gradients
                X_i = X_b[i : i + 1]
                y_i = y[i : i + 1]

                # Step 5c: Learning rate decay
                eta = schedule(t)

                # Step 5d: Compute Gradients
                gradients = 2 * X_i.T @ (X_i @ theta_hat - y_i)

                # Step 5e: How much to update theta_hat by
                update = eta * gradients

                # Step 5f: Update theta_hat
                theta_hat -= update

                # Step 5g: Compute cost and save it
                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                # Step 5h: Convergence check
                if np.linalg.norm(update) < convergence_tol:
                    if verbose:
                        print(f"[✔] Converged at epoch {epoch}")
                        print(
                            f"[ℹ️] Using schedule '{schedule_name}' with:"
                            f" {schedule_kwargs}"
                        )
                    break
                t += 1  # count each update step for learning rate decay

        # Step 6: Store:
        # Final theta_hat
        self.theta_hat = theta_hat

        # Diagnostics dictionary
        self.diagnostics = {
            "cost_history": cost_history,
            "epochs_run": epoch + 1,
            "schedule": schedule_name,
            "schedule_kwargs": schedule_kwargs,
            "training_kwargs": training_kwargs,
            "final_eta": eta,
            "converged": epoch < max_epochs - 1,
            "theta_hat": theta_hat.ravel(),
        }

    def _fit_gradient_descent_mini_batch(
        self,
        schedule_name="time_decay",
        schedule_kwargs=None,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using mini-batch gradient descent with a decaying learning rate.
        Each epoch processes randomly shuffled mini-batches of the training data.

        Args:
            schedule_name (str): Name of the learning rate schedule.
            schedule_kwargs (dict): Hyperparameters for the learning rate scheduler.
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - convergence_tol (float): Threshold for stopping early
                - theta_init_scale (float): Variance scale for initial theta
                - verbose (bool): Whether to print convergence updates
                - batch_size (int, optional): Only for mini-batch
        """
        # Step 1: Pull X_b and y from self
        m = self.X_b.shape[0]
        X_b = self.X_b
        y = self.y

        # Step 2: Set hyperparameters
        schedule_kwargs = get_config(
            schedule_kwargs, DEFAULT_SCHEDULE_KWARGS.get(schedule_name, {})
        )
        schedule = get_learning_rate_schedule(schedule_name, **schedule_kwargs)

        training_kwargs = get_config(
            training_kwargs, DEFAULT_TRAINING_KWARGS_FROM_SCRATCH
        )
        max_epochs = training_kwargs["max_epochs"]
        batch_size = training_kwargs["batch_size"]
        convergence_tol = training_kwargs["convergence_tol"]
        theta_init_scale = training_kwargs["theta_init_scale"]
        verbose = training_kwargs["verbose"]

        # Step 3: Initialize theta_hat randomly
        n_features = X_b.shape[1]
        theta_hat = np.random.randn(n_features, 1) * theta_init_scale
        t = 0

        # Step 4: Create storage
        cost_history = []

        # Step 5: Loop over iterations
        for epoch in range(max_epochs):

            # Step 5a: Shuffle order of observations for each epoch
            indices = np.random.permutation(m)
            X_b_shuffled, y_shuffled = X_b[indices], y[indices]

            for i in range(0, m, batch_size):
                # Step 5b: Pull out our single observation to update gradients
                X_mini = X_b_shuffled[i : i + batch_size]
                y_mini = y_shuffled[i : i + batch_size]

                # Step 5c: Learning rate decay
                eta = schedule(t)

                # Step 5d: Compute Gradients
                gradients = (2 / len(X_mini)) * X_mini.T @ (X_mini @ theta_hat - y_mini)

                # Step 5e: How much to update theta_hat by
                update = eta * gradients

                # Step 5f: Update theta_hat
                theta_hat -= update

                # Step 5g: Compute cost and save it
                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                # Step 5h: Convergence check
                if np.linalg.norm(update) < convergence_tol:
                    if verbose:
                        print(f"[✔] Converged at epoch {epoch}")
                        print(
                            f"[ℹ️] Using schedule '{schedule_name}' with:"
                            f" {schedule_kwargs}"
                        )
                    break
                t += 1  # count each update step for learning rate decay

        # Step 6: Store:
        # Final theta_hat
        self.theta_hat = theta_hat

        # Diagnostics dictionary
        self.diagnostics = {
            "cost_history": cost_history,
            "epochs_run": epoch + 1,
            "schedule": schedule_name,
            "schedule_kwargs": schedule_kwargs,
            "training_kwargs": training_kwargs,
            "final_eta": eta,
            "converged": epoch < max_epochs - 1,
            "theta_hat": theta_hat.ravel(),
        }

    def predict(self, x_new: pd.DataFrame = None):
        """
        Calculate prediced values based on learned Multivariate
        Linear Regression training.

        Args:
            x_new (pd.DataFrame, optional): New observations to predict target for.
                Defaults to None.

        Returns:
            float: Predicted value of x_new's 'y' value
        """
        if x_new is None:
            X_b = self.X_b
        else:
            if isinstance(x_new, pd.DataFrame):
                x_new = x_new.to_numpy()
            X_b = np.c_[np.ones((x_new.shape[0], 1)), x_new]

        return X_b @ self.theta_hat

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
            model = MultivariateLinearRegressionFromScratch(X, y)
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
