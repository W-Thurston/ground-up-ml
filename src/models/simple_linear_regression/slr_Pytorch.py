# src/models/simple_linear_regression/slr_Pytorch.py
"""
Implements Simple Linear Regression using PyTorch's interface.

Supports:
- normal_equation
- batch, stochastic, mini-batch gradient descent

Designed to mirror the from-scratch interface for comparison
"""
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config.defaults import DEFAULT_TRAINING_KWARGS_PYTORCH
from src.core.metrics.metrics import (
    calculate_adjusted_r_squared,
    calculate_mae,
    calculate_median_ae,
    calculate_mse,
    calculate_r_squared,
    calculate_rmse,
)
from src.core.registry import register_model
from src.data.generate_data import generate_singlevariate_synthetic_data_regression
from src.models.ground_up_ml_base_model import GroundUpMLBaseModel
from src.utils.config import get_config, safe_kwargs
from src.utils.utils import format_duration

# from src.utils.visualizations import plot_model_diagnostics
torch.set_default_dtype(torch.float64)


@register_model(
    name="PT:LinReg-Uni",
    learning_type="supervised",
    task_type="regression",
    data_shape="univariate",
    implementation="pytorch",
    model_type="linear regression",
)
class SimpleLinearRegressionPyTorch(GroundUpMLBaseModel):
    """
    A PyTorch implementation of Simple Linear Regression.
    The interface of SimpleLinearRegressionPyTorch matches
    SimpleLinearRegressionFromScratch and SimpleLinearRegressionSklearn.
    """

    ALL_METHODS = [
        "beta_estimations",
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
        self.x: torch.Tensor = torch.tensor(x.values, dtype=torch.float64).view(-1, 1)
        self.y: torch.Tensor = torch.tensor(y.values, dtype=torch.float64).view(-1, 1)

        self.model: Optional[nn.Module] = None
        self.beta_0_hat: Optional[float] = None
        self.beta_1_hat: Optional[float] = None

        self.mse: Optional[float] = None
        self.rmse: Optional[float] = None
        self.mae: Optional[float] = None
        self.median_ae: Optional[float] = None
        self.r_squared: Optional[float] = None
        self.adjusted_r_squared: Optional[float] = None

        self.duration_seconds: float = 0.0

        self.diagnostics: dict = {}

        # PyTorch initialization
        self.model = nn.Linear(1, 1)
        nn.init.normal_(self.model.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.model.bias)
        self.loss_fn = None
        self.optimizer = None

    @property
    def name(self):
        return "SimpleLinearRegression-PyTorch"

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
            method (str, optional): Training method. Defaults to None.

        Raises:
            ValueError: If the method is not a known option.
        """
        self.method = method

        FIT_METHODS = {
            "beta_estimations": (self._fit_beta_estimations, False),
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
                f"Unknown method '{method}' for SimpleLinearRegressionPyTorch."
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

    def _fit_beta_estimations(self) -> None:
        """
        Fits the model using a beta estimation equivalent in PyTorch.
        """

        x = self.x.double()
        y = self.y.double()

        # Initialize feature and target means
        x_bar = torch.mean(x)
        y_bar = torch.mean(y)

        # Computer coefficient estimates
        self.beta_1_hat = np.float64(
            torch.sum((x - x_bar) * (y - y_bar)) / torch.sum((x - x_bar) ** 2)
        )
        self.beta_0_hat = np.float64(y_bar - self.beta_1_hat * x_bar)

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_normal_equation(self) -> None:
        """
        Fits the model using a normal equation equivalent in PyTorch.
        """

        # Add intercept column
        X_b = torch.cat([torch.ones((self.x.shape[0], 1)), self.x], dim=1).double()
        y_b = self.y.double()

        # Normal Equation calculation
        theta_hat = torch.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y_b)

        # Unpack final parameters
        self.beta_0_hat = np.float64(theta_hat[0])
        self.beta_1_hat = np.float64(theta_hat[1])

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using batch gradient descent.

        Args:
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - learning_rate (float): Learning rate for gradient steps
        """
        # Re-initialize model freshly
        self.model = nn.Linear(1, 1)
        nn.init.normal_(self.model.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.model.bias)

        # Set hyperparameters
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_PYTORCH)
        max_epochs = training_kwargs["max_epochs"]

        # Remove keys not accepted by SGD
        optimizer_kwargs = safe_kwargs(torch.optim.SGD, training_kwargs)

        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_kwargs)

        cost_history = []

        for epoch in range(max_epochs):
            # 1. Forward pass
            y_pred = self.model(self.x)  # 1a. Compute predictions
            loss = self.loss_fn(y_pred, self.y)  # 1b. Calculate Loss

            # 2. Backward pass
            self.optimizer.zero_grad()  # 2a. Zero out any gradients
            loss.backward()  # 2b. Recompute fresh gradients

            # 3. Update parameters
            self.optimizer.step()  # 3. Update weights

            # 4. Track and save loss
            cost_history.append(loss.detach().cpu().item())

        # 5. Store parameters
        with torch.no_grad():
            self.beta_1_hat = self.model.weight[0, 0].detach().cpu().item()
            self.beta_0_hat = self.model.bias[0].detach().cpu().item()

        # 6. Populate diagnostic dict
        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def _fit_gradient_descent_stochastic(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using stochastic gradient descent.

        Args:
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - learning_rate (float): Learning rate for gradient steps
        """
        # Re-initialize model freshly
        self.model = nn.Linear(1, 1)
        nn.init.normal_(self.model.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.model.bias)

        # Set hyperparameters
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_PYTORCH)
        max_epochs = training_kwargs["max_epochs"]

        # Remove keys not accepted by SGD
        optimizer_kwargs = safe_kwargs(torch.optim.SGD, training_kwargs)

        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_kwargs)

        m = self.x.shape[0]
        cost_history = []

        for epoch in range(max_epochs):
            # Shuffle the order of observations each epoch
            indices = torch.randperm(m)
            for i in indices:

                # Pull out single observations
                x_i = self.x[i]
                y_i = self.y[i]

                # 1. Forward pass
                y_pred = self.model(x_i)  # 1a. Compute predictions
                loss = self.loss_fn(y_pred, y_i)  # 1b. Calculate Loss

                # 2. Backward pass
                self.optimizer.zero_grad()  # 2a. Zero out any gradients
                loss.backward()  # 2b. Recompute fresh gradients

                # 3. Update parameters
                self.optimizer.step()  # 3. Update weights

            # 4. Track and save loss
            with torch.no_grad():
                y_pred_epoch = self.model(self.x)
                epoch_loss = self.loss_fn(y_pred_epoch, self.y)
                cost_history.append(epoch_loss.detach().cpu().item())

        # 5. Store parameters
        with torch.no_grad():
            self.beta_1_hat = self.model.weight[0, 0].detach().cpu().item()
            self.beta_0_hat = self.model.bias[0].detach().cpu().item()

        # 6. Populate diagnostic dict
        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def _fit_gradient_descent_mini_batch(
        self,
        training_kwargs=None,
    ) -> None:
        """
        Fits the model using mini-batch gradient descent.

        Args:
            training_kwargs (dict): Hyperparameters for training loop, e.g.:
                - max_epochs (int): Maximum number of passes through data
                - learning_rate (float): Learning rate for gradient steps
                - batch_size (int, optional): Only for mini-batch
        """
        # Re-initialize model freshly
        self.model = nn.Linear(1, 1)
        nn.init.normal_(self.model.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.model.bias)

        # Set hyperparameters
        training_kwargs = get_config(training_kwargs, DEFAULT_TRAINING_KWARGS_PYTORCH)
        max_epochs = training_kwargs["max_epochs"]
        batch_size = training_kwargs["batch_size"]

        # Remove keys not accepted by SGD
        optimizer_kwargs = safe_kwargs(torch.optim.SGD, training_kwargs)

        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_kwargs)

        m = self.x.shape[0]

        cost_history = []

        for epoch in range(max_epochs):

            # Shuffle order of observations for each epoch
            indices = torch.randperm(m)
            x_shuffled = self.x[indices]
            y_shuffled = self.y[indices]

            for i in range(0, m, batch_size):

                x_mini = x_shuffled[i : i + batch_size]
                y_mini = y_shuffled[i : i + batch_size]

                # 1. Forward pass
                y_pred = self.model(x_mini)  # 1a. Compute predictions
                loss = self.loss_fn(y_pred, y_mini)  # 1b. Calculate Loss

                # 2. Backward pass
                self.optimizer.zero_grad()  # 2a. Zero out any gradients
                loss.backward()  # 2b. Recompute fresh gradients

                # 3. Update parameters
                self.optimizer.step()  # 3. Update weights

            # 4. Track and save loss
            with torch.no_grad():
                y_pred_epoch = self.model(self.x)
                epoch_loss = self.loss_fn(y_pred_epoch, self.y)
                cost_history.append(epoch_loss.detach().cpu().item())

        # 5. Store parameters
        with torch.no_grad():
            self.beta_1_hat = self.model.weight[0, 0].detach().cpu().item()
            self.beta_0_hat = self.model.bias[0].detach().cpu().item()

        # 6. Populate diagnostic dict
        self.diagnostics = {
            "cost_history": cost_history,
            "training_kwargs": training_kwargs,
        }

    def predict(self, X: Optional[pd.Series] = None) -> np.ndarray:
        """
        Predicts y values using the trained model.

        Args:
            X (Optional[pd.Series], optional): New input data. Defaults to training X.

        Returns:
            np.ndarray: Predicted y values.
        """
        if X is None:
            x_new = self.x
        else:
            if isinstance(X, pd.Series):
                x_new = torch.tensor(X.values, dtype=self.x.dtype).view(-1, 1)
            elif isinstance(X, np.ndarray):
                x_new = torch.tensor(X, dtype=self.x.dtype).reshape(-1, 1)
            else:
                raise ValueError(f"Unsupported input type for X: {type(X)}")

        # If trained via beta_estimations or normal_equation, use manual formula
        if self.method in ["beta_estimations", "normal_equation"]:
            y_pred = (
                self.beta_0_hat
                + self.beta_1_hat * x_new.detach().cpu().numpy().flatten()
            )
        else:
            # Use model prediction
            x_new = x_new.to(self.model.weight.dtype)
            y_pred = self.model(x_new).detach().cpu().numpy().flatten()

        return y_pred

    def residuals(self) -> np.ndarray:
        """
        Calculate residuals between actual and predicted y values.

        Returns:
            np.ndarray: Residuals.
        """
        return self.y - self.predict()

    def fitted(self) -> np.ndarray:
        """
        Return fitted (predicted) values for the training data.

        Returns:
            np.ndarray: Predicted values.
        """
        return self.predict()

    def calculate_metrics(self) -> None:
        y_pred = self.predict()
        y_actual = self.y.detach().cpu().numpy().flatten()  # convert tensor to numpy
        self.mse = calculate_mse(y_actual, y_pred)
        self.rmse = calculate_rmse(y_actual, y_pred)
        self.mae = calculate_mae(y_actual, y_pred)
        self.median_ae = calculate_median_ae(y_actual, y_pred)
        self.r_squared = calculate_r_squared(y_actual, y_pred)
        self.adjusted_r_squared = calculate_adjusted_r_squared(y_actual, y_pred)

    def _coefficient_estimators(
        self, x: pd.Series, y: pd.Series, n: int, methods: str = None
    ) -> list:
        """
        Run each coefficient estimation method on the current dataset.

        Returns:
            list: A list of dictionaries containing model diagnostics per method.
        """

        coeff_results = []

        for method in methods:
            model = SimpleLinearRegressionPyTorch(x, y)
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
        Simulate model fitting across different sample sizes and methods.

        Returns:
            pd.DataFrame: Benchmark results.
        """
        results = []
        methods = methods or self.ALL_METHODS

        if data is not None and "x" in data and "y" in data:
            x = data["x"].to_numpy()
            y = data["y"].to_numpy()

            results.extend(
                self._coefficient_estimators(x, y, n=x.size, methods=methods)
            )

        else:
            for n in n_samples_list:
                # Generate synthetic data
                x, y = generate_singlevariate_synthetic_data_regression(
                    n=n, noise=noise, seed=seed
                )

                results.extend(self._coefficient_estimators(x, y, n, methods))

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
        print(f"Intercept (β₀): {self.beta_0_hat:.4f}")
        print(f"Slope     (β₁): {self.beta_1_hat:.4f}\n")

        print("[Metrics]")
        print(f"MSE: {self.mse:.4f}")
        print(f"RMSE: {self.rmse:.4f}")
        print(f"MAE: {self.mae:.4f}")
        print(f"Median AE: {self.median_ae:.4f}")
        print(f"R²: {self.r_squared:.4f}")
        print(f"Adjusted R²: {self.adjusted_r_squared:.4f}")
        print()

    def benchmark_summary(self) -> pd.DataFrame:
        """
        Run a full simulation and display results.

        Returns:
            pd.DataFrame: Benchmark results.
        """
        df = self.simulate()
        print(df)
        print()
        return df
