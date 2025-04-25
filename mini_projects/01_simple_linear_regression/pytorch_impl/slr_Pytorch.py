# mini_project/01_simple_linear_regression/pytorch_impl/slr_Pytorch.py
"""
Implements Simple Linear Regression using PyTorch's interface.

Supports:
- normal_equation
- batch, stochastic, mini-batch gradient descent

Designed to mirror the from-scratch interface for comparison
"""
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from shared_utils.metrics import calculate_mae, calculate_r_squared, calculate_rmse
from shared_utils.utils import format_duration

# from shared_utils.visualizations import plot_model_diagnostics


class SimpleLinearRegressionPyTorch:
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
        self.x: torch.Tensor = torch.tensor(x.values, dtype=torch.float32).view(-1, 1)
        self.y: torch.Tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        self.model: Optional[nn.Module] = None
        self.beta_0_hat: Optional[float] = None
        self.beta_1_hat: Optional[float] = None

        self.rmse: Optional[float] = None
        self.mae: Optional[float] = None
        self.r_squared: Optional[float] = None

        self.diagnostics: dict = {}

        # PyTorch initialization
        self.model = nn.Linear(1, 1)
        self.loss_fn = None
        self.optimizer = None

    def fit(self, method: str = None) -> None:
        """
        Routing function to call specific coefficient estimator methods

        Args:
            method (str, optional): Training method. Defaults to None.

        Raises:
            ValueError: If the method is not a known option.
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
                f"Unknown method '{method}' for SimpleLinearRegressionPyTorch."
            )

        # Call respective coefficient estimator
        fit_methods[method]()

    def _fit_beta_estimations(self) -> None:
        """
        Fits the model using a beta estimation equivalent in PyTorch.
        """
        # Initialize feature and target means
        x_bar = torch.mean(self.x)
        y_bar = torch.mean(self.y)

        # Computer coefficient estimates
        self.beta_1_hat = (
            (
                torch.sum((self.x - x_bar) * (self.y - y_bar))
                / torch.sum((self.x - x_bar) ** 2)
            )
            .detach()
            .cpu()
            .item()
        )
        self.beta_0_hat = (y_bar - self.beta_1_hat * x_bar).detach().cpu().item()

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_normal_equation(self) -> None:
        """
        Fits the model using a normal equation equivalent in PyTorch.
        """

        # Add intercept column
        X_b = torch.cat([torch.ones((self.x.shape[0], 1)), self.x], dim=1)

        # Normal Equation calculation
        theta_hat = torch.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ self.y)

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0].detach().cpu().item()
        self.beta_1_hat = theta_hat[1].detach().cpu().item()

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(self, n_epochs: int = 50, lr: float = 0.01) -> None:
        """
        Fits the model using batch gradient descent.

        Args:
            n_epochs (int, optional): Number of iterations. Defaults to 50.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """

        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        cost_history = []

        for epoch in range(n_epochs):
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
        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_stochastic(
        self, n_epochs: int = 50, lr: float = 0.01
    ) -> None:
        """
        Fits the model using stochastic gradient descent.

        Args:
            n_epochs (int, optional): Number of iterations. Defaults to 50.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """

        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        m = self.x.shape[0]
        cost_history = []

        for epoch in range(n_epochs):
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
        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_mini_batch(
        self, n_epochs: int = 50, batch_size: int = 32, lr: float = 0.01
    ) -> None:
        """
        Fits the model using mini-batch gradient descent.

        Args:
            n_epochs (int, optional): Number of iterations. Defaults to 50.
            batch_size (int, optional): Size of each mini-batch. Defaults to 32.
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        # Initialize loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        m = self.x.shape[0]

        cost_history = []

        for epoch in range(n_epochs):

            # Shuffle order of observations for each epoch
            indices = torch.randperm(m)
            x_shuffled = self.x[indices]
            y_shuffled = self.y[indices]

            for i in range(0, m, batch_size):

                x = x_shuffled[i : i + batch_size]
                y = y_shuffled[i : i + batch_size]

                # 1. Forward pass
                y_pred = self.model(x)  # 1a. Compute predictions
                loss = self.loss_fn(y_pred, y)  # 1b. Calculate Loss

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
        self.diagnostics = {"cost_history": cost_history}

    def predict(
        self, x_new: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Predict target values for new input data.

        Args:
            x_new (Optional): New input features to predict on.

        Returns:
            np.ndarray: Predicted target values.
        """
        if x_new is None:
            x_new = self.x
        if isinstance(x_new, pd.Series):
            x_new = x_new.to_numpy()
        if isinstance(x_new, np.ndarray):
            x_new = torch.tensor(x_new, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            y_pred = self.model(x_new)
        return y_pred.detach().numpy().flatten()

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

    def _coefficient_estimators(
        self, x: pd.Series, y: pd.Series, n: int, methods: str = None
    ) -> list:
        """
        Run each coefficient estimation method on the current dataset.

        Returns:
            list: A list of dictionaries containing model diagnostics per method.
        """

        coeff_results = []
        methods = methods or SimpleLinearRegressionPyTorch.ALL_METHODS

        for method in methods:
            model = SimpleLinearRegressionPyTorch(x, y)
            try:
                start_time = time.perf_counter()
                model.fit(method=method)
                duration = time.perf_counter() - start_time
                formatted_time = format_duration(duration)

                y_hat = model.predict()

                """
                # Performance metrics
                # RMSE: Root Mean Squared Error - Square root of the average of the
                #   squared differences between the predicted values and the actual
                # MAE: Mean Absolute Error - Average absolute difference between the
                #   predicted values and the actual
                # R_squared:
                """
                model.rmse = calculate_rmse(y, y_hat)
                model.mae = calculate_mae(y, y_hat)
                model.r_squared = calculate_r_squared(y, y_hat)

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
        Simulate model fitting across different sample sizes and methods.

        Returns:
            pd.DataFrame: Benchmark results.
        """
        results = []

        if data is not None and "x" in data and "y" in data:
            x = data["x"].to_numpy()
            y = data["y"].to_numpy()

            results.extend(
                self._coefficient_estimators(x, y, n=x.size, methods=methods)
            )

        else:
            np.random.seed(seed)
            for n in n_samples_list:
                # Generate synthetic data
                x = pd.Series(2 * np.random.rand(n))
                y = pd.Series(4 + 3 * x + np.random.randn(n) * noise)

                results.extend(self._coefficient_estimators(x, y, n, methods))

        return pd.DataFrame(results).sort_values(
            ["n_samples", "method", "duration_seconds"]
        )

    def summary(self) -> None:
        """
        Print a summary of model parameters and diagnostics.
        """
        print(f"Method: {self.method}")
        print(f"Intercept (β₀): {self.beta_0_hat:.4f}")
        print(f"Slope (β₁): {self.beta_1_hat:.4f}")
        if self.r_squared is not None:
            print(f"R²: {self.r_squared:.4f}")
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


if __name__ == "__main__":

    x_test = pd.Series(2 * np.random.rand(100))
    y_test = pd.Series(4 + 3 * x_test + np.random.randn(100))

    benchmark_model = SimpleLinearRegressionPyTorch(x_test, y_test)
    benchmark_model.benchmark_summary()

    model_beta = SimpleLinearRegressionPyTorch(x_test, y_test)
    model_beta.fit("beta_estimations")
    model_beta.summary()
    # plot_model_diagnostics(model_beta)

    model_normal = SimpleLinearRegressionPyTorch(x_test, y_test)
    model_normal.fit("normal_equation")
    model_normal.summary()

    model_gd_batch = SimpleLinearRegressionPyTorch(x_test, y_test)
    model_gd_batch.fit("gradient_descent_batch")
    model_gd_batch.summary()

    model_gd_stochastic = SimpleLinearRegressionPyTorch(x_test, y_test)
    model_gd_stochastic.fit("gradient_descent_stochastic")
    model_gd_stochastic.summary()

    model_gd_mini_batch = SimpleLinearRegressionPyTorch(x_test, y_test)
    model_gd_mini_batch.fit("gradient_descent_mini_batch")
    model_gd_mini_batch.summary()
