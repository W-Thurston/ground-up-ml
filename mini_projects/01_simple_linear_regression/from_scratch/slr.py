# slr.py placeholder

import time

import numpy as np
import pandas as pd

# from shared_utils.mlflow_logger import log_metrics, log_params, start_run


class SimpleLinearRegression:
    ALL_METHODS = [
        "beta_estimations",
        "normal_equation",
        "gradient_descent_batch",
        "gradient_descent_stochastic",
        "gradient_descent_mini_batch",
    ]

    def __init__(self, x: pd.Series, y: pd.Series):
        self.x = x.to_numpy()
        self.y = y.to_numpy()

        # Method to calculate coefficient estimations
        self.method = ""

        # Coefficient Estimations
        self.beta_0_hat = None
        self.beta_1_hat = None

        # Performance metrics
        self.rmse = None
        self.mae = None
        self.r_squared = None

        self.diagnostics = {}

    def fit(self, method: str = None) -> None:
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
            raise ValueError(f"Unknown method '{method}' for SimpleLinearRegression.")

        # Call respective coefficient estimator
        fit_methods[method]()

    def _fit_beta_estimations(self):
        """
        Fits the model using Beta estimations

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
        Fits the model using the Normal Equation

        Theta_hat = ((X.T ⋅ X)^-1) ⋅ X.T ⋅ y
        """

        # Add intercept column
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        theta_hat = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y

        self.beta_0_hat = theta_hat[0]
        self.beta_1_hat = theta_hat[1]

        self.diagnostics = {}  # No dynamics for closed-form method

    def _fit_gradient_descent_batch(self):
        """
        Fits the model using batch gradient descent.
        Minimizes the MSE cost function by iteratively updating θ.

        Gradient: ∂/∂θ J(θ) = 2/m * Xᵀ(Xθ - y)
        """

        # Add intercept column
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]  # shape (m, 2)
        y = self.y.reshape(-1, 1)  # ensure shape (m, 1)
        m = self.x.shape[0]

        eta_0 = 0.1  # Initial learning rate
        decay_rate = 0.01  # Controls how fast learning rate decreases
        n_iterations = 1000
        tolerance = 1e-6

        # Initialize parameters randomly
        theta_hat = np.random.randn(2, 1) * 0.01
        prev_theta = theta_hat.copy()

        cost_history = []

        for iteration in range(n_iterations):
            eta_t = eta_0 / (1 + decay_rate * iteration)  # learning rate decay
            gradients = (2 / m) * X_b.T @ ((X_b @ theta_hat) - y)
            theta_hat -= eta_t * gradients

            cost = np.mean((X_b @ theta_hat - y) ** 2)
            cost_history.append(cost)

            # Convergence check
            if np.linalg.norm(theta_hat - prev_theta) < tolerance:
                print(f"[✔] Converged at iteration {iteration}")
                break
            prev_theta = theta_hat.copy()

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_stochastic(
        self, t0=5, t1=50, n_epochs=50, tolerance=1e-6
    ):
        """
        Fits the model using stochastic gradient descent (SGD) with a
        decaying learning rate.

        Parameters are updated after each training example, using one
        randomly shuffled sample per step.
        """

        # Add intercept column
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        y = self.y.reshape(-1, 1)
        m = self.x.shape[0]

        theta_hat = np.random.randn(2, 1) * 0.01
        t = 0
        cost_history = []

        def learning_schedule(t):
            return t0 / (t + t1)

        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            for i in indices:
                xi = X_b[i : i + 1]
                yi = y[i : i + 1]
                gradients = 2 * xi.T @ ((xi @ theta_hat) - yi)
                eta = learning_schedule(t)
                update = eta * gradients
                theta_hat -= update

                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                # Convergence check
                if np.linalg.norm(update) < tolerance:
                    print(f"[✔] SGD converged at epoch {epoch}, sample {i}, t={t}")
                    break
                t += 1  # count each update step

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def _fit_gradient_descent_mini_batch(
        self,
        batch_size=16,
        eta_0=0.1,
        decay_rate=0.01,
        n_epochs=50,
        tolerance=1e-6,
    ):
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
            tolerance (_type_, optional): Threshold for stopping based on
                parameter stability. Defaults to 1e-6.
        """

        # Add intercept column
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        y = self.y.reshape(-1, 1)
        m = self.x.shape[0]

        theta_hat = np.random.randn(2, 1) * 0.01
        t = 0
        cost_history = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            X_b_shuffled, y_shuffled = X_b[indices], y[indices]

            for i in range(0, m, batch_size):
                X_mini = X_b_shuffled[i : i + batch_size]
                y_mini = y_shuffled[i : i + batch_size]

                gradients = (2 / len(X_mini)) * X_mini.T @ (X_mini @ theta_hat - y_mini)
                eta_t = eta_0 / (1 + decay_rate * t)
                update = eta_t * gradients
                theta_hat -= update

                cost = np.mean((X_b @ theta_hat - y) ** 2)
                cost_history.append(cost)

                if np.linalg.norm(update) < tolerance:
                    print(
                        f"[✔] Mini-batch GD converged at epoch {epoch}, "
                        f"batch {i // batch_size}, t={t}"
                    )
                    break
                t += 1

        # Unpack final parameters
        self.beta_0_hat = theta_hat[0, 0]
        self.beta_1_hat = theta_hat[1, 0]

        self.diagnostics = {"cost_history": cost_history}

    def predict(self, x_new: pd.Series = None):
        if x_new is None:
            x_new = self.x
        return self.beta_0_hat + self.beta_1_hat * x_new

    def residuals(self):
        return self.y - self.predict()

    def fitted(self):
        return self.predict()

    @staticmethod
    def _calculate_rmse(self, y_actual, y_pred):
        return np.sqrt(np.mean((y_actual - y_pred) ** 2))

    @staticmethod
    def _calculate_mae(self, y_actual, y_pred):
        return np.mean(np.abs(y_actual - y_pred))

    @staticmethod
    def _calculate_r_squared(self, y_actual, y_pred):
        return 1 - np.sum((y_actual - y_pred) ** 2) / np.sum(
            (y_actual - np.mean(y_actual)) ** 2
        )

    @staticmethod
    def _format_duration(self, seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.2f}µs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {secs:.2f}s"

    def _coefficient_estimators(
        self, x: pd.Series, y: pd.Series, n: int, methods: str = None
    ):

        coeff_results = []
        methods = methods or SimpleLinearRegression.ALL_METHODS

        for method in methods:
            model = SimpleLinearRegression(x, y)
            try:
                start_time = time.perf_counter()
                model.fit(method=method)
                duration = time.perf_counter() - start_time
                formatted_time = self._format_duration(duration)

                y_hat = model.predict()

                """
                # Performance metrics
                # RMSE: Root Mean Squared Error - Square root of the average of the
                #   squared differences between the predicted values and the actual
                # MAE: Mean Absolute Error - Average absolute difference between the
                #   predicted values and the actual
                # R_squared:
                """
                model.rmse = self._calculate_rmse(y, y_hat)
                model.mae = self._calculate_mae(y, y_hat)
                model.r_squared = self._calculate_r_squared(y, y_hat)

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
                x = pd.Series(np.random.rand(n) * 10)
                y = 3 + 2 * x + np.random.randn(n) * noise

                results.extend(self._coefficient_estimators(x, y, n, methods))

        return pd.DataFrame(results).sort_values(
            ["n_samples", "method", "duration_seconds"]
        )

    def summary(self):
        print(f"Method: {self.method}")
        print(f"Intercept (β₀): {self.beta_0_hat}")
        print(f"Slope (β₁): {self.beta_1_hat}")
        print()

    def benchmark_summary(self):
        df = self.simulate()
        print(df)
        print()
        return df


if __name__ == "__main__":
    x_test = pd.Series([1, 2, 3])
    y_test = pd.Series([1, 2, 3])

    benchmark_model = SimpleLinearRegression(x_test, y_test)
    benchmark_model.benchmark_summary()

    model_beta = SimpleLinearRegression(x_test, y_test)
    model_beta.fit("beta_estimations")
    model_beta.summary()

    model_normal = SimpleLinearRegression(x_test, y_test)
    model_normal.fit("normal_equation")
    model_normal.summary()

    model_gd_batch = SimpleLinearRegression(x_test, y_test)
    model_gd_batch.fit("gradient_descent_batch")
    model_gd_batch.summary()

    model_gd_stochastic = SimpleLinearRegression(x_test, y_test)
    model_gd_stochastic.fit("gradient_descent_stochastic")
    model_gd_stochastic.summary()

    model_gd_mini_batch = SimpleLinearRegression(x_test, y_test)
    model_gd_mini_batch.fit("gradient_descent_mini_batch")
    model_gd_mini_batch.summary()
