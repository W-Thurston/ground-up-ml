# slr.py placeholder

import time

import numpy as np
import pandas as pd

# from shared_utils.mlflow_logger import log_metrics, log_params, start_run


class SimpleLinearRegression:
    ALL_METHODS = [
        "beta_estimations",
        "normal_equation",
    ]
    #     "gradient_descent_batch",
    #     "gradient_descent_stochastic",
    #     "gradient_descent_mini_batch"
    # ]

    def __init__(self, x: pd.Series, y: pd.Series):
        self.x = x.to_numpy()
        self.y = y.to_numpy()
        self.method = ""
        self.beta_0_hat = None
        self.beta_1_hat = None

        # Performance metrics
        self.rmse = None
        self.mae = None
        self.r_squared = None

    def fit(self, method: str = None):
        self.method = method

        if method == "beta_estimations":
            self._fit_beta_estimations()
        elif self.method == "normal_equation":
            self._fit_normal_equation()
        elif self.method == "gradient_descent_batch":
            self._fit_gradient_descent_batch()
        elif self.method == "gradient_descent_stochastic":
            self._fit_gradient_descent_stochastic()
        elif self.method == "gradient_descent_mini_batch":
            self._fit_gradient_descent_mini_batch()
        else:
            raise ValueError(
                f"Unknown method '{self.method}' for SimpleLinearRegression."
            )

    def _fit_beta_estimations(self):
        """
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

    def _fit_normal_equation(self):
        """
        Theta_hat = ((X.T ⋅ X)^-1) ⋅ X.T ⋅ y
        """

        X_b = np.c_[np.ones((self.x.shape[0],1)), self.x] # add x0 = 1 to each observation
        theta_hat = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y

        self.beta_0_hat = theta_hat[0]
        self.beta_1_hat = theta_hat[1]

    def _fit_gradient_descent_batch(self): ...
    def _fit_gradient_descent_stochastic(self): ...
    def _fit_gradient_descent_mini_batch(self): ...

    def predict(self, x_new: pd.Series = None):
        if x_new is None:
            x_new = self.x
        return self.beta_0_hat + self.beta_1_hat * x_new

    def residuals(self):
        return self.y - self.predict()

    def fitted(self):
        return self.predict()

    def _calculate_rmse(self, y_actual, y_pred):
        return np.sqrt(np.mean((y_actual - y_pred) ** 2))

    def _calculate_mae(self, y_actual, y_pred):
        return np.mean(np.abs(y_actual - y_pred))

    def _calculate_r_squared(self, y_actual, y_pred):
        return 1 - np.sum((y_actual - y_pred) ** 2) / np.sum(
            (y_actual - np.mean(y_actual)) ** 2
        )

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
