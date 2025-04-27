import numpy as np
import pandas as pd

from shared_utils.dispatcher import run_benchmarks
from shared_utils.report import generate_report

# Imports from your project
from shared_utils.utils import parse_visualization_input
from shared_utils.visualizations import plot_comparison_grid

# 1. Create small synthetic dataset
np.random.seed(42)
X = pd.Series(np.random.rand(1000))
y = 3.0 + 2.0 * X + np.random.randn(1000) * 0.2  # y = 3 + 2x + noise

# 2. Define benchmark plan (pick 1-2 methods from each model)
input_str = (
    "from_scratch:beta_estimations,from_scratch:normal_equation,"
    "from_scratch:gradient_descent_batch,from_scratch:gradient_descent_stochastic,"
    "from_scratch:gradient_descent_mini_batch,sklearn:normal_equation,"
    "sklearn:gradient_descent_batch,sklearn:gradient_descent_stochastic,"
    "sklearn:gradient_descent_mini_batch,pytorch:beta_estimations,"
    "pytorch:normal_equation,pytorch:gradient_descent_batch,"
    "pytorch:gradient_descent_stochastic,pytorch:gradient_descent_mini_batch"
)
parsed_pairs = parse_visualization_input(input_str)

# 3. Run benchmarks
trained_models = run_benchmarks(parsed_pairs, X, y)

# 4. Print benchmark results table
benchmark_results = []
for (model_name, method_name), model in trained_models.items():
    benchmark_results.append(
        {
            "model_name": model_name,
            "method_name": method_name,
            "mse": model.mse,
            "rmse": model.rmse,
            "mae": model.mae,
            "median_ae": model.median_ae,
            "r_squared": model.r_squared,
            "adjusted_r_squared": model.adjusted_r_squared,
            "beta_0": model.beta_0_hat,
            "beta_1": model.beta_1_hat,
            "training_time_sec": model.duration_seconds,
        }
    )

generate_report(benchmark_results, output_format="console")

# 5. Plot cost histories
plot_comparison_grid(
    trained_models,
    metrics=[
        "regression_line",
        "cost_history",
        "training_time",
        "mse",
        "rmse",
        "mae",
        "median_ae",
        "r_squared",
        "adjusted_r_squared",
    ],
    save_path="benchmark_grid.png",
)
