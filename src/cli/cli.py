# src/cli/cli.py

import argparse

from src.core.dispatcher import run_benchmarks
from src.core.registry import MODEL_REGISTRY
from src.data.generate_data import (
    generate_multivariate_synthetic_data_regression,
    generate_singlevariate_synthetic_data_regression,
)
from src.utils.report import generate_report
from src.visualizations.visualizations import plot_comparison_grid


def load_dataset(
    data_path: str = None,
    data_shape: str = None,
):
    """Load user dataset or generate synthetic."""
    if data_path:
        import pandas as pd

        df = pd.read_csv(data_path)
        X = df["x"]
        y = df["y"]

    else:
        # Generate synthetic if no file provided
        if data_shape == "univariate":
            X, y = generate_singlevariate_synthetic_data_regression(n=1000)
        elif data_shape == "multivariate":
            X, y = generate_multivariate_synthetic_data_regression(n=1000)

    return X, y


def run_experiment(settings: dict):
    """
    Run training, reporting, and visualization for one or more model-method pairs.

    Args:
        settings (dict): _description_
    """
    X, y = load_dataset(
        data_path=settings.get("data_path"), data_shape=settings.get("data_shape")
    )

    if settings.get("mode") == "benchmark_only":
        pairs = [
            (model, method)
            for model in settings["selected_models"]
            for method in settings["selected_methods"]
        ]
    else:
        # fallback for single model-mode
        pairs = [(settings["model_key"], settings["method"])]

    trained_models = run_benchmarks(pairs, X, y)

    benchmark_results = []
    for (model_name, method_name), model in trained_models.items():
        model_meta = MODEL_REGISTRY.get(model_name, {})
        benchmark_results.append(
            {
                "model_name": model_name,
                "implementation": model_meta.get("implementation", "unknown").title(),
                "method_name": method_name.replace("_", " ").title(),
                "mse": model.mse,
                "rmse": model.rmse,
                "mae": model.mae,
                "median_ae": model.median_ae,
                "r_squared": model.r_squared,
                "adjusted_r_squared": model.adjusted_r_squared,
                "training_time_sec": model.duration_seconds,
            }
        )

    generate_report(
        benchmark_results, output_format=settings.get("output_format", "console")
    )

    if settings.get("save_plot"):
        plot_comparison_grid(trained_models, save_path=settings["save_plot"])


def main():
    parser = argparse.ArgumentParser(description="Ground-Up ML CLI")
    parser.add_argument(
        "--config", type=str, help="Path to JSON config file", default=None
    )
    parser.add_argument("--model", type=str, help="Model key (e.g., from_scratch)")
    parser.add_argument("--method", type=str, help="Training method")
    parser.add_argument("--data-path", type=str, help="Path to data CSV")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["console", "json", "markdown"],
        default="console",
    )
    parser.add_argument("--save-plot", type=str, help="Filepath to save plot")

    args = parser.parse_args()

    if args.config:
        import json

        with open(args.config, "r") as f:
            settings = json.load(f)
    elif args.model and args.method:
        settings = {
            "model_key": args.model,
            "method": args.method,
            "data_path": args.data_path,
            "output_format": args.output_format,
            "save_plot": args.save_plot,
        }
    else:
        from src.cli.prompt_user import prompt_user

        settings = prompt_user()

    run_experiment(settings)


if __name__ == "__main__":
    main()
