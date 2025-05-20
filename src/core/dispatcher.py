# src/core/dispatcher.py

import time

import pandas as pd
from tqdm import tqdm

from src.config.defaults import DEFAULT_SCHEDULE_NAME
from src.core.registry import MODEL_REGISTRY
from src.models.multivariate_linear_regression import (  # noqa: F401;
    mlr_FromScratch,
    mlr_Sklearn,
)

# Import model files to trigger registration
from src.models.simple_linear_regression import (  # noqa: F401
    slr_FromScratch,
    slr_Pytorch,
    slr_Sklearn,
)


def run_benchmarks(pairs: list[tuple[str, str]], X: pd.Series, y: pd.Series) -> dict:
    """
    Runs benchmarks based on user-specified (model, method) pairs.

    Args:
        pairs (list[tuple[str, str]]): List of (model_name, method_name) pairs
        X (pd.Series): Feature data
        y (pd.Series): Target data

    Returns:
        dict: Mapping of (model_name, method_name) to trained model instance
    """
    results = {}

    filtered_pairs = [
        (model, method_name)
        for model, method_name in pairs
        if method_name in getattr(MODEL_REGISTRY[model]["class"], "ALL_METHODS", [])
    ]
    pairs = filtered_pairs

    with tqdm(total=len(pairs), ncols=100) as pbar:
        for model_name, method_name in pairs:

            ModelClass = MODEL_REGISTRY[model_name]["class"]
            model = ModelClass(X, y)

            pbar.set_description(f"Training {model_name}:{method_name}")

            # Detect if the model uses gradient descent
            uses_gd = "gradient_descent" in method_name

            # Define default safe kwargs
            fit_args = {"method": method_name}
            if uses_gd:
                fit_args.update(
                    {
                        "schedule": DEFAULT_SCHEDULE_NAME,
                        "schedule_kwargs": {},
                        "training_kwargs": {},
                    }
                )
            try:
                start_time = time.perf_counter()
                model.fit(**fit_args)
                model.duration_seconds = time.perf_counter() - start_time
            except TypeError as e:
                print(f"[!] Skipping {model_name} with method {method_name}: {e}")
                continue

            model.method = method_name

            model.calculate_metrics()

            results[(model_name, method_name)] = model

            pbar.update(1)
    print()

    return results
