# src/core/dispatcher.py

import time

import pandas as pd
from tqdm import tqdm

from src.core.registry import MODEL_REGISTRY

# Import model files to trigger registration
from src.models.simple_linear_regression import (  # noqa: F401
    slr_FromScratch,
    slr_Pytorch,
    slr_Sklearn,
)

# import src.models.simple_linear_regression.  # noqa: F401
# import src.models.simple_linear_regression.  # noqa: F401


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

    with tqdm(total=len(pairs), ncols=100) as pbar:
        for model_name, method_name in pairs:
            pbar.set_description(f"Training {model_name}:{method_name}")

            ModelClass = MODEL_REGISTRY[model_name]["class"]
            model = ModelClass(X, y)

            start = time.time()
            model.fit(X, y, method_name)
            end = time.time()

            model.duration_seconds = end - start
            model.method = method_name

            model.calculate_metrics()

            results[(model_name, method_name)] = model

            pbar.update(1)
    print()

    return results
