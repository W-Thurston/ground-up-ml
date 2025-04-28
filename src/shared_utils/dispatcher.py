# shared_utils/dispatcher.py

import time

import pandas as pd
from src.simple_linear_regression.from_scratch.slr_FromScratch import (
    SimpleLinearRegressionFromScratch,
)
from src.simple_linear_regression.pytorch_impl.slr_Pytorch import (
    SimpleLinearRegressionPyTorch,
)
from src.simple_linear_regression.sklearn_impl.slr_Sklearn import (
    SimpleLinearRegressionSklearn,
)
from tqdm import tqdm

MODEL_DISPATCH = {
    "from_scratch": SimpleLinearRegressionFromScratch,
    "sklearn": SimpleLinearRegressionSklearn,
    "pytorch": SimpleLinearRegressionPyTorch,
}


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

            ModelClass = MODEL_DISPATCH[model_name]
            model = ModelClass(X, y)

            start = time.time()
            model.fit(method_name)
            end = time.time()

            model.duration_seconds = end - start
            model.method = method_name

            model.calculate_metrics()

            results[(model_name, method_name)] = model

            pbar.update(1)
    print()

    return results
