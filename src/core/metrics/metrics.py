# src/core/metrics/metrics.py

import numpy as np


def calculate_mse(y_actual, y_pred):
    """
    Emphasizes larger errors (squares them)
    """
    return np.mean((y_actual - y_pred) ** 2)


def calculate_rmse(y_actual, y_pred):
    """
    Measures average magnitude of error (in same units as y)
    """
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))


def calculate_mae(y_actual, y_pred):
    """
    Measures average absolute error, less sensitive to outliers
    """
    return np.mean(np.abs(y_actual - y_pred))


def calculate_median_ae(y_actual, y_pred):
    """
    Robust to outliers; median magnitude of error
    """
    return np.median(np.abs(y_actual - y_pred))


def calculate_r_squared(y_actual, y_pred):
    """
    Measures proportion of variance explained
    """
    return 1 - np.sum((y_actual - y_pred) ** 2) / np.sum(
        (y_actual - np.mean(y_actual)) ** 2
    )


def calculate_adjusted_r_squated(y_actual, y_pred, p: int = 1):
    """
    Adjusts R² for number of predictors. Important for multivariate,
        but nice to include now.
    """
    n = len(y_actual)
    return 1 - (1 - calculate_r_squared(y_actual, y_pred)) * ((n - 1) / (n - p - 1))
