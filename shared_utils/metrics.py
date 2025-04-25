# shared_utils/metrics.py

import numpy as np


def calculate_mse(y_actual, y_pred):
    return np.mean((y_actual - y_pred) ** 2)


def calculate_rmse(y_actual, y_pred):
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))


def calculate_mae(y_actual, y_pred):
    return np.mean(np.abs(y_actual - y_pred))


def calculate_r_squared(y_actual, y_pred):
    return 1 - np.sum((y_actual - y_pred) ** 2) / np.sum(
        (y_actual - np.mean(y_actual)) ** 2
    )
