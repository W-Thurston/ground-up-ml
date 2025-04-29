# tests/test_metrics.py
import numpy as np

from src.core.metrics.metrics import calculate_mse


def test_calculate_mse():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    assert round(calculate_mse(y_true, y_pred), 3) == 0.375
