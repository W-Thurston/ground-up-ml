import numpy as np
import pandas as pd


def generate_singlevariate_synthetic_data_regression(n=100, noise=0.2, seed=42):
    np.random.seed(seed)
    x = pd.Series(np.random.rand(n))

    # True model: y = 4 + 3*x + ε
    eps = np.random.randn(n) * noise
    y = 4 + 3 * x + eps
    return x, y


def generate_multivariate_synthetic_data_regression(n=100, noise=0.2, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame(
        {
            "feature_1": 2 * np.random.rand(n),
            "feature_2": 3 * np.random.rand(n),
        }
    )
    # True model: y = 4 + 3*x1 + 2*x2 + ε
    eps = np.random.randn(n) * noise
    y = 4 + 3 * X["feature_1"] + 2 * X["feature_2"] + eps
    return X, y
