import numpy as np
import pandas as pd


def generate_synthetic_data(n=100, noise=0.2, seed=42):
    np.random.seed(seed)
    x = pd.Series(np.random.rand(n))
    y = 4 + 3 * x + np.random.randn(n) * noise
    return x, y
