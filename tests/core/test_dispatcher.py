# tests/test_dispatcher.py
from src.core.dispatcher import run_benchmarks
from src.data.generate_data import generate_singlevariate_synthetic_data_regression


def test_run_benchmarks():
    X, y = generate_singlevariate_synthetic_data_regression()
    pairs = [("from_scratch", "normal_equation")]
    results = run_benchmarks(pairs, X, y)
    assert len(results) == 1
