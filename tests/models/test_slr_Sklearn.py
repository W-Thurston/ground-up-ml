# tests/test_slr_sklearn.py
from src.data.generate_data import generate_singlevariate_synthetic_data_regression
from src.models.simple_linear_regression.slr_Sklearn import (
    SimpleLinearRegressionSklearn,
)


def test_sklearn_fit_predict():
    x, y = generate_singlevariate_synthetic_data_regression()
    model = SimpleLinearRegressionSklearn(x, y)
    model.fit(x, y, method="normal_equation")
    preds = model.predict()
    assert len(preds) == len(x)
