# tests/test_slr_pytorch.py
from src.data.generate_data import generate_singlevariate_synthetic_data_regression
from src.models.simple_linear_regression.slr_Pytorch import (
    SimpleLinearRegressionPyTorch,
)


def test_pytorch_fit_predict():
    x, y = generate_singlevariate_synthetic_data_regression()
    model = SimpleLinearRegressionPyTorch(x, y)
    model.fit(x, y, method="normal_equation")
    preds = model.predict()
    assert len(preds) == len(x)
