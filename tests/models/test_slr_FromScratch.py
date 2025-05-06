# tests/test_slr_from_scratch.py

from src.data.generate_data import generate_singlevariate_synthetic_data_regression
from src.models.simple_linear_regression.slr_FromScratch import (
    SimpleLinearRegressionFromScratch,
)


def test_from_scratch_fit_predict():
    x, y = generate_singlevariate_synthetic_data_regression()
    model = SimpleLinearRegressionFromScratch(x, y)
    model.fit(x, y, method="normal_equation")
    preds = model.predict()
    assert len(preds) == len(x)
