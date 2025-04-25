# shared_utils/visualizations.py

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_diagnostics(
    model: object,
    x: Optional[Union[pd.Series, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Dispatches to appropriate plotting function based on model class name.

    Args:
        model (object): A trained model instance.
        x (Optional): Feature values used to fit the model, if not stored in the model.
        y (Optional): Target values used to fit the model, if not stored in the model.
        title (Optional): Title to display on the regression plot.

    Returns:
        plt.Figure: The matplotlib figure object generated.

    Raises:
        ValueError: If the model type is unsupported.
    """
    VISUALIZATION_DISPATCH = {
        "SimpleLinearRegressionFromScratch": _plot_simple_linear_regression,
        "SimpleLinearRegressionSklearn": _plot_simple_linear_regression,
        "SimpleLinearRegressionPyTorch": _plot_simple_linear_regression,
        # Add future model types here
    }

    model_type = type(model).__name__

    if model_type not in VISUALIZATION_DISPATCH:
        raise ValueError(
            f"Model type: {model_type}, not supported for visualization yet."
        )

    return VISUALIZATION_DISPATCH[model_type](model, x, y, title)


def _plot_simple_linear_regression(
    model: object,
    x: Optional[Union[pd.Series, np.ndarray]] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot diagnostics for a Simple Linear Regression model:
        - Regression line and data points
        - Residual Plot

    Args:
        model (object): The SimpleLinearRegression model instance.
        x (Optional): Features used for fitting. Defaults to model.x if None.
        y (Optional): Targets used for fitting. Defaults to model.y if None.
        title (Optional): Optional plot title.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    if x is None:
        x = model.x
    if y is None:
        y = model.y

    y_pred = model.predict(x)
    residuals = y - y_pred

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Regression line
    ax[0].scatter(x, y, label="Actual", color="blue")
    ax[0].plot(x, y_pred, label="Predicted", color="red")
    ax[0].set_title(title or "Simple Linear Regression Fit")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()

    # Right: Residuals
    ax[1].scatter(x, residuals, label="Residuals", color="purple")
    ax[1].axhline(0, linestyle="--", color="black", linewidth=1)
    ax[1].set_title("Residual Plot")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Residuals")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    return fig
