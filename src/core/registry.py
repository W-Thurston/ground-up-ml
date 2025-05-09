# src/core/registry.py

from typing import Callable, Dict, List, Optional

# --- Model Registry ---
MODEL_REGISTRY: Dict[str, Dict] = {}


def register_model(
    name: str = None,
    learning_type: str = None,
    task_type: str = None,
    data_shape: str = None,
    model_type: str = None,
    implementation: str = None,
    method: str = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator for registering ML models into the global MODEL_REGISTRY.
    Enables CLI and programmatic filtering based on model traits.

    Args:
        learning_type (str, optional):
            Type of learning paradigm, e.g.,
                "supervised" or "unsupervised".
        task_type (str, optional):
            High-level ML task, e.g.,
                "regression", "classification".
        data_shape (str, optional):
            Data dimensionality, e.g.,
                "univariate" or "multivariate".
        model_type (str, optional):
            Specific family or type of model, e.g.,
                "simple_linear", "ridge", "logistic".
        implementation (str, optional):
            Source of the implementation, e.g.,
                "from_scratch", "sklearn", "pytorch".
        method (str, optional):
            Optimization or estimation method, e.g.,
                "normal_equation", "gradient_descent_batch".

    Returns:
        Callable: The original class unchanged.
    """

    def decorator(cls: Callable) -> Callable:
        key = name or cls.__name__
        MODEL_REGISTRY[key] = {
            "class": cls,
            "learning_type": learning_type,
            "task_type": task_type,
            "data_shape": data_shape,
            "model_type": model_type,
            "implementation": implementation,
            "method": method,
        }
        return cls

    return decorator


def filter_models(
    learning_type: Optional[str] = None,
    task_type: Optional[str] = None,
    data_shape: Optional[str] = None,
    model_type: Optional[str] = None,
    implementation: Optional[str] = None,
    method: Optional[str] = None,
) -> List[str]:
    """
    Return a list of registered model names that match the specified filters.

    This function enables dynamic querying of models based on their metadata
    attributes registered via the `@register_model` decorator. It supports partial
    filtering: any argument can be omitted (None) to include all values for that trait.

    Args:
        learning_type (str, optional):
            Type of learning paradigm, e.g.,
                "supervised" or "unsupervised".
        task_type (str, optional):
            High-level ML task, e.g.,
                "regression", "classification".
        data_shape (str, optional):
            Data dimensionality, e.g.,
                "univariate" or "multivariate".
        model_type (str, optional):
            Specific family or type of model, e.g.,
                "simple_linear", "ridge", "logistic".
        implementation (str, optional):
            Source of the implementation, e.g.,
                "from_scratch", "sklearn", "pytorch".
        method (str, optional):
            Optimization or estimation method, e.g.,
                "normal_equation", "gradient_descent_batch".

    Returns:
        List[str]: A list of model names (strings) that match all provided filters.

    Example:
        >>> filter_models(task_type="regression", data_shape="univariate")
        ['SimpleLinearRegressionFromScratch', 'SimpleLinearRegressionSklearn']

    Notes:
        - Matching is case-sensitive.
        - Filters are combined using logical AND: only models matching *all* specified
          criteria will be returned.
        - If no filters are specified, all registered model names will be returned.
    """
    filters = {
        "learning_type": learning_type,
        "task_type": task_type,
        "data_shape": data_shape,
        "model_type": model_type,
        "implementation": implementation,
        "method": method,
    }

    matching_models = [
        name
        for name, attrs in MODEL_REGISTRY.items()
        if all(filters[k] is None or attrs.get(k) == filters[k] for k in filters)
    ]

    return matching_models


def list_models_for_cli(**kwargs):
    """
    Alias filter_models() for CLI use
    """
    matches = filter_models(**kwargs)
    print(f"Found {len(matches)} model(s):")
    for name in matches:
        print(f" - {name}")
