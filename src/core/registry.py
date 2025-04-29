# src/core/registry.py

from typing import Callable, Dict

# --- Model Registry ---
MODEL_REGISTRY: Dict[str, Dict] = {}


def register_model(
    name: str, task_type: str = "regression", group: str = None
) -> Callable[[Callable], Callable]:
    """
    Decorator to register a model class under a given name.

    Args:
        name (str): Model registry key.

    Returns:
        Callable: The original class unchanged.
    """

    def decorator(cls: Callable) -> Callable:
        MODEL_REGISTRY[name] = {"class": cls, "task_type": task_type, "group": group}
        return cls

    return decorator


def list_registered_models(task_type: str = None, return_full: bool = False):
    """
    List registered model names or full metadata.

    Args:
        task_type (str, optional): Filter by 'regression' or 'classification'.
        return_full (bool, optional): If True, return full metadata dicts.

    Returns:
        list: List of model names (default) or list of (name, metadata) tuples.
    """
    if return_full:
        if task_type:
            return [
                (name, meta)
                for name, meta in MODEL_REGISTRY.items()
                if meta["task_type"] == task_type
            ]
        return list(MODEL_REGISTRY.items())

    else:
        if task_type:
            return [
                name
                for name, meta in MODEL_REGISTRY.items()
                if meta["task_type"] == task_type
            ]
        return list(MODEL_REGISTRY.keys())


def list_model_groups(task_type: str = None) -> list[str]:
    """List all model groups, optionally filtered by task_type."""
    groups = set()
    for meta in MODEL_REGISTRY.values():
        if task_type is None or meta["task_type"] == task_type:
            if meta["group"]:
                groups.add(meta["group"])
    return sorted(groups)


def get_models_by_group(group_name: str) -> list[str]:
    """Return list of model keys in the specified group."""
    return [
        name for name, meta in MODEL_REGISTRY.items() if meta.get("group") == group_name
    ]
