# src/utils/config.py

from inspect import signature


def get_config(user_dict, default_dict):
    """
    Merge user overrides with default config, giving precedence to user values.
    """
    return {**default_dict, **(user_dict or {})}


def safe_kwargs(target, kwargs: dict) -> dict:
    """Filter kwargs to only those accepted by target function/class."""
    valid = signature(target).parameters
    return {k: v for k, v in kwargs.items() if k in valid}
