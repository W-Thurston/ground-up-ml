# src/utils/config.py


def get_config(user_dict, default_dict):
    """
    Merge user overrides with default config, giving precedence to user values.
    """
    return {**default_dict, **(user_dict or {})}
