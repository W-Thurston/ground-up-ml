# src/utils/learning_rate.py

import math

from src.config.defaults import DEFAULT_SCHEDULE_KWARGS
from src.utils.config import get_config


def get_learning_rate_schedule(name="constant", **kwargs):
    """
    Returns a learning rate schedule function with fallback to default hyperparameters.

    Supported Schedules:
        - constant
        - reciprocal         (1 / (t + t1))
        - inverse_sqrt       (1 / sqrt(t + 1))
        - time_decay         (eta0 / (1 + decay * t))
        - exponential_decay  (eta0 * exp(-decay * t))
        - step_decay         (eta0 * drop ** floor(t / epochs_drop))
        - cosine_annealing   (cosine-warm restarts style annealing)
    """
    name = name.lower()
    config = get_config(kwargs, DEFAULT_SCHEDULE_KWARGS.get(name, {}))

    if name == "constant":
        eta = config["eta"]
        return lambda t: eta

    elif name == "reciprocal":
        t0 = config["t0"]
        t1 = config["t1"]
        return lambda t: t0 / (t + t1)

    elif name == "inverse_sqrt":
        eta0 = config["eta0"]
        return lambda t: eta0 / (t**0.5 + 1)

    elif name == "time_decay":
        eta0 = config["eta0"]
        decay = config["decay"]
        return lambda t: eta0 / (1 + decay * t)

    elif name == "exponential_decay":
        eta0 = config["eta0"]
        decay = config["decay"]
        return lambda t: eta0 * math.exp(-decay * t)

    elif name == "step_decay":
        eta0 = config["eta0"]
        drop = config["drop"]
        epochs_drop = config["epochs_drop"]
        return lambda t: eta0 * (drop ** (t // epochs_drop))

    elif name == "cosine_annealing":
        eta_min = config["eta_min"]
        eta_max = config["eta_max"]
        T_max = config["T_max"]
        return lambda t: eta_min + 0.5 * (eta_max - eta_min) * (
            1 + math.cos(math.pi * t / T_max)
        )

    else:
        raise ValueError(f"Unknown learning rate schedule: '{name}'")
