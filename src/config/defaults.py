# src/config/defaults.py

#############################
# Gradient Descent Defaults #
#############################
# General training hyperparameters
DEFAULT_TRAINING_KWARGS_FROM_SCRATCH = {
    "max_epochs": 1000,
    "convergence_tol": 1e-6,
    "theta_init_scale": 0.01,
    "verbose": False,
    "batch_size": 32,
}

DEFAULT_TRAINING_KWARGS_PYTORCH = {
    "max_epochs": 100,
    "batch_size": 32,
    "lr": 0.05,
}

DEFAULT_TRAINING_KWARGS_SKLEARN = {
    "max_epochs": 50,
    "eta0": 0.01,
    "power_t": 0.25,
    "loss": "squared_error",
    "learning_rate": "invscaling",
    "random_state": 42,
    "tol": None,
    "shuffle": False,
    "warm_start": True,
    "batch_size": 32,
}

# Learning rate schedule presets
DEFAULT_SCHEDULE_NAME = "time_decay"
DEFAULT_SCHEDULE_KWARGS = {
    "constant": {
        "eta": 0.1,
    },
    "reciprocal": {
        "t0": 1,
        "t1": 100,
    },
    "inverse_sqrt": {
        "eta0": 0.1,
    },
    "time_decay": {
        "eta0": 0.1,
        "decay": 0.01,
    },
    "exponential_decay": {
        "eta0": 0.1,
        "decay": 0.01,
    },
    "step_decay": {
        "eta0": 0.1,
        "drop": 0.5,
        "epochs_drop": 100,
    },
    "cosine_annealing": {
        "eta_min": 0.001,
        "eta_max": 0.1,
        "T_max": 100,
    },
}
