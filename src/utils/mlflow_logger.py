# src/utils/mlflow_logger.py

import mlflow


def start_run(run_name, tags=None):
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics):
    mlflow.log_metrics(metrics)


def log_artifact(path):
    mlflow.log_artifact(path)
