# slr.py placeholder
from shared_utils.mlflow_logger import start_run, log_params, log_metrics

with start_run("SLR_from_scratch"):
    log_params({"learning_rate": 0.01, "epochs": 100})
    log_metrics({"mse": 12.7, "r2": 0.91})