# src/utils/report.py

from tabulate import tabulate

from src.utils.utils import format_duration


def generate_report(results: list[dict], output_format: str = "console") -> None:
    """
    Generate benchmark results report.

    Args:
        results (list of dicts): Each dict must have keys:
            'model_name', 'method_name', 'rmse', 'mae', 'r_squared', 'beta_0', 'beta_1'
        output_format (str): 'console' (default), 'json' (TODO), or 'markdown' (TODO)
    """
    if output_format == "console":
        console_report(results)
    elif output_format == "json":
        export_to_json(results)
    elif output_format == "markdown":
        export_to_markdown(results)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def console_report(results: list[dict]) -> None:
    """
    Prints benchmark results to the console using tabulate.

    Args:
        results (list of dicts): Each dict must have required benchmark keys.
    """
    headers = [
        "Model",
        "Method",
        "Beta₀",
        "Beta₁",
        "RMSE",
        "MAE",
        "MSE",
        "Median AE",
        "R²",
        "Adjusted R²",
        "Training Time (s)",
    ]
    table = []
    for res in results:
        table.append(
            [
                res.get("model_name", ""),
                res.get("method_name", ""),
                (
                    f"{res.get('beta_0', float('nan')):.4f}"
                    if res.get("beta_0") is not None
                    else "N/A"
                ),
                (
                    f"{res.get('beta_1', float('nan')):.4f}"
                    if res.get("beta_1") is not None
                    else "N/A"
                ),
                f"{res.get('rmse', float('nan')):.4f}",
                f"{res.get('mae', float('nan')):.4f}",
                f"{res.get('mse', float('nan')):.4f}",
                f"{res.get('median_ae', float('nan')):.4f}",
                f"{res.get('r_squared', float('nan')):.4f}",
                f"{res.get('adjusted_r_squared', float('nan')):.4f}",
                format_duration(res.get("training_time_sec", 0.0)),
            ]
        )
    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="github",
            colalign=(
                "left",
                "left",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
            ),
        )
    )


def export_to_json(results: list[dict]) -> None:
    """
    TODO: Implement JSON export.
    """
    pass


def export_to_markdown(results: list[dict]) -> None:
    """
    TODO: Implement Markdown export.
    """
    pass
