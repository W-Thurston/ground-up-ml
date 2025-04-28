# src/utils/report.py

from tabulate import tabulate

from utils.utils import format_duration


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
    table = [
        [
            res["model_name"],
            res["method_name"],
            f"{res['beta_0']:.4f}",
            f"{res['beta_1']:.4f}",
            f"{res['rmse']:.4f}",
            f"{res['mae']:.4f}",
            f"{res['mse']:.4f}",
            f"{res['median_ae']:.4f}",
            f"{res['r_squared']:.4f}",
            f"{res['adjusted_r_squared']:.4f}",
            f"{format_duration(res['training_time_sec'])}",
        ]
        for res in results
    ]
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
