# src/visualizations/visualizations.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

METHOD_COLOR_MAP = {
    "normal_equation": "#4C72B0",  # muted blue
    "beta_estimations": "#55A868",  # muted green
    "gradient_descent_batch": "#C44E52",  # muted red
    "gradient_descent_stochastic": "#8172B2",  # muted purple
    "gradient_descent_mini_batch": "#CCB974",  # muted yellow-brown
}


def plot_comparison_grid(
    models: dict, metrics: list[str] = ["cost_history"], save_path: str = None
) -> None:
    """
    Creates a multi-panel comparison grid of diagnostic plots.

    Args:
        models (dict): Keys = (model_name, method_name),
            values = trained model instances.
        metrics (list[str], optional): Which metrics to plot per row.
        save_path (str, optional): If provided, save the figure to this path.
    """
    model_types = ["from_scratch", "sklearn", "pytorch"]
    num_rows = len(metrics)
    num_cols = len(model_types)

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 4.5 * num_rows)
    )
    sns.set_theme(style="darkgrid")

    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # Precompute y-axis limits
    y_axis_limits = {}
    for metric in metrics:
        if metric in [
            "rmse",
            "mae",
            "r_squared",
            "mse",
            "median_ae",
            "adjusted_r_squared",
            "explained_variance",
            "training_time",
        ]:
            metric_values = []
            for (model_name, method_name), model in models.items():
                value = (
                    getattr(model, metric, None)
                    if metric != "training_time"
                    else getattr(model, "duration_seconds", None)
                )
                if value is not None:
                    metric_values.append(value)
            if metric_values:
                min_val, max_val = min(metric_values), max(metric_values)
                if abs(max_val - min_val) < 1e-3:
                    buffer = 0.001
                else:
                    buffer = 0.1 * (max_val - min_val)
                y_axis_limits[metric] = (min_val - buffer, max_val + buffer)

    # Plotting
    for row_idx, metric in enumerate(metrics):
        handles_labels = []  # Collect legend entries per row

        for col_idx, model_type in enumerate(model_types):
            ax = axes[row_idx][col_idx]
            subset = {k: v for k, v in models.items() if k[0] == model_type}

            any_data = False
            plotted_actual = False

            for (m_name, method), model in subset.items():
                color = METHOD_COLOR_MAP.get(method, None)

                if metric == "regression_line":
                    x = model.x if hasattr(model, "x") else None
                    y = model.y if hasattr(model, "y") else None
                    if x is not None and y is not None:
                        x_sorted = np.sort(x)
                        y_pred = model.predict(np.sort(x))
                        if not plotted_actual:
                            _ = ax.scatter(x, y, label="Actual", alpha=0.5)
                            plotted_actual = True
                        (line,) = ax.plot(x_sorted, y_pred, label=method, color=color)
                        handles_labels.append((line, method))
                        any_data = True

                elif metric == "cost_history":
                    history = model.diagnostics.get("cost_history")
                    if history:
                        (line,) = ax.plot(
                            range(len(history)),
                            history,
                            label=method,
                            color=color,
                        )
                        any_data = True

                elif metric in [
                    "rmse",
                    "mae",
                    "r_squared",
                    "mse",
                    "median_ae",
                    "adjusted_r_squared",
                    "explained_variance",
                    "training_time",
                ]:
                    metric_value = (
                        getattr(model, metric, None)
                        if metric != "training_time"
                        else getattr(model, "duration_seconds", None)
                    )
                    if metric_value is not None:
                        ax.bar(method, metric_value, color=color)
                        any_data = True

            # Handle no data gracefully
            if not any_data:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {metric.replace('_', ' ').title()}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # Shared y-axis scaling
            if metric in y_axis_limits:
                ax.set_ylim(y_axis_limits[metric])

            ax.set_title(
                f"{model_type.replace('_', '-').title()}"
                f" - {metric.replace('_', ' ').title()}"
            )
            ax.set_xlabel("Epochs" if metric == "cost_history" else "")
            ax.set_ylabel(
                "Loss (MSE)"
                if metric == "cost_history"
                else metric.replace("_", " ").title()
            )

            if metric in [
                "rmse",
                "mae",
                "r_squared",
                "mse",
                "median_ae",
                "adjusted_r_squared",
                "explained_variance",
                "training_time",
            ]:
                plt.sca(ax)
                plt.xticks(rotation=45, ha="right")

        # Plot shared legend once per row if needed
        if metric in ["regression_line"] and handles_labels:
            label_to_handle = {}
            for handle, label in handles_labels:
                if label not in label_to_handle:
                    label_to_handle[label] = handle

            handles = list(label_to_handle.values())
            labels = list(label_to_handle.keys())

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02 - row_idx * 0.22),
                ncol=5,
                fontsize=12,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
