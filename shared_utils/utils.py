# shared_utils/utils.py


def format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {secs:.2f}s"


def parse_visualization_input(input_str: str) -> list[tuple[str, str]]:
    """
    Parses a CLI-style visualization input string into a list of (model, method) tuples.

    Args:
        input_str (str): Input string, e.g.,
            'from_scratch:normal_equation,pytorch:gradient_descent_batch'

    Returns:
        list[tuple[str, str]]: Parsed list of (model_name, method_name) pairs.
    """
    individual_models = input_str.split(",")
    method_model = [tuple(x.split(":")) for x in individual_models]
    return method_model


if __name__ == "__main__":
    input_example = (
        "from_scratch:normal_equation,pytorch:gradient_descent_batch"
        ",sklearn:gradient_descent_mini_batch"
    )
    parsed = parse_visualization_input(input_example)
    print(parsed)
