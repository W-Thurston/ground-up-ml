# src/cli/prompt.py

from src.core.registry import list_registered_models


def prompt_user() -> dict:
    """
    Prompt user interactively for settings.

    Returns:
        settings (dict): Collected settings for the training run.
    """
    settings = {}

    print("=" * 50)
    print("Ground-Up ML Interactive Mode")
    print("=" * 50)

    # Choose mode
    print("\nWhat would you like to do?")
    print("[1] Train only your own model")
    print("[2] Benchmark standard models only")
    print("[3] Compare your model against standard models")

    while True:
        mode_choice = input("Enter your choice (1/2/3): ").strip()
        if mode_choice in ("1", "2", "3"):
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    if mode_choice == "1":
        settings["mode"] = "custom_only"
    elif mode_choice == "2":
        settings["mode"] = "benchmark_only"

        # Regression or Classification
        print("\nWhich task type would you like to benchmark?")
        print("[1] Regression")
        print("[2] Classification")

        while True:
            task_choice = input("Enter your choice (1 or 2): ").strip()
            if task_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2")

        task_type = "regression" if task_choice == "1" else "classification"
        settings["task_type"] = task_type
        all_models = list_registered_models(task_type=task_type)

        print("\nWould you like to benchmark all standard models?")
        print("[1] Yes — use all registered models")
        print("[2] No — I want to select a subset")

        while True:
            all_models_choice = input("Enter your choice (1 or 2): ").strip()
            if all_models_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        if all_models_choice == "1":
            selected_models = all_models
        else:
            print("\nAvailable models:")
            for idx, model in enumerate(all_models, 1):
                print(f"[{idx}] {model}")

            while True:
                selected = input(
                    "Enter numbers separated by commas (e.g., 1,3): "
                ).strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selected.split(",")]
                    selected_models = [
                        all_models[i] for i in indices if 0 <= i < len(all_models)
                    ]
                    if selected_models:
                        break
                except Exception:
                    pass
                print("Invalid selection. Please try again.")

        settings["selected_models"] = selected_models

        # Now: method selection
        print("\nWould you like to use all available training methods?")
        print("[1] Yes — use all methods for each model")
        print("[2] No — I want to select a subset")

        while True:
            all_methods_choice = input("Enter your choice (1 or 2): ").strip()
            if all_methods_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        default_methods = [
            "normal_equation",
            "beta_estimations",
            "gradient_descent_batch",
            "gradient_descent_stochastic",
            "gradient_descent_mini_batch",
        ]

        if all_methods_choice == "1":
            selected_methods = default_methods
        else:
            print("\nAvailable methods:")
            for idx, method in enumerate(default_methods, 1):
                print(f"[{idx}] {method}")

            while True:
                selected = input(
                    "Enter numbers separated by commas (e.g., 1,3,4): "
                ).strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selected.split(",")]
                    selected_methods = [
                        default_methods[i]
                        for i in indices
                        if 0 <= i < len(default_methods)
                    ]
                    if selected_methods:
                        break
                except Exception:
                    pass
                print("Invalid selection. Please try again.")

        settings["selected_methods"] = selected_methods

    elif mode_choice == "3":
        settings["mode"] = "compare"

    return settings
