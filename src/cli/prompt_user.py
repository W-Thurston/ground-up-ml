# src/cli/prompt.py

from src.core.registry import MODEL_REGISTRY, filter_models, list_models_for_cli


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

    print()
    print("*" * 50)
    debug_choice = input("Debug Mode: ").strip()
    debug = True if debug_choice == "y" else False
    print("*" * 50)
    print()

    # Choose mode
    print("\nWhat would you like to do?")
    print("[1] Train only your own model")
    print("[2] Benchmark standard models only")
    print("[3] Compare your model against standard models")

    while True:
        mode_choice = input("Enter your choice (1, 2, or 3): ").strip()
        if mode_choice in ("1", "2", "3"):
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    if mode_choice == "1":
        settings["mode"] = "custom_only"
        # TODO
        ...

    elif mode_choice == "2":
        settings["mode"] = "benchmark_only"

        ############################
        # Supervised or Unsupervised
        print("\nWhich learning type would you like to benchmark?")
        print("[1] Supervised")
        print("[2] Unsupervised")

        while True:
            learning_type_choice = input("Enter your choice (1 or 2): ").strip()
            if learning_type_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2")

        learning_type = "supervised" if learning_type_choice == "1" else "unsupervised"
        settings["learning_type"] = learning_type

        if debug:
            print()
            print("*" * 10, "Debug", "*" * 10)
            list_models_for_cli(learning_type=learning_type)
            print("*" * 20)
            print()
        ############################

        ############################
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

        if debug:
            print()
            print("*" * 10, "Debug", "*" * 10)
            list_models_for_cli(
                learning_type=learning_type,
                task_type=task_type,
            )
            print("*" * 20)
            print()
        ############################

        ############################
        # Univariate or Multivariate
        print("\nWould you like to use Univariate or Multivariate data?")
        print("[1] Univariate")
        print("[2] Multivariate")

        while True:
            data_choice = input("Enter your choice (1 or 2): ").strip()
            if data_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2")

        data_shape_type = "univariate" if data_choice == "1" else "multivariate"
        settings["data_shape"] = data_shape_type
        all_models = filter_models(
            learning_type=learning_type,
            task_type=task_type,
            data_shape=data_shape_type,
        )

        if debug:
            print()
            print("*" * 10, "Debug", "*" * 10)
            list_models_for_cli(
                learning_type=learning_type,
                task_type=task_type,
                data_shape=data_shape_type,
            )
            print("*" * 20)
            print()
        ############################

        ############################
        # Model Type
        print("\nWould you like to train all model types?")
        print("[1] Yes — use all model types")
        print("[2] No — I want to select a subset")
        all_model_types = sorted(
            set(
                MODEL_REGISTRY[model]["model_type"]
                for model in all_models
                if "model_type" in MODEL_REGISTRY[model]
            )
        )

        while True:
            model_type_choice = input("Enter your choice (1 or 2): ").strip()
            if model_type_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        if model_type_choice == "1":
            selected_models = all_models
        else:
            while True:
                print("\nAvailable model types:")
                for idx, model_type in enumerate(all_model_types, 1):
                    print(f"[{idx}] {model_type.title()}")
                print()
                selected = input(
                    "Enter model numbers separated by commas (e.g., 1,3): "
                ).strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selected.split(",")]
                    selected_model_types = {
                        all_model_types[i]
                        for i in indices
                        if 0 <= i < len(all_model_types)
                    }
                    selected_models = [
                        model
                        for model in all_models
                        if MODEL_REGISTRY[model]["model_type"] in selected_model_types
                    ]
                    if selected_models:
                        break
                except Exception:
                    pass
                print("Invalid selection. Please try again.")

        settings["selected_models"] = selected_models
        ############################

        ############################
        # Implementation type
        print("\nWould you like to use all implementation types?")
        print("[1] Yes — use all implementation types")
        print("[2] No — I want to select a subset")
        all_implementations = sorted(
            set(
                MODEL_REGISTRY[model]["implementation"]
                for model in all_models
                if "implementation" in MODEL_REGISTRY[model]
            )
        )

        while True:
            impl_choice = input("Enter your choice (1 or 2): ").strip()
            if impl_choice in ("1", "2"):
                break
            print("Invalid choice. Please enter 1 or 2.")

        if impl_choice == "1":
            selected_models = all_models
        else:
            while True:
                print("\nAvailable implementation types:")
                for idx, impl in enumerate(all_implementations, 1):
                    print(f"[{idx}] {impl.title()}")
                print()
                selected = input(
                    "Enter implementation numbers separated by commas (e.g., 1,3): "
                ).strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selected.split(",")]
                    selected_impls = {
                        all_implementations[i]
                        for i in indices
                        if 0 <= i < len(all_implementations)
                    }
                    selected_models = [
                        model
                        for model in all_models
                        if MODEL_REGISTRY[model]["implementation"] in selected_impls
                    ]
                    if selected_models:
                        break
                except Exception:
                    pass
                print("Invalid selection. Please try again.")

        settings["selected_models"] = selected_models
        ############################

        ############################
        # Method selection
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
        ############################

    elif mode_choice == "3":
        settings["mode"] = "compare"
        # TODO
        ...

    save = input("\nSave these settings to a JSON file? (y/n): ").strip().lower()
    if save == "y":
        import json

        with open("src/config/saved_config.json", "w") as f:
            json.dump(settings, f, indent=2)
        print("Saved to src/config/saved_config.json ✅")

    return settings
