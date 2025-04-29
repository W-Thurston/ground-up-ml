# main.py

import os
import sys


def print_welcome():
    print("=" * 60)
    print("🚀 Welcome to Ground-Up ML 🚀")
    print("- A scalable benchmarking and model evaluation tool")
    print("=" * 60)
    print()


def validate_project_structure():
    """Ensure the user is running from the correct project root."""
    if not os.path.isdir("src") or not os.path.exists("src/cli/cli.py"):
        print("[!] Error: Please run 'python main.py' from the project root directory.")
        sys.exit(1)


def main():
    print_welcome()
    validate_project_structure()

    # Defer heavy imports until after validation
    from src.cli.cli import main as cli_main

    # Launch the CLI
    cli_main()


if __name__ == "__main__":
    main()
