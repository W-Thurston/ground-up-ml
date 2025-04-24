# Changelog

## [v0.3.0] — _Full Gradient Descent Implementation + Linting Cleanup_

**Release Date:** 2025-04-23

### ✨ Features

- ✅ **Implemented all gradient descent variants**:
  - `gradient_descent_batch` with decaying learning rate and convergence check
  - `gradient_descent_stochastic` with per-sample updates and shuffling per epoch
  - `gradient_descent_mini_batch` with tunable batch size and dynamic decay
- ✅ Added consistent **cost tracking (`cost_history`)** across all gradient descent methods to support robust visual diagnostics in future releases
- ✅ Each `_fit_*` method now standardizes outputs via `self.diagnostics` for plotting and analysis

### 🧼 Code & Dev Tooling

- ✅ Resolved conflicts between `black` and `flake8` (E203) by configuring a `.flake8` file
- ✅ Updated `pyproject.toml` and unified project style settings across all tools
- ✅ Pre-commit hooks now run cleanly with proper config resolution

### 🧪 Internal Enhancements

- Refactored logic and docstrings for readability, consistency, and extensibility
- Updated benchmark runner to consistently handle all methods via `ALL_METHODS`

## [v0.2.0] — _Adds Normal Equation and Benchmarking Enhancements_

**Release Date:** 2025-04-23

### ✨ Features

- Implemented `normal_equation` method for coefficient estimation using the matrix form of linear regression.
- Integrated both `beta_estimations` and `normal_equation` into the `simulate()` benchmarking function.
- Added detailed benchmarking metrics:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
  - Execution time in raw seconds and human-readable format
- Refactored summary and benchmarking methods for easier extension and future logging support.

### 🧪 Testing & Usability

- Added `__main__` test block to print method summaries and benchmark results for sample data.
- Included support for simulated or user-supplied datasets.

---

## [v0.1.0] — _Initial SLR with Beta Estimation_

**Release Date:** 2025-04-22

### ✨ Features

- Implemented simple linear regression using the ISLR-style `beta_estimations` method.
- Added:
  - `fit()`, `predict()`, `residuals()`, and `fitted()` methods
  - RMSE, MAE, and R² metric calculations
  - Benchmarking framework (`simulate()`) with support for timing and varying sample sizes
- Included timing formatter for readable benchmarking logs.
