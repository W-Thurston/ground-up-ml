# Changelog

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
