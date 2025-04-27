# Ground-up ML Reboot

**A hands-on machine learning reboot — from first principles to production-ready benchmarking and diagnostics.**

This project is a structured, from-scratch journey through core machine learning topics.
Each mini-project includes multiple implementations to help build understanding and transition from intuition to industrial-strength workflows.

---

## 🔍 What’s Inside

Each mini-project demonstrates the same ML concept in three ways:

- ✅ **Pure Python/Numpy** — for building foundational intuition.
- ⚙️ **Scikit-learn** — for standard modeling pipelines.
- 🔥 **PyTorch** — for scalable, deep learning-friendly extensions.

Projects include:

```
src/
├── simple_linear_regression/
│   ├── from_scratch/     # Linear regression built manually with NumPy
│   ├── sklearn_impl/     # Linear regression using scikit-learn
│   ├── pytorch_impl/     # Linear regression using PyTorch
│   └── notebook.ipynb    # Visual + code comparison
...
```

Shared folders for `data/` and `shared_utils/` allow easy reuse of common loaders, metrics, and plotting functions across implementations.

---

## 🚀 Current Status (v1.0.0)

**Simple Linear Regression Fully Implemented:**

- Five training methods supported:
  - Beta estimations
  - Normal equation
  - Batch gradient descent
  - Stochastic gradient descent
  - Mini-batch gradient descent
- Benchmarking runner to compare models/methods dynamically
- Cost history tracking and convergence visualizations
- Unified APIs for easy extension
- Extensible dispatch system for future models

---

## 🛠 Key Features

- 📈 Benchmark model performance (RMSE, MAE, R²) across implementations
- 🧠 Understand optimizer behaviors through loss curves
- 🔀 Flexibly specify any model/method combination to compare
- 🏗️ Access trained models programmatically for further diagnostics
- 📚 Built for education, research, and real-world ML pipeline foundations

---

## 📚 Vision for Future Work

- Multivariate Linear Regression benchmarking
- Logistic Regression with gradient descent solvers
- Model assumption validation module integration (homoscedasticity, normality, etc.)
- Batch size, learning rate, and epoch tuning sweeps

---

## 🧠 Who This Is For

- Developers rebuilding their ML foundation with greater rigor
- Educators seeking clean, step-by-step ML concept demos
- Practitioners needing side-by-side framework benchmarks
- Researchers profiling computational trade-offs between frameworks

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/w-thurston/ground-up-ml.git
cd ground-up-ml

# Set up environment (using poetry or pip)
pip install -r requirements.txt
# or
poetry install
```

Then dive into any `notebook.ipynb` in `src/` to see side-by-side comparisons and results!

---

## 📚 License

MIT License. Use, remix, or contribute!

---

_This project is part of an educational blog series, **Ground-up ML Reboot** — teaching machines (and ourselves) how to learn from scratch._
