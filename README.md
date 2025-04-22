# Ground-up ML Reboot

**A hands-on machine learning reboot from first principles to production-ready code.**

This project is a structured, from-scratch journey through core machine learning topics. Each mini-project includes multiple implementations to help build understanding and transition from intuition to industrial-strength workflows.

## 🔍 What’s Inside

Each mini-project demonstrates the same ML concept in three ways:

- ✅ **Pure Python/Numpy** — for building foundational intuition.
- ⚙️ **Scikit-learn** — for standard modeling pipelines.
- 🔥 **PyTorch** — for scalable, deep learning-friendly extensions.

Projects include:

```
mini_projects/
├── 01_simple_linear_regression/
│   ├── from_scratch/     # Linear regression with NumPy
│   ├── sklearn_impl/     # Using scikit-learn
│   ├── pytorch_impl/     # Using PyTorch
│   └── notebook.ipynb    # Visual + code comparison
...
```

Shared folders for `data/` and `shared_utils/` make it easy to reuse common loaders, metrics, and plotting functions across implementations.

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

Then dive into any `notebook.ipynb` in `mini_projects/` to see side-by-side comparisons and results.

---

## 🧠 Who This Is For

- Developers rebuilding their ML foundation with more rigor.
- Educators looking for step-by-step concept demos.
- Practitioners comparing frameworks for learning or performance.

---

## 📚 License

MIT License. Use, remix, or contribute!

---

_This is part of an educational blog series, **Ground-up ML Reboot** — teaching machines (and ourselves) how to learn from scratch._
