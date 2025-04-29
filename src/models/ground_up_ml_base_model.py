"""Base class for all models in the Ground-Up ML project.

Defines the minimal interface that all models must implement
to integrate cleanly with benchmarking, metrics, and visualization systems.
"""

from abc import ABC, abstractmethod


class GroundUpMLBaseModel(ABC):
    """Abstract base class for machine learning models in Ground-Up ML."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model on features X and targets y.

        Parameters:
            X: Feature matrix or tensor
            y: Target vector or tensor
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions from input features.

        Parameters:
            X: Feature matrix or tensor

        Returns:
            y_pred: Predicted target values
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return a human-readable name for the model."""
        pass
