# tests/test_registry.py
from src.core.registry import list_registered_models


def test_model_registry_structure():
    models = list_registered_models()
    assert isinstance(models, list)
