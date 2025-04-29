# tests/test_utils.py
from src.utils.utils import format_duration


def test_format_duration():
    assert format_duration(1.2345) == "1.23s"
