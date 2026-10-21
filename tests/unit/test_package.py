import json
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import salary_predictor


def test_package_is_installed() -> None:
    assert salary_predictor.__version__ == version("salary-predictor")
    assert salary_predictor.__version__ == "0.0.0a1"


def test_example_fixture_is_readable() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "example_profile.json"
    profile = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert profile["occupation"] == "Sample profession"
    assert profile["location"] == "NCR"


def test_future_modules_are_separate_packages() -> None:
    assert import_module("salary_predictor.trainer")
    assert import_module("salary_predictor.web")
