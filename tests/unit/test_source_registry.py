from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from salary_predictor.source_registry import load_source_register, validate_source_register

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "data_sources.json"
EXAMPLES = ROOT / "tests" / "fixtures" / "source_evidence_samples.json"


def test_source_register_is_complete_and_valid() -> None:
    register = load_source_register(REGISTRY)
    assert not validate_source_register(register)
    assert len(register["sources"]) == 6


def test_public_benchmarks_are_separate_from_training_data() -> None:
    sources = load_source_register(REGISTRY)["sources"]
    approved = [s["id"] for s in sources if s["public_benchmark_eligible"]]

    assert approved == ["psa_ows_2022_public_tables", "psa_ows_2024_public_tables"]
    assert all(s["dataset_role"] == "aggregate_benchmark" for s in sources if s["public_benchmark_eligible"])
    assert all(not s["enabled_for_import"] for s in sources)
    assert not any(s["dataset_role"] in {"individual_earnings", "advertised_salary"} for s in sources)


def test_source_verified_sample_shapes_match_declared_scopes() -> None:
    sources = {s["id"]: s for s in load_source_register(REGISTRY)["sources"]}
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    esco = examples["esco"]

    assert esco["relation"]["occupationUri"] == esco["occupation"]["conceptUri"]
    assert esco["relation"]["skillUri"] == esco["skill"]["conceptUri"]
    assert esco["relation"]["relationType"] == "essential"
    assert sources["esco_english_bundle"]["dataset_role"] == "taxonomy"

    wages = examples["psa_ows_2022"]
    assert len(wages) == 3
    assert wages[0]["monthly_wage_php"] == 18174
    assert wages[1]["monthly_wage_php"] == 45520
    assert wages[2]["monthly_wage_php"] == 24530
    assert {row["scope"] for row in wages} == {
        "selected_industry_occupation_national",
        "two_benchmark_occupations_by_region",
    }
    assert sources["psa_ows_2022_public_tables"]["wage_period"] == "monthly"


@pytest.mark.parametrize(
    ("change", "expected_error"),
    [
        (lambda s: s.pop("rights_status"), "missing rights_status"),
        (lambda s: s.update(dataset_role="invented_role"), "unknown dataset role"),
        (lambda s: s.update(currency="USD"), "published salary basis is unclear"),
        (lambda s: s.update(approved_use=[]), "benchmark has no approved public source"),
        (lambda s: s.update(reference_date=None, enabled_for_import=True), "enabled source needs verified rights"),
    ],
)
def test_invalid_source_metadata_is_rejected(change, expected_error: str) -> None:
    register = copy.deepcopy(load_source_register(REGISTRY))
    change(register["sources"][0])
    errors = validate_source_register(register)

    assert any(expected_error in message for message in errors)


def test_unreviewed_esco_import_cannot_be_enabled() -> None:
    register = copy.deepcopy(load_source_register(REGISTRY))
    esco = next(source for source in register["sources"] if source["id"] == "esco_english_bundle")
    esco["enabled_for_import"] = True

    assert any("enabled source needs verified rights" in e for e in validate_source_register(register))


def test_repeated_source_ids_are_rejected() -> None:
    register = copy.deepcopy(load_source_register(REGISTRY))
    register["sources"].append(copy.deepcopy(register["sources"][0]))

    assert any("repeated source ID" in e for e in validate_source_register(register))


def test_training_row_without_actual_salary_column_is_rejected() -> None:
    register = copy.deepcopy(load_source_register(REGISTRY))
    source = register["sources"][0]
    source.update(dataset_role="individual_earnings", public_benchmark_eligible=False)

    assert any("cannot train without observed salary" in e for e in validate_source_register(register))
