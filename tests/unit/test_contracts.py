from __future__ import annotations

import json
from pathlib import Path

import pytest

from salary_predictor.contracts import (
    CONTRACT_VERSION,
    FEATURE_SCHEMA_VERSION,
    BenchmarkObservation,
    ContractError,
    PredictionResult,
    SalaryObservation,
    SourceManifest,
    StatutoryFloorObservation,
    TaxonomyConcept,
    UserProfile,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests" / "fixtures" / "contracts_v1"
SCHEMA = ROOT / "docs" / "schemas" / "contracts-v1.schema.json"

EXAMPLES = [
    ("profile.json", UserProfile),
    ("salary_observation.json", SalaryObservation),
    ("benchmark_observation.json", BenchmarkObservation),
    ("result_unavailable.json", PredictionResult),
    ("result_benchmark.json", PredictionResult),
    ("result_model_synthetic.json", PredictionResult),
    ("source_manifest.json", SourceManifest),
    ("taxonomy_concept.json", TaxonomyConcept),
    ("statutory_floor_synthetic.json", StatutoryFloorObservation),
]


def sample(filename: str) -> dict:
    return json.loads((GOLDENS / filename).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("filename", "kind"), EXAMPLES)
def test_frozen_v1_json_round_trip(filename: str, kind: type) -> None:
    raw = sample(filename)
    value = kind.from_dict(raw)
    assert value.to_dict() == raw
    assert kind.from_dict(json.loads(json.dumps(value.to_dict()))) == value


def test_machine_readable_v1_schema_matches_the_examples() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("/2020-12/schema")
    assert FEATURE_SCHEMA_VERSION == CONTRACT_VERSION == 1
    for filename, kind in EXAMPLES:
        definition = schema["$defs"][kind.__name__]
        keys = set(sample(filename))
        assert set(definition["required"]).issubset(keys)
        assert keys.issubset(definition["properties"])
        assert definition["additionalProperties"] is False


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("feature_schema_version", 2),
        ("feature_schema_version", True),
        ("occupation_psoc_code", "NCR"),
        ("region_psgc_code", "13a"),
        ("years_experience", "-1"),
        ("years_experience", "70.01"),
        ("years_experience", 5.0),
        ("education_level", "college-ish"),
        ("normalized_skills", ["demo:python", "demo:python"]),
        ("normalized_skills", "demo:python"),
        ("employment_context", "whatever"),
    ],
)
def test_invalid_profile_is_rejected(field: str, bad: object) -> None:
    value = sample("profile.json")
    value[field] = bad
    with pytest.raises(ContractError):
        UserProfile.from_dict(value)


def test_null_profile_fields_are_explicitly_valid() -> None:
    value = sample("profile.json")
    value.update(
        years_experience=None, education_level=None, occupation_psoc_code=None,
        region_psgc_code=None, normalized_skills=[]
    )
    assert UserProfile.from_dict(value).years_experience is None


def test_missing_or_unknown_profile_fields_fail() -> None:
    missing = sample("profile.json")
    del missing["years_experience"]
    with pytest.raises(ContractError, match="missing fields"):
        UserProfile.from_dict(missing)

    extra = sample("profile.json")
    extra["monthly_salary"] = "40000"
    with pytest.raises(ContractError, match="unknown fields"):
        UserProfile.from_dict(extra)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("record_type", "benchmark"),
        ("target_scope", "job_posting"),
        ("original_period", "fortnightly"),
        ("original_amount_min", "0"),
        ("original_amount_min", "50000.456"),
        ("original_amount_min", 20000),
        ("original_amount_max", "1"),
        ("tax_basis", "tax_free"),
        ("thirteenth_month_included", "yes"),
        ("observation_date", "2026-15"),
        ("predictor_completeness", {}),
        ("contract_version", 2),
    ],
)
def test_invalid_salary_observations_fail(field: str, bad: object) -> None:
    value = sample("salary_observation.json")
    value[field] = bad
    with pytest.raises(ContractError):
        SalaryObservation.from_dict(value)


def test_salary_unknown_basis_is_preserved_without_conversion() -> None:
    value = sample("salary_observation.json")
    value.update(
        currency="unknown", original_period="unknown", pay_component="unknown",
        tax_basis="unknown", thirteenth_month_included=None
    )
    actual = SalaryObservation.from_dict(value)
    assert actual.to_dict() == value


def test_valid_job_post_is_not_mixed_with_employee_earnings() -> None:
    value = sample("salary_observation.json")
    value.update(record_type="advertised_salary", target_scope="job_posting")
    assert SalaryObservation.from_dict(value).record_type == "advertised_salary"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("currency", "USD"),
        ("period", "daily"),
        ("source_page", 0),
        ("metric", "individual_salary"),
        ("amount_php_monthly", "-100"),
        ("published_dimensions", ["occupation", "occupation"]),
        ("published_dimensions", ["occupation", "region", "industry"]),
    ],
)
def test_invalid_benchmarks_fail(field: str, bad: object) -> None:
    value = sample("benchmark_observation.json")
    value[field] = bad
    with pytest.raises(ContractError):
        BenchmarkObservation.from_dict(value)


def test_benchmark_cannot_invent_an_occupation_region_cell() -> None:
    value = sample("benchmark_observation.json")
    value["region_label"] = "National Capital Region"
    with pytest.raises(ContractError, match="region"):
        BenchmarkObservation.from_dict(value)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("central_php_monthly", "50000"),
        ("source_id", "fake-source"),
        ("basis", "model_prediction"),
        ("artifact_version", "made-up"),
        ("data_as_of", "2026-01"),
    ],
)
def test_unavailable_result_cannot_claim_evidence(field: str, bad: str) -> None:
    value = sample("result_unavailable.json")
    value[field] = bad
    with pytest.raises(ContractError):
        PredictionResult.from_dict(value)


def test_benchmark_needs_real_source_and_date() -> None:
    value = sample("result_benchmark.json")
    value["source_id"] = None
    with pytest.raises(ContractError):
        PredictionResult.from_dict(value)


def test_model_result_needs_an_artifact_and_cannot_be_relabelled() -> None:
    value = sample("result_model_synthetic.json")
    value["artifact_version"] = None
    with pytest.raises(ContractError):
        PredictionResult.from_dict(value)

    value = sample("result_model_synthetic.json")
    value["basis"] = "published_average_monthly_wage"
    with pytest.raises(ContractError):
        PredictionResult.from_dict(value)


def test_salary_range_needs_documented_support() -> None:
    value = sample("result_benchmark.json")
    value.update(lower_php_monthly="40000", upper_php_monthly="49000")
    with pytest.raises(ContractError, match="range_basis"):
        PredictionResult.from_dict(value)

    value["range_basis"] = "published_source_range"
    assert PredictionResult.from_dict(value).lower_php_monthly is not None

    value["upper_php_monthly"] = "41000"
    with pytest.raises(ContractError, match="range"):
        PredictionResult.from_dict(value)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("file_sha256", "not-a-hash"),
        ("local_path", "../private.csv"),
        ("local_path", "data/raw/../private.csv"),
        ("local_path", "/tmp/private.csv"),
        ("dataset_role", "mystery"),
        ("file_format", "exe"),
        ("column_mapping", {"salary": 10}),
        ("parser_options", {"exec": {"cmd": "bad"}}),
        ("source_url", "file:///tmp/source.csv"),
        ("contract_version", 2),
    ],
)
def test_untrusted_source_manifests_fail(field: str, bad: object) -> None:
    value = sample("source_manifest.json")
    value[field] = bad
    with pytest.raises(ContractError):
        SourceManifest.from_dict(value)


def test_taxonomy_is_a_separate_kind_not_a_salary_record() -> None:
    concept = sample("taxonomy_concept.json")
    assert TaxonomyConcept.from_dict(concept).kind == "occupation"
    with pytest.raises(ContractError):
        SalaryObservation.from_dict(concept)


def test_examples_never_activate_a_model_or_data_import() -> None:
    assert sample("source_manifest.json")["dataset_role"] == "aggregate_benchmark"
    assert sample("salary_observation.json")["source_id"] == "fixture-only-not-real"
    assert sample("result_model_synthetic.json")["artifact_version"] == "fixture-only-not-approved"


def test_statutory_floor_stays_separate_from_monthly_market_pay() -> None:
    original = sample("statutory_floor_synthetic.json")
    assert StatutoryFloorObservation.from_dict(original).to_dict() == original
    invalid = dict(original, period="monthly")
    with pytest.raises(ContractError):
        StatutoryFloorObservation.from_dict(invalid)
    with pytest.raises(ContractError):
        BenchmarkObservation.from_dict(original)


def test_range_kind_must_match_result_mode() -> None:
    benchmark = sample("result_benchmark.json")
    benchmark.update(
        lower_php_monthly="40000", upper_php_monthly="49000",
        range_basis="calibrated_model_interval",
    )
    with pytest.raises(ContractError):
        PredictionResult.from_dict(benchmark)

    model = sample("result_model_synthetic.json")
    model.update(
        lower_php_monthly="40000", upper_php_monthly="60000",
        range_basis="published_source_range",
    )
    with pytest.raises(ContractError):
        PredictionResult.from_dict(model)


def test_result_requires_machine_readable_reliability_flags() -> None:
    value = sample("result_benchmark.json")
    value["reliability_flags"] = ["historical_source", "historical_source"]
    with pytest.raises(ContractError, match="duplicate"):
        PredictionResult.from_dict(value)


def test_result_source_version_is_required_for_benchmarks() -> None:
    value = sample("result_benchmark.json")
    value["source_version"] = None
    with pytest.raises(ContractError):
        PredictionResult.from_dict(value)


def test_benchmark_pay_component_must_match_its_metric() -> None:
    value = sample("benchmark_observation.json")
    value.update(metric="median_basic_pay", pay_component="total")
    with pytest.raises(ContractError, match="pay_component"):
        BenchmarkObservation.from_dict(value)

    value["pay_component"] = "basic"
    assert BenchmarkObservation.from_dict(value).metric == "median_basic_pay"


def test_contract_mapping_fields_cannot_be_changed_after_validation() -> None:
    source = SourceManifest.from_dict(sample("source_manifest.json"))
    with pytest.raises(TypeError):
        source.parser_options["encoding"] = "utf-16"

    salary = SalaryObservation.from_dict(sample("salary_observation.json"))
    with pytest.raises(TypeError):
        salary.predictor_completeness["years_experience"] = False
