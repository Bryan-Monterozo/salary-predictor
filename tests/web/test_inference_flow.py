from __future__ import annotations

import io
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from salary_predictor.contracts import BenchmarkObservation, UserProfile
from salary_predictor.web import create_app
from salary_predictor.web.inference import (
    ApprovedInferenceService,
    _approved_records,
    result_from_verified_records,
)
from tests.unit.test_artifact_release import isolated_root as isolated_root

ROOT = Path(__file__).resolve().parents[2]


def profile(*, occupation="2512", region="1300000000", skills=None):
    return UserProfile.from_dict({
        "feature_schema_version": 1, "occupation_psoc_code": occupation,
        "region_psgc_code": region, "years_experience": "4.00",
        "education_level": "bachelor", "normalized_skills": skills or ["SQL"],
    })


def observed_benchmark():
    return BenchmarkObservation.from_dict({
        "contract_version": 1, "observation_id": "fictional-verified-test",
        "source_id": "psa_ows_2022_public_tables",
        "source_version": "fictional-reviewed-fixture-v1",
        "reference_date": "2023-03", "source_table": "Table 3",
        "source_page": 17, "population": "Fictional fixture full-time wage workers",
        "published_dimensions": ["occupation"], "occupation_psoc_code": "2512",
        "occupation_label": "Software Developers", "region_psgc_code": None,
        "region_label": None, "industry_code": None, "industry_label": None,
        "metric": "mean_monthly_wage", "pay_component": "total",
        "tax_basis": "unknown", "amount_php_monthly": "42000",
        "currency": "PHP", "period": "monthly",
    })


def fields():
    return {
        "job_title": "Developer",
        "occupation_psoc_code": "2512",
        "region_psgc_code": "1300000000",
        "years_experience": "4",
        "education_level": "bachelor",
        "skills": "SQL",
        "employment_context": "full_time",
    }


def token(page):
    match = re.search(r'name="_csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    assert match
    return match.group(1)


def test_default_real_runtime_never_shows_unapproved_local_data():
    service = ApprovedInferenceService(ROOT)
    result = service(profile())
    assert result.mode == "UNAVAILABLE"
    assert result.central_php_monthly is None
    assert result.source_id is None
    assert result.artifact_version is None
    assert "approved" in result.explanation


@pytest.mark.parametrize("occupation", [None, "9999", "2513"])
def test_benchmark_requires_verified_exact_psoc_mapping(occupation):
    result = result_from_verified_records(profile(occupation=occupation), [observed_benchmark()])
    assert result.mode == "UNAVAILABLE"
    assert result.central_php_monthly is None


def test_verified_published_cell_is_a_national_benchmark_not_personal_model():
    result = result_from_verified_records(profile(), [observed_benchmark()])
    assert result.mode == "MARKET_BENCHMARK"
    assert result.central_php_monthly == Decimal("42000")
    assert result.basis == "published_average_monthly_wage"
    assert result.region_specificity == "national"
    assert result.lower_php_monthly is None
    assert result.upper_php_monthly is None
    assert "not your personal salary estimate" in result.explanation
    assert "skills" in " ".join(result.limitations)


def test_same_verified_benchmark_does_not_apply_skill_or_education_premium():
    first = result_from_verified_records(profile(skills=["SQL"]), [observed_benchmark()])
    second = result_from_verified_records(profile(skills=["ESCO:some-skill"]), [
        observed_benchmark()
    ])
    assert first.to_dict() == second.to_dict()


def test_duplicate_or_joint_scope_never_claimed_as_exact_personal_salary():
    row = observed_benchmark()
    assert result_from_verified_records(profile(), [row, row]).mode == "UNAVAILABLE"
    mixed = BenchmarkObservation.from_dict({
        **row.to_dict(),
        "published_dimensions": ["occupation", "region"],
        "region_psgc_code": "1300000000",
        "region_label": "National Capital Region",
    })
    assert result_from_verified_records(profile(), [mixed]).mode == "UNAVAILABLE"


def test_runtime_does_not_accept_unreviewed_pointer_or_config(tmp_path):
    service = ApprovedInferenceService(tmp_path)
    assert service(profile()).mode == "UNAVAILABLE"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "result_decision_v1.json").write_text(json.dumps({
        "schema_version": 1,
        "runtime_release_enabled": True,
        "benchmark_release_enabled": True,
        "model_release_enabled": False,
        "benchmark": {"allow_region_industry_table_join": False,
                      "allow_occupation_hierarchy_fallback": False,
                      "allow_published_ranges": False,
                      "max_reference_age_months": 36},
    }))
    (config_dir / "release_policy.yaml").write_text(json.dumps({
        "status": "APPROVED_FOR_RUNTIME",
        "evidence": {"supervised_training_enabled": False},
    }))
    assert _approved_records(tmp_path, date(2023, 1, 1)) is None
    artifact = tmp_path / "data/artifacts"
    artifact.mkdir(parents=True)
    (artifact / "active.json").write_text(json.dumps({
        "release_id": "../../outside", "scope": "published_benchmark_only"
    }))
    assert service(profile()).mode == "UNAVAILABLE"


def test_manual_and_resume_review_route_identical_normalized_profile_and_result():
    from tests.web.test_resume_upload import docx_file

    seen = []
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "same-profile-fixture"},
        result_provider=lambda p: (
            seen.append(p.to_dict()), result_from_verified_records(p, [observed_benchmark()])
        )[1],
    )
    client = app.test_client()
    manual = client.post("/results", data={
        "_csrf_token": token(client.get("/profile")), **fields()
    })
    assert manual.status_code == 200
    upload = client.post("/upload", data={
        "_csrf_token": token(client.get("/upload")),
        "resume": (io.BytesIO(docx_file([
            "Role: Developer", "Occupation: SOFTWARE DEVELOPERS",
            "Region: Metro Manila", "Skills: SQL",
        ])), "fixture.docx"),
    }, content_type="multipart/form-data")
    assert upload.status_code == 200
    review = client.post("/review", data={
        "_csrf_token": token(upload), **fields()
    })
    assert review.status_code == 200
    resumed = client.post("/results", data={
        "_csrf_token": token(review), **fields()
    })
    assert resumed.status_code == 200
    assert seen[0] == seen[1]
    assert manual.get_data(as_text=True) == resumed.get_data(as_text=True)
    text = resumed.get_data(as_text=True)
    assert "₱42,000.00" in text
    assert "Published aggregate benchmark" in text
    assert "Published aggregate benchmark" in manual.get_data(as_text=True)
    assert "Reviewed predictive model" not in text
    with client.session_transaction() as session:
        assert set(session) == {"_csrf_token"}


@pytest.mark.parametrize("mode,label", [
    ("MODEL_ESTIMATE", "Reviewed predictive model"),
    ("MARKET_BENCHMARK", "Published aggregate benchmark"),
])
def test_mock_result_mode_methodology_is_distinct(mode, label):
    from tests.web.test_shell import fixture_result

    app = create_app(
        {"TESTING": True, "SECRET_KEY": "mode-matrix"},
        result_provider=lambda _p: fixture_result(mode),
    )
    client = app.test_client()
    result = client.post("/results", data={
        "_csrf_token": token(client.get("/profile")), **fields()
    })
    assert result.status_code == 200
    html = result.get_data(as_text=True)
    assert label in html
    assert ("₱55,000.00" if mode == "MODEL_ESTIMATE" else "₱42,000.00") in html


def test_isolated_approved_release_gateway_verifies_manifest_and_fails_closed_on_age(
    isolated_root,
):
    from salary_predictor.trainer import release

    root = isolated_root
    p = root / "configs/release_policy.yaml"
    data = json.loads(p.read_text())
    data["status"] = "APPROVED_FOR_RUNTIME"
    p.write_text(json.dumps(data))
    p = root / "configs/result_decision_v1.json"
    data = json.loads(p.read_text())
    data["runtime_release_enabled"] = True
    data["benchmark_release_enabled"] = True
    p.write_text(json.dumps(data))
    candidate = release.build_benchmark_candidate(root, as_of=date(2023, 1, 21))
    name = candidate["run_id"]
    approval_dir = root / "data/artifacts/approvals"
    approval_dir.mkdir()
    (approval_dir / f"{name}.json").write_text(json.dumps({
        "schema_version": 1, "release_id": name,
        "approved_for": "published_benchmark_only",
        "operator_reviewed": True, "source_rights_reviewed": True,
        "tests_passed_reviewed": True, "release_policy_reviewed": True,
        "manifest_sha256": candidate["manifest_sha256"],
        "operator_id": "fixture-operator",
        "approved_at_utc": "2023-02-01T00:00:00+00:00",
    }))
    artifact = root / "data/artifacts/active.json"
    artifact.write_text(json.dumps({
        "release_id": name, "scope": "published_benchmark_only",
        "manifest_sha256": candidate["manifest_sha256"],
    }))
    records = _approved_records(root, date(2023, 3, 1))
    assert len(records) == 560
    assert all(item.occupation_psoc_code is None for item in records)
    assert result_from_verified_records(profile(), records).mode == "UNAVAILABLE"
    assert _approved_records(root, date(2026, 10, 22)) is None
    bundle = root / candidate["bundle_path"]
    payload = json.loads(bundle.read_text())
    payload["records"][0]["amount_php_monthly"] = "999999"
    bundle.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Benchmark content has changed"):
        _approved_records(root, date(2023, 3, 1))
