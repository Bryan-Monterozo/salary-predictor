from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from salary_predictor.web import create_app
from salary_predictor.web.experimental import parameters, worker_predict

ROOT = Path(__file__).resolve().parents[2]


def _csrf(page) -> str:
    match = re.search(r'name="_csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    assert match is not None
    return match.group(1)


def _details():
    return {
        "job_title": "Software Developer",
        "occupation_psoc_code": "2512",
        "region_psgc_code": "1300000000",
        "years_experience": "4",
        "education_level": "bachelor",
        "skills": "C#, SQL, Flutter",
        "employment_context": "full_time",
    }


def _assumptions(**kwargs):
    return {
        "usd_to_php": "58",
        "current_salary_php": "",
        "seniority": "Any",
        "isco_major": "",
        "step_region_code": "",
        "education_years": "",
        "psa_occupation": "",
        "psa_industry": "",
        "psa_region": "",
        "scenario_annual_growth": "",
        **kwargs,
    }


def _client(enabled=True):
    return create_app({
        "TESTING": True,
        "SECRET_KEY": "experimental-web-test",
        "EXPERIMENTAL_MODE": enabled,
    }).test_client()


def _start(client):
    review = client.post(
        "/review", data={"_csrf_token": _csrf(client.get("/profile")), **_details()}
    )
    assert review.status_code == 200
    setup = client.post(
        "/experimental/start",
        data={"_csrf_token": _csrf(review), **_details()},
    )
    assert setup.status_code == 200
    return review, setup


def test_experiment_is_separate_from_unapproved_runtime_and_requires_opt_in():
    client = _client(False)
    review = client.post(
        "/review", data={"_csrf_token": _csrf(client.get("/profile")), **_details()}
    )
    assert b"Explore experimental salary" not in review.data
    assert client.post("/experimental/start", data={
        "_csrf_token": _csrf(review), **_details(),
    }).status_code == 404
    assert client.post("/experimental/results", data={
        "_csrf_token": _csrf(review), **_details(), **_assumptions(),
    }).status_code == 404
    approved = client.post("/results", data={
        "_csrf_token": _csrf(review), **_details(),
    })
    assert approved.status_code == 200
    assert b"NOT YET AVAILABLE" in approved.data


def test_experiment_rejects_non_loopback_access():
    client = _client()
    token = _csrf(client.get("/profile"))
    response = client.post(
        "/experimental/start",
        data={"_csrf_token": token, **_details()},
        environ_overrides={"REMOTE_ADDR": "198.51.100.9"},
    )
    assert response.status_code == 404


def test_form_review_experimental_setup_keeps_title_skills_and_user_assumptions():
    client = _client()
    review, setup = _start(client)
    assert b"Run experimental salary scenario" in review.data
    text = setup.get_data(as_text=True)
    assert "EXPERIMENTAL" not in text or "not a Philippine salary prediction" in text
    assert "Software Developer" in text
    assert "C#, SQL, Flutter" in text
    assert 'name="usd_to_php"' in text
    assert 'name="current_salary_php"' in text
    assert 'name="psa_occupation"' in text
    assert 'name="step_region_code"' in text
    assert 'name="scenario_annual_growth"' in text
    assert 'value="58"' not in text


def test_worker_called_only_after_validated_experimental_submission(monkeypatch):
    client = _client()
    _review, setup = _start(client)
    seen = []

    def fake_worker(_root, payload):
        seen.append(payload)
        return {
            "mode": "EXPERIMENTAL_LOCAL_ONLY",
            "model_artifact": "exp-test-fixture",
            "profile": {"job_title": "Software Developer"},
            "sources": {"jobicy_remote_offer": {
                "monthly_usd": 3316.2, "monthly_php_scenario": 192339.68,
                "rate_assumption_php_per_usd": 58,
                "holdout_mae_usd_monthly": 2849.85,
                "supported_ph_salary_ads": 0,
            }},
            "estimated_monthly_php_scenario": 192339.68,
            "scenario_label": "REMOTE_OFFER_FX_ONLY_NOT_PH_DOMESTIC_ESTIMATE",
            "blend_weights_illustrative_not_learned": {"jobicy_remote_offer": 1.0},
            "current_salary_php_reference": 45000,
            "forecast": None,
            "limitations": ["Not validated as Philippine domestic pay"],
        }

    monkeypatch.setattr("salary_predictor.web.worker_predict", fake_worker)
    page = client.post("/experimental/results", data={
        "_csrf_token": _csrf(setup), **_details(),
        **_assumptions(current_salary_php="45000"),
    })
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "EXPERIMENTAL LOCAL ONLY" in text
    assert "US$3,316.20" in text
    assert "₱192,339.68" in text
    assert "No domestic PHP test accuracy" in text
    assert "Philippine domestic estimate" in text
    assert "₱45,000.00" in text
    assert len(seen) == 1
    assert seen[0]["job_title"] == "Software Developer"
    assert seen[0]["skills"] == ["C#", "SQL", "Flutter"]
    assert seen[0]["years_experience"] == 4.0
    assert seen[0]["usd_to_php"] == 58.0
    assert seen[0]["current_salary_php"] == 45000.0
    with client.session_transaction() as session:
        assert set(session) == {"_csrf_token"}


@pytest.mark.parametrize("override,marker", [
    ({"usd_to_php": "NaN"}, "Enter a value from 1 to 150"),
    ({"usd_to_php": "99999"}, "Enter a value from 1 to 150"),
    ({"current_salary_php": "-3"}, "Enter a value from 0.01"),
    ({"seniority": "Fictional"}, "Choose a listed advertised seniority proxy"),
    ({"isco_major": "11"}, "Choose an ISCO major group"),
    ({"step_region_code": "40"}, "Choose a historical STEP region code"),
    ({"step_region_code": "13"}, "STEP requires ISCO major"),
    ({"psa_industry": "J62"}, "Enter the exact published PSA occupation"),
    ({"psa_occupation": "Software Developers"}, "Choose a published source industry"),
    ({"psa_occupation": "Software Developers", "psa_industry": "J62", "psa_region": "NCR"}, "Use either source industry"),
    ({"scenario_annual_growth": "1"}, "Enter a value from -0.5 to 0.5"),
])
def test_invalid_assumptions_fail_without_running_model(override, marker, monkeypatch):
    client = _client()
    _review, setup = _start(client)
    def fail(_root, _payload):
        raise AssertionError("Should not run invalid input")
    monkeypatch.setattr("salary_predictor.web.worker_predict", fail)
    response = client.post("/experimental/results", data={
        "_csrf_token": _csrf(setup), **_details(), **_assumptions(**override),
    })
    assert response.status_code == 422
    assert marker in response.get_data(as_text=True)


def test_experimental_routes_require_csrf():
    client = _client()
    client.get("/profile")
    assert client.post("/experimental/start", data=_details()).status_code == 400
    assert client.post("/experimental/results", data={
        **_details(), **_assumptions(),
    }).status_code == 400


def test_experimental_worker_failure_is_safe_and_does_not_echo_profile(monkeypatch):
    client = _client()
    _review, setup = _start(client)
    def error(_root, _payload):
        raise ValueError("private@example.com worker error")
    monkeypatch.setattr("salary_predictor.web.worker_predict", error)
    page = client.post("/experimental/results", data={
        "_csrf_token": _csrf(setup), **_details(), **_assumptions(),
    })
    assert page.status_code == 503
    assert b"Local experimental model unavailable" in page.data
    assert b"private@example.com" not in page.data


def test_real_saved_random_forest_model_runs_in_local_worker():
    values, _raw, errors = parameters(_assumptions())
    assert errors == {}
    result = worker_predict(ROOT, {
        "job_title": "Software Developer", "skills": ["C#", "SQL", "Flutter"],
        "years_experience": 4.0, **values,
    })
    assert result["mode"] == "EXPERIMENTAL_LOCAL_ONLY"
    assert result["model_artifact"] == "exp-0789113371ab0b1f059c"
    assert result["sources"]["jobicy_remote_offer"]["supported_ph_salary_ads"] == 0
    assert result["sources"]["jobicy_remote_offer"]["monthly_usd"] > 0
    assert result["estimated_monthly_php_scenario"] == 192339.68


def test_resume_to_experimental_setup_preserves_suggestions():
    from tests.web.test_resume_upload import docx_file
    client = _client()
    upload = client.post("/upload", data={
        "_csrf_token": _csrf(client.get("/upload")),
        "resume": (io.BytesIO(docx_file([
            "Role: Software Developer",
            "Skills: SQL, Flutter",
        ])), "sample.docx"),
    }, content_type="multipart/form-data")
    assert upload.status_code == 200
    values = _details()
    values["skills"] = "SQL, Flutter"
    review = client.post("/review", data={
        "_csrf_token": _csrf(upload), **values,
    })
    assert review.status_code == 200
    setup = client.post("/experimental/start", data={
        "_csrf_token": _csrf(review), **values,
    })
    assert setup.status_code == 200
    assert b"SQL, Flutter" in setup.data


def test_actual_web_inference_three_source_experiment_matches_cli():
    client = _client()
    _review, setup = _start(client)
    response = client.post("/experimental/results", data={
        "_csrf_token": _csrf(setup),
        **_details(),
        **_assumptions(
            current_salary_php="45000", isco_major="2", step_region_code="13",
            education_years="16", psa_occupation="Software Developers",
            psa_industry="J62", scenario_annual_growth="0.03",
        ),
    })
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "EXPERIMENTAL LOCAL ONLY" in html
    assert "₱192,339.68" in html
    assert "₱15,891.22" in html
    assert "₱45,520.00" in html
    assert "₱43,972.35" in html
    assert "65%" in html and "30%" in html and "5%" in html
    assert "1 year(s)" in html and "5 year(s)" in html
    assert "No verified PH-specific heldout monthly PHP accuracy" in html
