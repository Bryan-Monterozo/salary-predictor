from __future__ import annotations

import builtins
import re
from decimal import Decimal
from pathlib import Path

import pytest

from salary_predictor.contracts import PredictionResult
from salary_predictor.web import create_app

ROOT = Path(__file__).resolve().parents[2]


def fixture_result(mode: str) -> PredictionResult:
    if mode == "UNAVAILABLE":
        return PredictionResult.from_dict({
            "contract_version": 1, "mode": mode,
            "central_php_monthly": None, "lower_php_monthly": None,
            "upper_php_monthly": None, "range_basis": None, "basis": "none",
            "data_as_of": None, "occupation_specificity": "unknown",
            "region_specificity": "unknown", "explanation": "Fixture unavailable.",
            "limitations": ["Fake test evidence"], "reliability_flags": ["fixture"],
            "source_id": None, "source_version": None, "artifact_version": None,
            "currency": "PHP", "period": "monthly",
        })
    return PredictionResult.from_dict({
        "contract_version": 1, "mode": mode,
        "central_php_monthly": "55000" if mode == "MODEL_ESTIMATE" else "42000",
        "lower_php_monthly": None, "upper_php_monthly": None,
        "range_basis": None,
        "basis": ("model_prediction" if mode == "MODEL_ESTIMATE"
                  else "published_average_monthly_wage"),
        "data_as_of": "2023-04-01",
        "occupation_specificity": "group", "region_specificity": "national",
        "explanation": "Fictional test result. Not a salary quote.",
        "limitations": ["Fixture only"], "reliability_flags": ["fixture"],
        "source_id": None if mode == "MODEL_ESTIMATE" else "fixture_psa",
        "source_version": None if mode == "MODEL_ESTIMATE" else "fictional-v1",
        "artifact_version": "fictional-ml" if mode == "MODEL_ESTIMATE" else None,
        "currency": "PHP", "period": "monthly",
    })


@pytest.fixture
def app():
    return create_app({"TESTING": True, "SECRET_KEY": "example-only-unit-tests"})


@pytest.fixture
def client(app):
    return app.test_client()


def _csrf(response) -> str:
    text = response.get_data(as_text=True)
    match = re.search(r'name="_csrf_token" value="([^"]+)"', text)
    assert match
    return match.group(1)


def _details():
    return {
        "job_title": "Software Developer",
        "occupation_psoc_code": "2512",
        "region_psgc_code": "1300000000",
        "years_experience": "4",
        "education_level": "bachelor",
        "skills": "SQL, Python",
        "employment_context": "full_time",
    }


@pytest.mark.parametrize("path, phrase", [
    ("/", "Philippine Salary Predictor"),
    ("/profile", "Enter your profile"),
    ("/review", "No profile has been submitted"),
    ("/results", "NOT YET AVAILABLE"),
    ("/upload", "Resume upload"),
])
def test_public_pages_render_without_raw_data_or_models(client, path, phrase) -> None:
    response = client.get(path)
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "<main" in text
    assert phrase in text


@pytest.mark.parametrize("path, target", [
    ("/start", "/profile"),
    ("/how-it-works", "/"),
    ("/privacy", "/"),
])
def test_old_information_routes_redirect(client, path, target) -> None:
    response = client.get(path)
    assert response.status_code == 302
    assert response.headers["Location"].endswith(target)


def test_main_navigation_mobile_meta_and_local_assets(client) -> None:
    home = client.get("/")
    text = home.get_data(as_text=True)
    assert '<meta name="viewport"' in text
    assert 'href="#main"' in text
    assert 'aria-label="Main navigation"' in text
    assert 'src="/static/js/site.js"' not in text
    assert "cdn.jsdelivr.net" not in text
    css = client.get("/static/css/site.css")
    assert css.status_code == 200
    assert b"@media (max-width: 640px)" in css.data
    assert b":focus-visible" in css.data

def test_manual_review_and_default_result_do_not_store_profile(client) -> None:
    token = _csrf(client.get("/profile"))
    review = client.post("/review", data={"_csrf_token": token, **_details()})
    assert review.status_code == 200
    assert b"Software Developer" in review.data
    assert b"SQL, Python" in review.data
    token2 = _csrf(review)
    result = client.post("/results", data={"_csrf_token": token2, **_details()})
    assert result.status_code == 200
    assert b"NOT YET AVAILABLE" in result.data
    assert b"55000" not in result.data
    with client.session_transaction() as sess:
        assert set(sess) == {"_csrf_token"}
        assert "job_title" not in sess


@pytest.mark.parametrize("mode, expected", [
    ("MODEL_ESTIMATE", "₱55,000.00"),
    ("MARKET_BENCHMARK", "₱42,000.00"),
    ("UNAVAILABLE", "NOT YET AVAILABLE"),
])
def test_all_three_stubbed_result_modes_render(mode, expected) -> None:
    app = create_app({"TESTING": True, "SECRET_KEY": "mock-test"},
                     result_provider=lambda _profile: fixture_result(mode))
    client = app.test_client()
    token = _csrf(client.get("/profile"))
    response = client.post("/results", data={"_csrf_token": token, **_details()})
    assert response.status_code == 200
    assert expected in response.get_data(as_text=True)
    if mode == "UNAVAILABLE":
        assert "Fixture unavailable." in response.get_data(as_text=True)
    else:
        assert "Fictional test result." in response.get_data(as_text=True)
    assert "0.00 –" not in response.get_data(as_text=True)


def test_invalid_form_feedback_preserves_input(client) -> None:
    token = _csrf(client.get("/profile"))
    form = {**_details(), "job_title": "A", "years_experience": "999"}
    response = client.post("/review", data={"_csrf_token": token, **form})
    assert response.status_code == 422
    assert b"Check the following fields:" in response.data
    assert b'value="999"' in response.data
    assert b"Job title" in response.data


def test_csrf_missing_and_incorrect_token_rejected(client) -> None:
    first = client.post("/review", data=_details())
    assert first.status_code == 400
    client.get("/profile")
    second = client.post(
        "/results", data={"_csrf_token": "forged-value", **_details()}
    )
    assert second.status_code == 400


def test_template_injection_is_escaped(client) -> None:
    token = _csrf(client.get("/profile"))
    title = '<img src=x onerror=alert(1)>'
    review = client.post("/review", data={
        "_csrf_token": token, **_details(), "job_title": title,
    })
    assert review.status_code == 200
    assert "<img src=x" not in review.get_data(as_text=True)
    assert "&lt;img" in review.get_data(as_text=True)


def test_headers_errors_and_unsupported_upload(client) -> None:
    missing = client.get("/unknown-page")
    assert missing.status_code == 404
    assert "Error 404" in missing.get_data(as_text=True)
    page = client.get("/upload")
    assert page.status_code == 200
    assert 'name="resume"' in page.get_data(as_text=True)
    for header in ("Content-Security-Policy", "X-Content-Type-Options",
                   "Referrer-Policy", "Cache-Control"):
        assert header in page.headers
    assert "no-store" in page.headers["Cache-Control"]


def test_importing_and_rendering_web_does_not_load_trainer(monkeypatch) -> None:
    base_import = builtins.__import__

    def no_trainer(name, *args, **kwargs):
        if name.startswith("salary_predictor.trainer"):
            raise AssertionError("Web attempted to import trainer")
        return base_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trainer)
    client = create_app({"TESTING": True, "SECRET_KEY": "test"}).test_client()
    assert client.get("/").status_code == 200
    token = _csrf(client.get("/profile"))
    assert client.post("/results", data={
        "_csrf_token": token, **_details()
    }).status_code == 200


def test_php_formatter_and_no_fabricated_salary():
    from salary_predictor.web import php_monthly, unavailable_result

    assert php_monthly(Decimal("45520.5")) == "₱45,520.50"
    assert php_monthly(None) == "Not available"
    result = unavailable_result()
    assert result.mode == "UNAVAILABLE"
    assert result.central_php_monthly is None


def test_app_starts_from_empty_directory_without_source_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app({"TESTING": True, "SECRET_KEY": "isolated-shell-test"})
    client = app.test_client()
    for path in ("/", "/profile", "/results", "/upload"):
        response = client.get(path)
        assert response.status_code == 200
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "configs").exists()

def test_edit_returns_profile_with_all_values_preserved(client) -> None:
    token = _csrf(client.get("/profile"))
    review = client.post("/review", data={"_csrf_token": token, **_details()})
    assert review.status_code == 200
    assert "Edit details" in review.get_data(as_text=True)
    edit = client.post("/profile", data={"_csrf_token": _csrf(review), **_details()})
    assert edit.status_code == 200
    html = edit.get_data(as_text=True)
    assert 'value="Software Developer"' in html
    assert 'value="1300000000"' in html
    assert "SQL, Python" in html
    with client.session_transaction() as sess:
        assert set(sess) == {"_csrf_token"}
