from __future__ import annotations

import re

import pytest

from salary_predictor.web import create_app
from salary_predictor.web.profile_flow import get_catalog


@pytest.fixture
def client():
    return create_app({"TESTING": True, "SECRET_KEY": "part13-tests"}).test_client()


def csrf(response):
    found = re.search(r'name="_csrf_token" value="([^"]+)"', response.get_data(as_text=True))
    assert found is not None
    return found.group(1)


def data(**changes):
    values = {
        "job_title": "Software Engineer",
        "occupation_psoc_code": "2512",
        "region_psgc_code": "1300000000",
        "years_experience": "4.5",
        "education_level": "bachelor",
        "skills": "Python (computer programming), SQL",
        "employment_context": "full_time",
    }
    return {**values, **changes}


def submit(client, path="/review", **changes):
    token = csrf(client.get("/profile"))
    return client.post(path, data={"_csrf_token": token, **data(**changes)})


def test_datalist_and_regional_selector_are_real_psoc_psgc_values(client):
    html = client.get("/profile").get_data(as_text=True)
    assert 'list="psoc-options"' in html
    assert 'value="2512 — SOFTWARE DEVELOPERS"' in html
    assert 'value="1300000000"' in html
    assert "National Capital Region (NCR)" in html
    assert html.count("<option value=") > 480
    assert 'id="skill-search"' in html
    assert "js/site.js" in html


def test_complete_review_shows_normalization_but_preserves_original_user_title(client):
    page = submit(client, occupation_psoc_code="2512 — SOFTWARE DEVELOPERS")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Software Engineer" in html
    assert "SOFTWARE DEVELOPERS (2512)" in html
    assert "National Capital Region (NCR) (1300000000)" in html
    assert "Python (computer programming) (ESCO matched)" in html
    assert "ccd0a1d9-afda-43d9-b901-96344886e14d" in html
    assert "See the exact normalized profile fields" in html
    assert "SQL (Free text)" in html


def test_ambiguous_job_title_stays_unmapped_without_chosen_psoc(client):
    page = submit(client, job_title="software developer", occupation_psoc_code="")
    assert page.status_code == 200
    assert "Not mapped" in page.get_data(as_text=True)


@pytest.mark.parametrize("occupation", ["2999", "software developer", "2512 — Wrong label", "251"])
def test_unverified_or_ambiguous_psoc_rejected(client, occupation):
    page = submit(client, occupation_psoc_code=occupation)
    assert page.status_code == 422
    assert "choose a listed PSOC occupation" in page.get_data(as_text=True)
    assert "Software Engineer" in page.get_data(as_text=True)


@pytest.mark.parametrize("region", ["0000000000", "Metro Manila", "130000000"])
def test_noncanonical_region_rejected(client, region):
    page = submit(client, region_psgc_code=region)
    assert page.status_code == 422
    assert "Choose a listed Philippine region" in page.get_data(as_text=True)


@pytest.mark.parametrize("skills", ["SQL, sql", "a," * 31, "a" * 101, "x" * 501])
def test_malformed_skills_rejected(client, skills):
    page = submit(client, skills=skills)
    assert page.status_code == 422
    assert b'name="skills"' in page.data


@pytest.mark.parametrize("years", ["-1", "71", "1.234", "four"])
def test_invalid_experience_rejected(client, years):
    page = submit(client, years_experience=years)
    assert page.status_code == 422
    assert 'aria-invalid="true"' in page.get_data(as_text=True)


def test_partial_profile_with_only_title_is_valid(client):
    page = submit(client, occupation_psoc_code="", region_psgc_code="",
                  years_experience="", education_level="", skills="", employment_context="")
    assert page.status_code == 200
    assert "Not mapped" in page.get_data(as_text=True)


def test_review_edit_roundtrip_then_provider_gets_canonical_profile():
    received = []
    from tests.web.test_shell import fixture_result

    app = create_app(
        {"TESTING": True, "SECRET_KEY": "part13-roundtrip"},
        result_provider=lambda profile: (received.append(profile), fixture_result("UNAVAILABLE"))[1],
    )
    client = app.test_client()
    reviewed = submit(client, occupation_psoc_code="2512 — SOFTWARE DEVELOPERS")
    assert reviewed.status_code == 200
    edited = client.post("/profile", data={"_csrf_token": csrf(reviewed), **data()})
    assert edited.status_code == 200
    html = edited.get_data(as_text=True)
    assert 'value="4.5"' in html
    assert 'name="skills"' in html and "Python (computer programming), SQL" in html
    token = csrf(edited)
    renewed = client.post("/review", data={
        "_csrf_token": token,
        **data(occupation_psoc_code="2512", years_experience="6", skills="SQL"),
    })
    assert renewed.status_code == 200
    result = client.post("/results", data={
        "_csrf_token": csrf(renewed),
        **data(occupation_psoc_code="2512", years_experience="6", skills="SQL"),
    })
    assert result.status_code == 200
    assert len(received) == 1
    assert received[0].occupation_psoc_code == "2512"
    assert received[0].region_psgc_code == "1300000000"
    assert str(received[0].years_experience) == "6"
    assert received[0].normalized_skills == ("SQL",)
    with client.session_transaction() as session:
        assert set(session) == {"_csrf_token"}


def test_skill_lookup_is_bounded_and_source_backed(client):
    valid = client.get("/lookup/skills?q=python")
    assert valid.status_code == 200
    rows = valid.get_json()
    assert rows and len(rows) <= 10
    assert all(row["id"].startswith("http://data.europa.eu/esco/skill/") for row in rows)
    assert client.get("/lookup/skills?q=x").get_json() == []
    assert client.get("/lookup/skills?q=" + "x" * 81).get_json() == []


def test_form_values_and_review_are_html_escaped(client):
    response = submit(client, job_title='<img src=x onerror=alert(1)>')
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "<img src=x" not in text
    assert "&lt;img" in text


def test_csrf_required_for_review_edits_and_results(client):
    client.get("/profile")
    assert client.post("/profile", data=data()).status_code == 400
    assert client.post("/review", data=data()).status_code == 400
    assert client.post("/results", data=data()).status_code == 400


def test_missing_taxonomy_does_not_crash_site_or_accept_forged_codes(tmp_path):
    get_catalog.cache_clear()
    client = create_app({
        "TESTING": True, "SECRET_KEY": "missing-taxonomy",
        "TAXONOMY_ROOT": str(tmp_path),
    }).test_client()
    html = client.get("/profile").get_data(as_text=True)
    assert "lookup files are missing" in html
    assert client.get("/lookup/skills?q=python").get_json() == []
    page = submit(client)
    assert page.status_code == 422
    assert "occupation lookup is unavailable" in page.get_data(as_text=True)
    partial = submit(client, occupation_psoc_code="", region_psgc_code="", skills="SQL")
    assert partial.status_code == 200


def test_user_can_reclassify_role_and_remove_skill_and_region(client):
    reviewed = submit(client)
    assert reviewed.status_code == 200
    revised = client.post("/profile", data={"_csrf_token": csrf(reviewed), **data()})
    assert revised.status_code == 200
    changed = client.post("/review", data={
        "_csrf_token": csrf(revised),
        **data(occupation_psoc_code="2513", region_psgc_code="", skills=""),
    })
    assert changed.status_code == 200
    html = changed.get_data(as_text=True)
    assert "2513" in html
    assert '"region_psgc_code": null' in html
    assert '"normalized_skills": []' in html
