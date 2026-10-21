from __future__ import annotations

from pathlib import Path

import pytest

from salary_predictor.contracts import ContractError, UserProfile
from salary_predictor.taxonomy import Taxonomy, normalize_experience

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return Taxonomy(ROOT)


def test_region_dictionary_has_18_dated_psgc_codes(taxonomy: Taxonomy) -> None:
    regions = taxonomy.browse("region", limit=100)
    assert len(regions) == 18
    assert len({entry["id"] for entry in regions}) == 18
    assert all(len(entry["id"]) == 10 for entry in regions)
    assert taxonomy.resolve("region", "Metro Manila").chosen.id == "1300000000"
    assert taxonomy.resolve("region", "NCR").chosen.id == "1300000000"
    assert taxonomy.resolve("region", "NIR").chosen.id == "1800000000"


def test_region_unknown_and_colliding_codes_are_not_guessed(taxonomy: Taxonomy) -> None:
    assert taxonomy.resolve("region", "Manila").status != "matched"
    assert taxonomy.resolve("region", "1300000000").chosen.id == "1300000000"
    missing = taxonomy.resolve("region", "Region XIV")
    assert missing.status in {"unknown", "suggested"}
    assert missing.chosen is None


def test_psoc_workbook_preserves_its_hierarchy(taxonomy: Taxonomy) -> None:
    group = taxonomy.browse("occupation", "Software Developers", limit=25)
    official = [entry for entry in group if entry["source"] == "psoc_2022_workbook"]
    assert official[0]["id"] == "2512"
    assert official[0]["parent_code"] == "251"
    assert taxonomy.resolve("occupation", "2512").chosen.id == "2512"


def test_psoc_and_esco_same_wording_is_ambiguous(taxonomy: Taxonomy) -> None:
    result = taxonomy.resolve("occupation", "software developer")
    assert result.status == "ambiguous"
    assert result.chosen is None
    assert {x.source for x in result.alternatives} >= {"psoc_2022_workbook", "esco_local"}


def test_crosswalk_is_a_review_suggestion_never_an_auto_mapping(taxonomy: Taxonomy) -> None:
    uri = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
    mapping = taxonomy.crosswalk_for_esco(uri)
    assert mapping["psoc_code"] == "2512"
    assert mapping["auto_assign"] is False
    assert mapping["confidence"] == 0.95
    assert taxonomy.crosswalk_for_esco("not-a-uri") is None


def test_all_esco_concepts_are_available_without_double_counting(taxonomy: Taxonomy) -> None:
    occupations = [
        item for item in taxonomy._entries["occupation"] if item.source == "esco_local"
    ]
    skills = taxonomy._entries["skill"]
    assert len(occupations) == 3039
    assert len(skills) == 13939
    assert len({i.id for i in occupations}) == 3039


@pytest.mark.parametrize("query", ["nurse", "teacher", "driver", "cook", "farmer", "carpenter"])
def test_broad_profession_search_does_not_require_salary_data(
    taxonomy: Taxonomy, query: str
) -> None:
    matches = taxonomy.browse("occupation", query, limit=10)
    assert matches
    assert all(row["source"] in {"esco_local", "psoc_2022_workbook"} for row in matches)


def test_occupation_search_provides_browse_groups(taxonomy: Taxonomy) -> None:
    labels = {
        entry.browse_group for entry in taxonomy._entries["occupation"]
        if entry.source == "esco_local"
    }
    assert "Agriculture and fisheries" in labels
    assert "Craft and trades" in labels
    assert "Services and sales" in labels


def test_skills_keep_user_text_when_not_found(taxonomy: Taxonomy) -> None:
    missing = taxonomy.resolve("skill", "super made-up plasma javascript magic skill")
    assert missing.status == "unknown"
    assert missing.raw_input == "super made-up plasma javascript magic skill"
    assert missing.chosen is None


def test_misspellings_only_suggest_not_auto_assign(taxonomy: Taxonomy) -> None:
    result = taxonomy.resolve("occupation", "softwre developer")
    assert result.status in {"suggested", "unknown"}
    assert result.chosen is None
    if result.alternatives:
        assert result.alternatives[0].score < 1
        assert result.alternatives[0].evidence == "possible_spelling_only"


def test_education_aliases_and_ambiguous_college(taxonomy: Taxonomy) -> None:
    assert taxonomy.resolve("education", "SHS").chosen.id == "secondary"
    assert taxonomy.resolve("education", "PhD").chosen.id == "doctorate"
    mixed = taxonomy.resolve("education", "college")
    assert mixed.status == "ambiguous"
    assert mixed.chosen is None
    assert {item.id for item in mixed.alternatives} == {"bachelor", "associate"}
    assert taxonomy.resolve("education", "college undergraduate").chosen is None


@pytest.mark.parametrize(
    ("source", "status", "years"),
    [
        ("0 years", "matched", "0"),
        ("3 yrs", "matched", "3"),
        ("1.5 years", "matched", "1.5"),
        ("6 months", "matched", "0.50"),
        ("1 month", "matched", "0.08"),
        ("3+ years", "ambiguous", None),
        ("2-4 years", "ambiguous", None),
        ("", "unknown", None),
        ("75 years", "ambiguous", None),
    ],
)
def test_experience_normalizer(source: str, status: str, years: str | None) -> None:
    normalized = normalize_experience(source)
    assert normalized.status == status
    assert (str(normalized.years) if normalized.years is not None else None) == years


def test_reviewed_profile_shares_v1_contract(taxonomy: Taxonomy) -> None:
    skill = taxonomy._entries["skill"][0].id
    profile = taxonomy.reviewed_profile(
        occupation_psoc_code="2512", region_psgc_code="1300000000",
        education_level="bachelor", years_experience="2.5", normalized_skills=[skill],
    )
    assert isinstance(profile, UserProfile)
    assert UserProfile.from_dict(profile.to_dict()) == profile


def test_reviewed_profile_rejects_unknown_codes_and_skills(taxonomy: Taxonomy) -> None:
    values = dict(
        occupation_psoc_code="2512", region_psgc_code="1300000000",
        education_level="bachelor", years_experience="2", normalized_skills=[],
    )
    with pytest.raises(ContractError):
        taxonomy.reviewed_profile(**dict(values, occupation_psoc_code="9876"))
    with pytest.raises(ContractError):
        taxonomy.reviewed_profile(**dict(values, region_psgc_code="0000000000"))
    with pytest.raises(ContractError):
        taxonomy.reviewed_profile(**dict(values, normalized_skills=["imaginary-skill"]))


def test_lookup_is_deterministic_and_frontend_ready(taxonomy: Taxonomy) -> None:
    a = taxonomy.browse("occupation", "nurse", limit=8)
    assert a == taxonomy.browse("occupation", "nurse", limit=8)
    assert all({"id", "label", "source", "browse_group"}.issubset(row) for row in a)
    assert taxonomy.resolve("region", "Metro Manila").to_dict()["chosen"]["id"] == "1300000000"


def test_invalid_limit_and_kind_fail_closed(taxonomy: Taxonomy) -> None:
    with pytest.raises(ValueError):
        taxonomy.browse("salary", "nurse")
    with pytest.raises(ValueError):
        taxonomy.browse("skill", limit=1000)
    with pytest.raises(ValueError):
        taxonomy.resolve("region", "NCR", limit=0)



def test_esco_occupation_skills_are_relationships_not_salary_factors(
    taxonomy: Taxonomy,
) -> None:
    occupation = "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"
    skills = taxonomy.skills_for_esco_occupation(occupation, limit=15)
    assert skills
    assert all(row["relation_type"] in {"essential", "optional"} for row in skills)
    assert all(row["skill_uri"].startswith("http://data.europa.eu/esco/skill/") for row in skills)
    assert all("salary" not in row and "premium" not in row for row in skills)
    assert skills == taxonomy.skills_for_esco_occupation(occupation, limit=15)


def test_esco_skill_edges_reject_unknown_occupation(taxonomy: Taxonomy) -> None:
    with pytest.raises(ValueError, match="Unknown ESCO"):
        taxonomy.skills_for_esco_occupation("not-a-real-uri")


def test_education_and_region_results_have_no_fake_ph_occupation_code(
    taxonomy: Taxonomy,
) -> None:
    match = taxonomy.resolve("skill", "SQL")
    if match.chosen is not None:
        assert match.chosen.source == "esco_local"
        assert match.chosen.parent_code is None



def test_separate_consumers_get_identical_normalization(taxonomy: Taxonomy) -> None:
    second = Taxonomy(ROOT)
    for kind, input_text in (
        ("region", "Metro Manila"),
        ("education", "college"),
        ("occupation", "software developer"),
        ("skill", "SQL"),
    ):
        assert taxonomy.resolve(kind, input_text).to_dict() == second.resolve(
            kind, input_text
        ).to_dict()
    assert taxonomy.browse("occupation", "carpenter", limit=5) == second.browse(
        "occupation", "carpenter", limit=5
    )



def test_workbook_has_ten_major_groups_and_complete_parent_hierarchy(taxonomy: Taxonomy) -> None:
    units = [row for row in taxonomy.psoc_groups if row.level == "unit"]
    minor = [row for row in taxonomy.psoc_groups if row.level == "minor"]
    sub_major = [row for row in taxonomy.psoc_groups if row.level == "sub_major"]
    assert len(units) == 466
    assert len(minor) == 130
    assert len(sub_major) == 43
    assert {row.major_group for row in taxonomy.psoc_groups} == set("0123456789")
    assert all(row.parent_code in taxonomy._psoc_by_code for row in units + minor)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("2512", "SOFTWARE DEVELOPERS"),
        ("2221", "NURSING PROFESSIONALS"),
        ("6111", "RICE FARMERS"),
        ("7115", "CARPENTERS AND JOINERS"),
        ("8322", "CAR, TAXI AND VAN DRIVERS"),
        ("0110", "COMMISSIONED ARMED FORCES OFFICERS"),
    ],
)
def test_source_file_contains_the_actual_occupations(
    taxonomy: Taxonomy, code: str, name: str
) -> None:
    groups = taxonomy.psoc_hierarchy(code)
    assert [row["code"] for row in groups] == [
        code[:1], code[:2], code[:3], code
    ]
    assert groups[-1]["label"] == name
    assert groups[-1]["source_sheet"] == f"Group {code[0]}"
    assert taxonomy.resolve("occupation", code).chosen.id == code


def test_workbook_codes_work_as_shared_profile_values(taxonomy: Taxonomy) -> None:
    profile = taxonomy.reviewed_profile(
        occupation_psoc_code="6111", region_psgc_code="1300000000",
        education_level=None, years_experience=None, normalized_skills=[]
    )
    assert profile.occupation_psoc_code == "6111"
    assert UserProfile.from_dict(profile.to_dict()) == profile


def test_psoc_workbook_exposes_all_four_digit_units_to_browser(taxonomy: Taxonomy) -> None:
    units = [
        entry for entry in taxonomy._entries["occupation"]
        if entry.source == "psoc_2022_workbook"
    ]
    assert len(units) == 466
    assert all(entry.parent_code == entry.id[:3] for entry in units)
    assert taxonomy.browse("occupation", "RICE FARMERS")


def test_psoc_rejects_unlisted_or_partial_codes(taxonomy: Taxonomy) -> None:
    with pytest.raises(ValueError, match="Unknown PSOC"):
        taxonomy.psoc_hierarchy("9999")
    with pytest.raises(ContractError, match="not a known PSOC unit"):
        taxonomy.reviewed_profile(
            occupation_psoc_code="251", region_psgc_code=None,
            education_level=None, years_experience=None, normalized_skills=[]
        )
