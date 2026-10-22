from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from pathlib import Path

from salary_predictor.feasibility.consolidated import consolidate, preview_career_match
from salary_predictor.taxonomy import Taxonomy
from salary_predictor.trainer.baseline import BenchmarkOnlyBaseline
from salary_predictor.trainer.step_research import (
    _estimator,
    _frame,
    load_research_sample,
    split_clusters,
)

from .experimental_multisource import verify_experiment

MAJOR_GROUPS = frozenset({"1", "2", "3", "4", "5", "6", "7", "8", "9"})


def _accepted_isco_major(taxonomy: Taxonomy, title: str) -> dict:
    match = taxonomy.resolve("occupation", title)
    candidates = [c for c in (match.chosen, *match.alternatives) if c is not None]
    strong = [c for c in candidates if c.score >= .95 and c.parent_code]
    majors = {c.parent_code[:1] for c in strong}
    if len(majors) == 1 and majors <= MAJOR_GROUPS:
        return {"code": next(iter(majors)), "evidence": "high_similarity_cross_taxonomy_major_only"}
    return {"code": None, "evidence": "ambiguous_or_unknown_requires_explicit_isco_major"}


def _step_hourly(root: Path, *, major: str, region: str, years_education: float) -> float:
    import numpy as np

    rows, _ = load_research_sample(root)
    split = split_clusters(rows)
    tr = split["train"]
    model = _estimator("ridge")
    training = _frame(rows, tr)
    target = np.asarray([rows[i]["hourly_php"] for i in tr])
    weights = np.asarray([rows[i]["person_weight"] for i in tr])
    model.fit(training, target, ridge__sample_weight=weights / weights.mean())
    import pandas as pd

    features = pd.DataFrame([{
        "occupation": major,
        "region_code": region,
        "years_educ": years_education,
    }])
    return max(0.0, float(model.predict(features)[0]))



def _esco_path_suggestions(taxonomy: Taxonomy, title: str, skills: list[str]) -> list[dict]:
    match = taxonomy.resolve("occupation", title)
    candidates = [
        c for c in (match.chosen, *match.alternatives)
        if c is not None and c.source == "esco_local" and c.score >= 0.95
    ]
    if not candidates:
        return []
    observed = {skill.casefold().strip() for skill in skills}
    found = []
    for record in taxonomy.skills_for_esco_occupation(candidates[0].id, limit=100):
        if record["skill_label"].casefold() in observed:
            continue
        found.append({
            "skill": record["skill_label"],
            "relation": record["relation_type"],
            "source": "ESCO_occupation_taxonomy",
            "is_personally_observed": False,
            "salary_premium_established": False,
        })
        if len(found) == 10:
            break
    return found


def predict_experimental(
    root: Path, *,
    job_title: str,
    skills: list[str],
    years_experience: float | None,
    current_salary_php: float | None,
    usd_to_php: float | None,
    seniority: str = "Any",
    isco_major: str | None = None,
    step_region_code: str | None = None,
    education_years: float | None = None,
    psa_occupation: str | None = None,
    psa_industry: str | None = None,
    psa_region: str | None = None,
    scenario_annual_growth: float | None = None,
    run_id: str | None = None,
    as_of: date | None = None,
) -> dict:
    import joblib
    import pandas as pd

    root = root.resolve(strict=True)
    if not isinstance(job_title, str) or not 2 <= len(job_title.strip()) <= 150:
        raise ValueError("Provide a job title between 2 and 150 characters")
    if any(not isinstance(s, str) or len(s) > 100 for s in skills) or len(skills) > 100:
        raise ValueError("Invalid skills")
    if years_experience is not None and (not math.isfinite(years_experience) or not 0 <= years_experience <= 70):
        raise ValueError("Years of experience must be 0–70")
    if current_salary_php is not None and (not math.isfinite(current_salary_php) or current_salary_php <= 0):
        raise ValueError("Current salary must be positive")
    if usd_to_php is not None and (not math.isfinite(usd_to_php) or not 1 <= usd_to_php <= 150):
        raise ValueError("Provide an explicit USD to PHP scenario rate")
    if scenario_annual_growth is not None and (
        not math.isfinite(scenario_annual_growth) or not -.5 <= scenario_annual_growth <= .5
    ):
        raise ValueError("Scenario annual growth must be from -0.5 to 0.5")
    if not isinstance(seniority, str) or seniority not in {
        "Any", "Senior", "Director", "Midweight", "Entry-Level, Junior"
    }:
        raise ValueError("Select a supported advertised seniority level")
    if isco_major is not None and isco_major not in MAJOR_GROUPS:
        raise ValueError("ISCO major must be a one-digit occupational code")
    if step_region_code is not None and (
        not step_region_code.isdigit() or not 1 <= int(step_region_code) <= 16
    ):
        raise ValueError("STEP region must be an historical survey-era code")
    if education_years is not None and (
        not math.isfinite(education_years) or not 0 <= education_years <= 30
    ):
        raise ValueError("Education years must be between 0 and 30")
    if psa_industry and psa_region:
        raise ValueError("PSA source does not support joint industry and region")
    evaluation, directory = verify_experiment(root, run_id=run_id)
    # Only locally generated, sha-verified operator-owned artifacts. Never web uploads.
    model = joblib.load(directory / "model.joblib")
    taxonomy = Taxonomy(root)
    mapped = _accepted_isco_major(taxonomy, job_title)
    accepted_major = isco_major or mapped["code"]
    profile = pd.DataFrame([{
        "title": job_title.strip(),
        "skill_mentions": " ".join(sorted(set(s.casefold().strip() for s in skills if s.strip())))
        or "unspecified",
        "seniority": seniority,
        "eligibility_bucket": "PH_ELIGIBLE",
    }])
    offered_usd = max(0, float(model.predict(profile)[0]))
    sources: dict[str, dict] = {
        "jobicy_remote_offer": {
            "monthly_usd": round(offered_usd, 2),
            "monthly_php_scenario": round(offered_usd * usd_to_php, 2)
            if usd_to_php is not None else None,
            "rate_assumption_php_per_usd": usd_to_php,
            "target": "2026_remote_advertised_midpoint",
            "holdout_mae_usd_monthly": evaluation["test_selected"]["mae_usd_monthly"],
            "supported_ph_salary_ads": evaluation["test_selected"]["philippines_eligible_count"],
            "evidence": "actual_archived_ad_salaries_not_domestic_ph_wages",
            "worker_skills_not_validated_against_ad_mentions": True,
        }
    }
    if accepted_major and step_region_code and education_years is not None:
        hourly = _step_hourly(
            root, major=accepted_major, region=step_region_code, years_education=education_years
        )
        sources["step_historical"] = {
            "monthly_php_scenario": round(hourly * (40 * 52 / 12), 2),
            "predicted_hourly_php": round(hourly, 2),
            "hours_per_week_assumption": 40,
            "weeks_per_year_assumption": 52,
            "reference": "2015_2016_urban_philippines_hourly_earnings",
            "isco_major": accepted_major,
            "survey_region_code": step_region_code,
            "occupation_mapping_evidence": "explicit_major" if isco_major else mapped["evidence"],
            "not_a_current_monthly_wage": True,
        }
    if psa_occupation:
        preview = BenchmarkOnlyBaseline(root, as_of=as_of or date.today()).lookup(
            psa_occupation, industry_code=psa_industry, region_label=psa_region
        )
        if preview.observation:
            sources["psa_published_2022"] = {
                "monthly_php_scenario": float(preview.observation.amount_php_monthly),
                "source_date": "2022-08",
                "source_label": psa_occupation,
                "source_industry": psa_industry,
                "source_region": psa_region,
                "source_locator": preview.pdf_locator,
                "historical_published_mean_not_personal_prediction": True,
            }
    # Blending is a transparent hand-set scenario, never a learned causal meta-model.
    scenarios = {
        name: row["monthly_php_scenario"] for name, row in sources.items()
        if row.get("monthly_php_scenario") is not None
    }
    if "psa_published_2022" in scenarios:
        weights = {"psa_published_2022": .65, "step_historical": .30,
                   "jobicy_remote_offer": .05}
    elif "step_historical" in scenarios:
        weights = {"step_historical": .85, "jobicy_remote_offer": .15}
    else:
        weights = {"jobicy_remote_offer": 1.0}
    weights = {k: v for k, v in weights.items() if k in scenarios}
    total = sum(weights.values())
    normalized = {k: round(v / total, 6) for k, v in weights.items()}
    central = round(sum(scenarios[k] * weights[k] for k in weights) / total, 2) if total else None
    report, postings, _queue = consolidate(root, as_of=as_of or date.today())
    research = preview_career_match(
        postings,
        job_title=job_title,
        skills=skills,
        years_experience=Decimal(str(years_experience)) if years_experience is not None else None,
        current_salary_php=Decimal(str(current_salary_php)) if current_salary_php is not None else None,
        min_support=3,
    )
    forecast = None
    if central is not None and scenario_annual_growth is not None:
        forecast = {
            "method": "user_chosen_constant_annual_growth_not_trained_forecast",
            "annual_growth": scenario_annual_growth,
            "scenario_years": [
                {"year": years, "php_monthly": round(central * (1 + scenario_annual_growth) ** years, 2)}
                for years in (1, 3, 5)
            ],
        }
    return {
        "mode": "EXPERIMENTAL_LOCAL_ONLY",
        "model_artifact": directory.name,
        "profile": {
            "job_title": job_title, "skills_supplied_count": len(skills),
            "seniority_ad_proxy": seniority, "years_experience": years_experience,
            "experience_not_learned_in_jobicy_or_step_model": True,
            "isco_major": accepted_major,
        },
        "estimated_monthly_php_scenario": central,
        "scenario_label": (
            "REMOTE_OFFER_FX_ONLY_NOT_PH_DOMESTIC_ESTIMATE"
            if list(scenarios) == ["jobicy_remote_offer"] else
            "EXPERIMENTAL_HISTORICAL_AND_REMOTE_WEIGHTED_SCENARIO"
        ),
        "blend_weights_illustrative_not_learned": normalized if central is not None else None,
        "sources": sources,
        "current_salary_php_reference": current_salary_php,
        "difference_from_current_php": (
            round(central - current_salary_php, 2)
            if central is not None and current_salary_php is not None else None
        ),
        "current_salary_is_not_training_label_or_forced_floor": True,
        "skill_salary_effect_not_validated": True,
        "career_skill_research_suggestions": research["career_skill_research_suggestions"],
        "esco_occupation_skill_gap_ideas": _esco_path_suggestions(taxonomy, job_title, skills),
        "related_role_options": research["related_role_path_options"],
        "forecast": forecast,
        "calibrated_php_prediction_interval": None,
        "model_release_approved": False,
        "limitations": [
            "Prototype remote advertised-pay model trained on USD listings, not Filipino local wages",
            "STEP is historical hourly self-reported research data; PSA 2022 is aggregate historical mean",
            "Weighted combination is a user-visible scenario, NOT a validated fusion model",
            "Skills are description-text proxies, not empirically learned worker salary premiums",
            "No verified PH-specific heldout monthly PHP accuracy or calibrated range",
        ],
        "evidence_counts": {
            "jobicy_unique": report["unique_job_postings"],
            "jobicy_disclosed_salaries": report["valid_disclosed_salary_ranges"],
            "step_hourly_candidate_rows": report["step"]["verified_wage_worker_hourly_occupation_candidates"],
            "psa_2022_published_cells": report["psa_ows"]["published_cells"],
        },
    }
