from __future__ import annotations

import hmac
import os
import secrets
from decimal import Decimal
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

from salary_predictor.contracts import PredictionResult

from .experimental import SENIORITIES, parameters, worker_predict
from .inference import ApprovedInferenceService
from .profile_flow import EMPTY, draft_from, get_catalog, reference_choices, validate_profile
from .resume_extract import MAX_UPLOAD_BYTES, ResumeError, process_upload
from .resume_suggestions import suggest_profile

EDUCATION_CHOICES = (
    ("", "Not specified"),
    ("none", "No formal education"),
    ("primary", "Primary"),
    ("secondary", "Secondary"),
    ("vocational", "Vocational"),
    ("associate", "Associate degree"),
    ("bachelor", "Bachelor's degree"),
    ("master", "Master's degree"),
    ("doctorate", "Doctorate"),
)
def unavailable_result() -> PredictionResult:
    return PredictionResult.from_dict({
        "contract_version": 1,
        "mode": "UNAVAILABLE",
        "central_php_monthly": None,
        "lower_php_monthly": None,
        "upper_php_monthly": None,
        "range_basis": None,
        "basis": "none",
        "data_as_of": None,
        "occupation_specificity": "unknown",
        "region_specificity": "unknown",
        "explanation": (
            "No production-approved, comparable monthly Philippine salary model "
            "or benchmark is available yet."
        ),
        "limitations": [
            "The local research model is not enabled for website inference.",
            "Source wages and salaries cannot be treated as personal predictions.",
        ],
        "reliability_flags": ["no_approved_web_result_provider"],
        "source_id": None,
        "source_version": None,
        "artifact_version": None,
        "currency": "PHP",
        "period": "monthly",
    })


def _csrf_token() -> str:
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)
    return session["_csrf_token"]


def _check_csrf() -> None:
    supplied = request.form.get("_csrf_token", "")
    if not hmac.compare_digest(supplied, session.get("_csrf_token", "")) or not supplied:
        abort(400, description="Form expired. Refresh the page and submit again.")


def php_monthly(value: Decimal | str | int | float | None) -> str:
    if value is None:
        return "Not available"
    amount = Decimal(str(value))
    return f"₱{amount:,.2f}"


def create_app(config: dict | None = None, *, result_provider=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=os.environ.get("SALARY_PREDICTOR_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("SALARY_PREDICTOR_HTTPS") == "1",
        TEMPLATES_AUTO_RELOAD=True,
        EXPERIMENTAL_MODE=False,
    )
    if config:
        app.config.update(config)

    # Only inject an operator-owned provider. Never import a trainer in a request.
    root = str(Path(app.config.get("TAXONOMY_ROOT") or Path(__file__).resolve().parents[3]))
    app.extensions["result_provider"] = (
        result_provider if result_provider is not None
        else ApprovedInferenceService(Path(root))
    )
    catalog = get_catalog(root)
    app.extensions["taxonomy"] = catalog
    occupations, regions = reference_choices(catalog)
    app.extensions["occupation_choices"] = occupations
    app.extensions["region_choices"] = regions
    app.jinja_env.filters["php_monthly"] = php_monthly
    app.context_processor(lambda: {
        "csrf_token": _csrf_token, "education_choices": EDUCATION_CHOICES,
        "occupations": occupations, "regions": regions,
        "taxonomy_available": catalog is not None,
        "experimental_mode": app.config["EXPERIMENTAL_MODE"],
    })

    @app.after_request
    def secure_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; object-src 'none'; "
            "frame-ancestors 'none'; form-action 'self'; "
            "style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/lookup/skills")
    def skill_lookup():
        query = request.args.get("q", "").strip()
        if catalog is None or not 2 <= len(query) <= 80:
            return jsonify([])
        matches = catalog.browse("skill", query, limit=10)
        return jsonify([{"id": row["id"], "label": row["label"]} for row in matches])

    @app.get("/")
    def home():
        return render_template("home.html", title="Salary Predictor")

    @app.get("/start")
    def choose():
        return redirect(url_for("profile"))

    @app.route("/profile", methods=["GET", "POST"])
    def profile():
        if request.method == "POST":
            _check_csrf()
        draft = draft_from(request.form) if request.method == "POST" else EMPTY
        return render_template("profile.html", title="Enter profile", step=1,
                               draft=draft, errors={})

    @app.route("/review", methods=["GET", "POST"])
    def review():
        if request.method == "GET":
            return render_template("review.html", title="Review profile", step=2, draft=None)
        _check_csrf()
        raw = draft_from(request.form)
        person, overview, errors = validate_profile(raw, catalog)
        if errors:
            return render_template("profile.html", title="Fix your profile", step=1,
                                   draft=raw, errors=errors), 422
        return render_template("review.html", title="Review profile", step=2,
                               draft=raw, profile=person, overview=overview)

    @app.route("/results", methods=["GET", "POST"])
    def results():
        if request.method == "GET":
            return render_template("results.html", title="Salary result", step=3,
                                   result=unavailable_result(), preview=False)
        _check_csrf()
        raw = draft_from(request.form)
        person, _overview, errors = validate_profile(raw, catalog)
        if errors:
            return render_template("profile.html", title="Fix your profile", step=1,
                                   draft=raw, errors=errors), 422
        result = app.extensions["result_provider"](person)
        if not isinstance(result, PredictionResult):
            raise TypeError("Result provider must return a PredictionResult")
        PredictionResult.from_dict(result.to_dict())
        return render_template("results.html", title="Salary result", step=3,
                               result=result, preview=True)

    def _require_experiment():
        if not app.config["EXPERIMENTAL_MODE"] or request.remote_addr not in {
            "127.0.0.1", "::1"
        }:
            abort(404)

    @app.post("/experimental/start")
    def experimental_start():
        _require_experiment()
        _check_csrf()
        raw = draft_from(request.form)
        _profile, _overview, errors = validate_profile(raw, catalog)
        if errors:
            return render_template("profile.html", title="Fix your profile", step=1,
                                   draft=raw, errors=errors), 422
        return render_template(
            "experimental_setup.html", title="Experimental assumptions", step=3,
            draft=raw, values=dict.fromkeys((
                "usd_to_php", "current_salary_php", "seniority", "isco_major",
                "step_region_code", "education_years", "psa_occupation",
                "psa_industry", "psa_region", "scenario_annual_growth",
            ), ""), errors={}, seniorities=SENIORITIES,
        )

    @app.post("/experimental/results")
    def experimental_results():
        _require_experiment()
        _check_csrf()
        raw = draft_from(request.form)
        person, _overview, errors = validate_profile(raw, catalog)
        if errors:
            return render_template("profile.html", title="Fix your profile", step=1,
                                   draft=raw, errors=errors), 422
        params, values, param_errors = parameters(request.form)
        if param_errors:
            return render_template(
                "experimental_setup.html", title="Fix assumptions", step=3,
                draft=raw, values=values, errors=param_errors,
                seniorities=SENIORITIES,
            ), 422
        try:
            evidence = worker_predict(Path(root), {
                "job_title": raw["job_title"],
                "skills": [
                    item.strip() for item in raw["skills"].split(",") if item.strip()
                ],
                "years_experience": (
                    float(person.years_experience)
                    if person.years_experience is not None else None
                ),
                **params,
            })
        except ValueError:
            return render_template(
                "experimental_setup.html", title="Experimental calculation", step=3,
                draft=raw, values=values, errors={
                    "model": "Local experimental model unavailable. Verify the "
                             "trusted experimental artifact and try again."
                }, seniorities=SENIORITIES,
            ), 503
        return render_template(
            "experimental_results.html", title="Experimental scenario", step=3,
            report=evidence,
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html", title="Resume upload", step=1, error=None)
        _check_csrf()
        try:
            document = process_upload(request.files.get("resume"))
            draft, notes = suggest_profile(document.text, catalog)
        except ResumeError as exc:
            return render_template("upload.html", title="Resume upload", step=1,
                                   error=str(exc)), 422
        return render_template(
            "profile.html", title="Confirm resume suggestions", step=1,
            draft=draft, errors={}, upload_notes=notes,
        )

    @app.get("/how-it-works")
    @app.get("/privacy")
    def old_information_pages():
        return redirect(url_for("home"))

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def client_error(error):
        message = error.description
        if error.code == 413:
            message = "Upload exceeds the 2 MB limit. Select a smaller PDF or DOCX."
        return render_template("error.html", title="Request error",
                               status=error.code, message=message), error.code

    @app.errorhandler(500)
    def server_error(error):
        return render_template("error.html", title="Server error",
                               status=500, message="Something went wrong. Please try again."), 500

    return app
