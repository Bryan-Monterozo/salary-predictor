from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path

FEED_URL = "https://jobicy.com/jobs/feed?geo=philippines"
API_URL = "https://jobicy.com/api/v2/remote-jobs"
MAX_PAYLOAD = 8_000_000
VALID_PERIODS = {"hourly", "daily", "weekly", "monthly", "yearly", "annually"}
SKILL_PROBES = {
    "Python (computer programming)": ("python", "ccd0a1d9-afda-43d9-b901-96344886e14d"),
    "SQL": ("sql", "598de5b0-5b58-4ea7-8058-a4bc4d18c742"),
    "JavaScript": ("javascript", "3cd569a2-4f88-4c1e-9995-8dce8c5e51a7"),
}


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, value: str) -> None:
        self.parts.append(value)


def _text(html: str) -> str:
    parser = _PlainText()
    parser.feed(html)
    return " ".join(parser.parts)



def _context() -> ssl.SSLContext:
    pem = Path("/etc/ssl/cert.pem")
    return ssl.create_default_context(cafile=str(pem) if pem.is_file() else None)


def _request(url: str, accept: str) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "jobicy.com":
        raise ValueError("Jobicy archive only accepts official HTTPS host")
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "SalaryPredictorResearch/0.0.0a1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=16, context=_context()) as response:
            content = response.read(MAX_PAYLOAD + 1)
    except urllib.error.HTTPError as exc:
        # Save JSON error bodies too. It still was an API response.
        content = exc.read(MAX_PAYLOAD + 1)
        if not content.strip().startswith(b"{"):
            raise
    if len(content) > MAX_PAYLOAD:
        raise ValueError("Jobicy response exceeds archive size cap")
    return content


def _record_file(root: Path, content: bytes, *, url: str, kind: str, retrieved: str) -> dict:
    digest = hashlib.sha256(content).hexdigest()
    directory = root / "data/raw/jobicy"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = retrieved.replace("-", "").replace(":", "").replace("+00:00", "Z")
    stem = f"jobicy-{kind}-{timestamp}-{digest[:16]}"
    suffix = ".json" if kind == "api" else ".xml"
    path = directory / (stem + suffix)
    meta_path = directory / (stem + ".meta.json")
    meta = {
        "source": "jobicy",
        "api_source": "https://jobicy.com/jobs-rss-feed",
        "retrieved_utc": retrieved,
        "request_url": url,
        "response_sha256": digest,
        "raw_filename": path.name,
        "archive_type": kind,
        "training_or_redistribution_approved": False,
    }
    meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
    for file, data in ((path, content), (meta_path, meta_bytes)):
        if file.exists():
            if file.read_bytes() != data:
                raise ValueError("Existing Jobicy archive collision")
        else:
            with file.open("xb") as out:
                out.write(data)

    manifest_path = root / "configs/local_raw_file_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = {item["path"] for item in manifest["files"]}
    for file, data in ((path, content), (meta_path, meta_bytes)):
        rel = file.relative_to(root).as_posix()
        if rel not in known:
            manifest["files"].append({
                "path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data), "source_id": "jobicy_public_remote_jobs",
            })
    manifest["files"].sort(key=lambda item: item["path"])
    tempfile_path = manifest_path.with_suffix(".json.pending")
    tempfile_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tempfile_path, manifest_path)
    return {"body_path": path.relative_to(root).as_posix(),
            "metadata_path": meta_path.relative_to(root).as_posix(),
            "sha256": digest, "bytes": len(content)}


def _jobicy_url(*, kind: str, geo: str | None, count: int, cursor: str | None) -> str:
    if geo is not None and not re.fullmatch(r"[a-z0-9-]{2,60}", geo):
        raise ValueError("Invalid Jobicy geographic eligibility slug")
    params: dict[str, str] = {}
    if kind == "api":
        params["count"] = str(count)
    if geo is not None:
        params["geo"] = geo
    if cursor:
        params["cursor"] = cursor
    base = API_URL if kind == "api" else FEED_URL.split("?")[0]
    return base + "?" + urllib.parse.urlencode(params) if params else base


def fetch_archived(
    root: Path,
    *,
    kind: str = "api",
    count: int = 200,
    cursor: str | None = None,
    geo: str | None = None,
) -> dict:
    if kind not in {"api", "rss"} or not 1 <= count <= 200:
        raise ValueError("Invalid Jobicy request")
    if cursor is not None and (kind != "api" or not 1 <= len(cursor) <= 2048):
        raise ValueError("Invalid opaque pagination cursor")
    root = root.resolve()
    url = _jobicy_url(kind=kind, geo=geo, count=count, cursor=cursor)
    if cursor is None:
        archive_dir = root / "data/raw/jobicy"
        now = datetime.now(timezone.utc)
        for file in archive_dir.glob("*.meta.json") if archive_dir.is_dir() else ():
            existing = json.loads(file.read_text(encoding="utf-8"))
            if existing.get("archive_type") != kind:
                continue
            old_query = urllib.parse.parse_qs(
                urllib.parse.urlsplit(existing.get("request_url", "")).query
            )
            if "cursor" in old_query:
                continue
            prior = datetime.fromisoformat(existing["retrieved_utc"])
            if 0 <= (now - prior).total_seconds() < 3600:
                raise ValueError("New Jobicy collection more than once per hour; reuse archived data")
    payload = _request(url, "application/json" if kind == "api" else "application/rss+xml, application/xml")
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    saved = _record_file(root, payload, url=url, kind=kind, retrieved=retrieved)

    # Don't parse until the exact response is archived and hashed.
    if kind == "api":
        response = json.loads(payload)
        if not isinstance(response, dict) or response.get("success") is False:
            raise ValueError("Archived Jobicy API error response")
        if not isinstance(response.get("jobs"), list):
            raise ValueError("Archived API payload has no jobs array")
        saved["jobs_on_page"] = len(response["jobs"])
        saved["next_cursor"] = response.get("nextCursor")
    else:
        root_node = ET.fromstring(payload)
        saved["items_on_page"] = len(root_node.findall(".//item"))
    return saved


def collect_available(
    root: Path,
    *,
    target: int = 5000,
    count: int = 200,
    geo: str | None = None,
    max_pages: int = 60,
) -> dict:
    if not 1 <= target <= 100000 or not 1 <= max_pages <= 1000:
        raise ValueError("Invalid unique job collection limits")
    if not 1 <= count <= 200:
        raise ValueError("Page size must be between 1 and 200")
    cursor: str | None = None
    cursors: set[str] = set()
    fetched_ids: set[str] = set()
    page_summaries = []
    completion = "MAX_PAGES_REACHED"
    while len(page_summaries) < max_pages and len(fetched_ids) < target:
        result = fetch_archived(root, kind="api", count=count, geo=geo, cursor=cursor)
        body = (root / result["body_path"]).read_bytes()
        response = json.loads(body)
        duplicate_page_ids = 0
        for job in response["jobs"]:
            identity = str(job.get("id") or "").strip()
            if not identity:
                continue
            if identity in fetched_ids:
                duplicate_page_ids += 1
            fetched_ids.add(identity)
        page_summaries.append({
            "archive": result["body_path"],
            "jobs": len(response["jobs"]),
            "duplicate_ids_in_this_traversal": duplicate_page_ids,
        })
        next_cursor = result.get("next_cursor")
        if not next_cursor:
            completion = "SOURCE_EXHAUSTED"
            break
        if not isinstance(next_cursor, str) or next_cursor in cursors:
            raise ValueError("Jobicy returned an invalid or repeated cursor")
        cursors.add(next_cursor)
        cursor = next_cursor
    aggregate = inspect_archive(root)
    if len(fetched_ids) >= target:
        completion = "TARGET_REACHED"
    elif aggregate["distinct_job_ids"] >= target:
        completion = "HISTORICAL_ARCHIVE_TARGET_REACHED"
    return {
        "status": completion,
        "requested_unique_jobs": target,
        "unique_job_ids_this_traversal": len(fetched_ids),
        "target_shortfall": max(0, target - aggregate["distinct_job_ids"]),
        "unique_job_ids_all_archives": aggregate["distinct_job_ids"],
        "salary_disclosed_valid_range_all_archives": aggregate["salary_disclosed_valid_range"],
        "pages_archived_this_run": len(page_summaries),
        "page_summaries": page_summaries,
        "geo_filter": geo,
        "published_window": "rolling last seven days",
        "deduplication": "Jobicy listing ID, across pages and saved snapshots",
        "training_approved": False,
    }


def _salary(job: dict) -> tuple[str, str] | None:
    try:
        low = Decimal(str(job.get("salaryMin")))
        high = Decimal(str(job.get("salaryMax")))
    except (InvalidOperation, TypeError):
        return None
    currency = job.get("salaryCurrency")
    period = job.get("salaryPeriod")
    if (
        not low.is_finite() or not high.is_finite()
        or low <= 0 or high < low
        or not isinstance(currency, str) or len(currency) != 3
        or not currency.isalpha() or currency.upper() != currency
        or period not in VALID_PERIODS
    ):
        return None
    return currency, period


def inspect_archive(root: Path, *, salary_preview_limit: int | None = 20) -> dict:
    folder = root / "data/raw/jobicy"
    metadata = sorted(folder.glob("*.meta.json"), reverse=True) if folder.exists() else []
    from .geo import geo_priority, load_draft_weighting

    policy = (
        load_draft_weighting(root)
        if (root / "configs/feasibility/jobicy_weighting.json").is_file()
        else None
    )
    counts = Counter()
    chosen: dict[str, tuple[dict, dict]] = {}
    links: dict[str, str] = {}
    for meta_file in metadata:
        info = json.loads(meta_file.read_text(encoding="utf-8"))
        relative = info["raw_filename"]
        if Path(relative).name != relative:
            raise ValueError("Archive path escape")
        payload = (folder / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["response_sha256"]:
            raise ValueError("Jobicy archive checksum mismatch")
        if info["archive_type"] == "rss":
            counts["rss_snapshots"] += 1
            counts["rss_items_total"] += len(ET.fromstring(payload).findall(".//item"))
            continue
        if info["archive_type"] != "api":
            raise ValueError("Unknown Jobicy archive type")
        counts["api_snapshots"] += 1
        response = json.loads(payload)
        if response.get("success") is False or not isinstance(response.get("jobs"), list):
            counts["api_error_snapshots"] += 1
            continue
        for job in response["jobs"]:
            counts["api_entries_seen"] += 1
            id_ = str(job.get("id") or "").strip()
            link = str(job.get("url") or "").strip().split("?")[0].rstrip("/")
            if not id_:
                counts["duplicate_or_missing_ids"] += 1
                continue
            canonical_id = id_ if id_ in chosen else links.get(link) if link else None
            if canonical_id is not None:
                counts["duplicate_or_missing_ids"] += 1
                previous_job, _previous_info = chosen[canonical_id]
                # Prefer a documented salary over a later snapshot that omits it.
                if _salary(job) and not _salary(previous_job):
                    chosen[canonical_id] = (job, info)
                continue
            chosen[id_] = (job, info)
            if link:
                links[link] = id_
    currencies = Counter()
    periods = Counter()
    levels = Counter()
    skill_mentions = Counter()
    paid_skill_mentions = Counter()
    geography = Counter()
    apac_stats = Counter()
    salary_records: list[dict] = []
    for id_, (job, info) in chosen.items():
        counts["distinct_jobs"] += 1
        geo_label = str(job.get("jobGeo") or "Unknown")
        geography[geo_label] += 1
        bucket = geo_priority(geo_label, policy)["bucket"] if policy else None
        if bucket:
            apac_stats["all:" + bucket] += 1
        amount = _salary(job)
        if amount:
            counts["salary_disclosed_valid_range"] += 1
            currencies[amount[0]] += 1
            periods[amount[1]] += 1
            if bucket:
                apac_stats["salary:" + bucket] += 1
            salary_records.append({
                "job_id": id_,
                "title": str(job.get("jobTitle") or ""),
                "source_url": str(job.get("url") or "").strip().split("?")[0].rstrip("/"),
                "job_geo": geo_label,
                "salary_min": str(job["salaryMin"]),
                "salary_max": str(job["salaryMax"]),
                "currency": amount[0],
                "period": amount[1],
                "source_archive_sha256": info["response_sha256"],
                "retrieved_utc": info["retrieved_utc"],
                **({"proposed_geo_weight": geo_priority(geo_label, policy)} if policy else {}),
            })
        else:
            counts["salary_missing_or_invalid"] += 1
        if job.get("jobLevel"):
            counts["with_job_level"] += 1
            levels[str(job["jobLevel"])] += 1
        if job.get("jobGeo"):
            counts["with_location_eligibility"] += 1
        if job.get("jobTitle"):
            counts["with_title"] += 1
        if job.get("jobDescription"):
            counts["with_description"] += 1
            plain = _text(str(job["jobDescription"])).casefold()
            for label, (term, _uri_id) in SKILL_PROBES.items():
                if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", plain):
                    skill_mentions[label] += 1
                    if amount:
                        paid_skill_mentions[label] += 1
    total = counts["distinct_jobs"]
    sorted_records = sorted(salary_records, key=lambda row: row["job_id"])
    return {
        "source_id": "jobicy_public_remote_jobs",
        "api_archive_count": counts["api_snapshots"],
        "api_error_snapshots": counts["api_error_snapshots"],
        "rss_archive_count": counts["rss_snapshots"],
        "rss_items_total_not_deduplicated": counts["rss_items_total"],
        "api_entries_seen": counts["api_entries_seen"],
        "distinct_job_ids": total,
        "duplicate_or_missing_ids": counts["duplicate_or_missing_ids"],
        "salary_disclosed_valid_range": counts["salary_disclosed_valid_range"],
        "salary_absent_or_invalid": counts["salary_missing_or_invalid"],
        "salary_disclosure_rate": round(
            counts["salary_disclosed_valid_range"] / total, 4
        ) if total else None,
        "currency_counts": dict(sorted(currencies.items())),
        "job_geo_distribution": dict(sorted(geography.items())),
        "geo_priority_counts": dict(sorted(apac_stats.items())),
        "geo_priority_policy": "DRAFT, not a salary effect, not training approval",
        "salary_range_preview": (
            sorted_records[:salary_preview_limit]
            if salary_preview_limit is not None else sorted_records
        ),
        "total_valid_salary_records": len(salary_records),
        "period_counts": dict(sorted(periods.items())),
        "with_job_level": counts["with_job_level"],
        "with_description": counts["with_description"],
        "with_location_eligibility": counts["with_location_eligibility"],
        "seniority_categories": dict(sorted(levels.items())),
        "philippines_local_jobs_confirmed": None,
        "geography_caveat": "Unfiltered remote ads are global; jobGeo is eligibility, not pay market",
        "verified_esco_seed_skill_mentions": dict(sorted(skill_mentions.items())),
        "esco_seed_mentions_on_salary_disclosing_ads": dict(
            sorted(paid_skill_mentions.items())
        ),
        "skills": "Small URI-backed text probes are not verified job requirements or skill premiums",
        "eligibility": "READ_ONLY_ARCHIVE_NOT_TRAINING_APPROVED",
        "training_or_deployment_approved": False,
    }


def export_unique_salary_ranges(root: Path) -> dict:
    root = root.resolve(strict=True)
    report = inspect_archive(root, salary_preview_limit=None)
    records = report["salary_range_preview"]
    payload = {
        "source_id": "jobicy_public_remote_jobs",
        "cohort": "remote_advertised_salary_not_employee_earnings",
        "training_approved": False,
        "source_licenses_for_ml_unverified": True,
        "source_records_deduplicated_by_id_and_url": True,
        "valid_salary_record_count": len(records),
        "records": records,
    }
    body = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(body).hexdigest()
    folder = root / "data/audit"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"jobicy-valid-salary-ranges-{digest[:16]}.json"
    try:
        with destination.open("xb") as file:
            file.write(body)
    except FileExistsError:
        if destination.read_bytes() != body:
            raise ValueError("Existing salary-range export differs")
    return {
        "path": destination.relative_to(root).as_posix(),
        "sha256": digest,
        "unique_valid_salary_ranges": len(records),
        "training_approved": False,
    }
