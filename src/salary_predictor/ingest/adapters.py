from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from salary_predictor.contracts import (
    BenchmarkObservation,
    ContractError,
    SalaryObservation,
    SourceManifest,
    TaxonomyConcept,
)
from salary_predictor.taxonomy.catalog import _esco_rows

_OCCUPATION = re.compile(
    r"^\s*([0-9]{4}b?)\s{2,}(.+?)\s{2,}([0-9]{1,3}(?:,[0-9]{3})+)\s*$"
)
_INDUSTRY = re.compile(
    r"^\s*([A-Z][0-9]{2,5}(?:/[A-Z][0-9]{2,5})*"
    r"(?:\s+excl\.\s+[A-Z][0-9]{2,5})?)\s{2,}(.+?)\s*$"
)
_REGION_ROW = re.compile(
    r"^\s*(PHILIPPINES|National Capital Region|Region [IVX]+(?:-[AB])? - [A-Za-z ]+"
    r"|Caraga|Cordillera Administrative Region|Bangsamoro Autonomous Region"
    r" in Muslim Mindanao)\s{2,}([0-9]{1,3}(?:,[0-9]{3})+)"
    r"\s+([0-9]{1,3}(?:,[0-9]{3})+)\s*$"
)


@dataclass
class Batch:
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scanned: int = 0

    def accept(self, record: dict, locator: str, **details: str | int) -> None:
        self.accepted.append({"record": record, "locator": locator, **details})

    def reject(self, locator: str, reason: str) -> None:
        self.rejected.append({"locator": locator, "reason": reason})


def _benchmark(
    manifest: SourceManifest,
    *,
    table: str,
    page: int,
    row: int,
    scope: tuple[str, ...],
    occupation: str,
    industry: str | None = None,
    industry_label: str | None = None,
    region_label: str | None = None,
    amount: str,
) -> dict:
    identifier = f"{manifest.source_id}-{table.replace(' ', '')}-p{page}-r{row}"
    data = {
        "contract_version": 1,
        "observation_id": identifier,
        "source_id": manifest.source_id,
        "source_version": manifest.source_version,
        "reference_date": manifest.reference_date,
        "source_table": table,
        "source_page": page,
        "population": manifest.population,
        "published_dimensions": list(scope),
        "occupation_psoc_code": None,
        "occupation_label": occupation,
        "region_psgc_code": None,
        "region_label": region_label,
        "industry_code": industry,
        "industry_label": industry_label,
        "metric": "mean_monthly_wage",
        "pay_component": "total",
        "tax_basis": "unknown",
        "currency": "PHP",
        "period": "monthly",
        "amount_php_monthly": amount.replace(",", ""),
    }
    return BenchmarkObservation.from_dict(data).to_dict()


def parse_ows_2022(path: Path, manifest: SourceManifest) -> Batch:
    if manifest.dataset_role != "aggregate_benchmark" or manifest.file_format != "pdf":
        raise ValueError("OWS PDF adapter needs an aggregate-benchmark PDF manifest")
    if manifest.reference_date != "2022-08":
        raise ValueError("This adapter only supports the checked 2022 OWS release")

    from pypdf import PdfReader

    reader = PdfReader(path)
    if len(reader.pages) != 58:
        raise ValueError("Unexpected OWS page count; review the PDF version")
    result = Batch()
    industry: str | None = None
    industry_label: str | None = None
    for page_number in range(1, 14):
        lines = reader.pages[page_number - 1].extract_text(extraction_mode="layout").splitlines()
        if not any("Table 1. Average Monthly Wage Rates" in line for line in lines):
            raise ValueError(f"Table 1 is missing on PDF page {page_number}")
        for index, line in enumerate(lines, 1):
            header = _INDUSTRY.fullmatch(line)
            if header:
                industry, industry_label = header.groups()
                continue
            match = _OCCUPATION.fullmatch(line)
            if not match:
                continue
            original_code, title, amount = match.groups()
            result.scanned += 1
            locator = f"page:{page_number}/line:{index}"
            if not industry:
                result.reject(locator, "Missing industry header")
                continue
            if original_code.endswith("b"):
                result.warnings.append(
                    f"{locator}: OWS-only occupation code {original_code}; no PSOC mapping"
                )
            try:
                record = _benchmark(
                    manifest,
                    table="Table 1",
                    page=page_number,
                    row=index,
                    scope=("occupation", "industry"),
                    occupation=title.strip(),
                    industry=industry,
                    industry_label=industry_label,
                    amount=amount,
                )
                result.accept(record, locator, original_code=original_code)
            except ContractError as exc:
                result.reject(locator, f"Invalid published cell: {exc.field}")

    page_number = 17
    lines = reader.pages[page_number - 1].extract_text(extraction_mode="layout").splitlines()
    if not any("Table 3. Average Monthly Wage Rates" in line for line in lines):
        raise ValueError("Regional benchmark table is missing on page 17")
    for index, line in enumerate(lines, 1):
        row = _REGION_ROW.fullmatch(line)
        if not row:
            continue
        name, clerks, elementary = row.groups()
        name = name.strip()
        for title, amount in (
            ("Accounting and Bookkeeping Clerks", clerks),
            ("Elementary Occupations (Unskilled Workers)", elementary),
        ):
            result.scanned += 1
            locator = f"page:17/line:{index}/{title}"
            try:
                record = _benchmark(
                    manifest,
                    table="Table 3",
                    page=17,
                    row=index,
                    scope=("occupation",) if name == "PHILIPPINES" else ("occupation", "region"),
                    occupation=title,
                    region_label=None if name == "PHILIPPINES" else name,
                    amount=amount,
                )
                record["observation_id"] += (
                    "-accounting" if title.startswith("Accounting") else "-elementary"
                )
                result.accept(record, locator)
            except ContractError as exc:
                result.reject(locator, f"Invalid published cell: {exc.field}")
    return result


def parse_esco(path: Path, manifest: SourceManifest) -> Batch:
    if manifest.dataset_role != "taxonomy" or manifest.file_format != "csv":
        raise ValueError("ESCO adapter needs a taxonomy CSV manifest")
    if path.name not in {"occupations_en.csv", "skills_en.csv", "occupationSkillRelations_en.csv"}:
        raise ValueError("Unsupported ESCO table")
    result = Batch()
    csv.field_size_limit(sys.maxsize)
    if path.name == "occupationSkillRelations_en.csv":
        raise ValueError("Relationship table requires the dedicated ESCO relation adapter")
    kind = "occupation" if path.name == "occupations_en.csv" else "skill"
    items = _esco_rows(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        count = sum(1 for _ in csv.DictReader(file))
    result.scanned = count
    result.warnings.append(f"{count - len(items)} duplicate revisions collapsed by conceptUri")
    for index, (uri, row) in enumerate(sorted(items.items()), 1):
        try:
            record = TaxonomyConcept.from_dict({
                "contract_version": 1,
                "source_id": manifest.source_id,
                "concept_uri": uri,
                "kind": kind,
                "label": row["preferredLabel"].strip(),
                "code": row.get("code", "").strip() or None,
            })
            result.accept(record.to_dict(), f"concept:{uri}", original_count=index)
        except (ContractError, KeyError) as exc:
            result.reject(f"concept:{uri}", f"Invalid {kind} concept: {type(exc).__name__}")
    return result


def parse_esco_relations(path: Path, manifest: SourceManifest, root: Path) -> Batch:
    if (
        manifest.dataset_role != "taxonomy"
        or manifest.file_format != "csv"
        or path.name != "occupationSkillRelations_en.csv"
    ):
        raise ValueError("ESCO relationship adapter needs the relationship CSV")
    occupations = _esco_rows(root / "data/raw/esco/occupations_en.csv")
    skills = _esco_rows(root / "data/raw/esco/skills_en.csv")
    result = Batch()
    csv.field_size_limit(sys.maxsize)
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        for line, row in enumerate(csv.DictReader(file), 2):
            result.scanned += 1
            locator = f"csv-row:{line}"
            occ, skill, relation = (
                row.get("occupationUri", ""), row.get("skillUri", ""),
                row.get("relationType", ""),
            )
            if occ not in occupations or skill not in skills or relation not in {
                "essential", "optional"
            }:
                result.reject(locator, "Unknown occupation/skill URI or relation type")
                continue
            identity = (occ, skill, relation)
            if identity in seen:
                result.reject(locator, "Duplicate occupation-skill relationship")
                continue
            seen.add(identity)
            result.accept({
                "source_id": manifest.source_id,
                "occupation_uri": occ,
                "skill_uri": skill,
                "relation_type": relation,
            }, locator)
    return result


def parse_psoc_xlsx(path: Path, manifest: SourceManifest) -> Batch:
    if manifest.dataset_role != "taxonomy" or manifest.file_format != "xlsx":
        raise ValueError("PSOC importer needs a taxonomy XLSX manifest")
    from salary_predictor.taxonomy.psoc_workbook import load_psoc_groups

    result = Batch()
    for group in load_psoc_groups(path, manifest.file_sha256):
        result.scanned += 1
        try:
            concept = TaxonomyConcept.from_dict({
                "contract_version": 1,
                "source_id": manifest.source_id,
                "concept_uri": f"urn:psoc:local:{group.level}:{group.code}",
                "kind": "occupation_group",
                "label": group.label,
                "code": group.code,
            })
            result.accept({
                "concept": concept.to_dict(),
                "parent_code": group.parent_code,
                "major_group": group.major_group,
                "level": group.level,
                "source_sheet": group.source_sheet,
                "source_row": group.source_row,
                "psoc_1992_relation": group.psoc_1992_relation,
                "isco_2008_relation": group.isco_2008_relation,
            }, f"{group.source_sheet}/row:{group.source_row}")
        except ContractError as exc:
            result.reject(
                f"{group.source_sheet}/row:{group.source_row}",
                f"Invalid occupation group: {exc.field}",
            )
    return result


def _money(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (ValueError, ArithmeticError) as exc:
        raise ValueError("Invalid salary number") from exc
    if not result.is_finite() or result <= 0 or result > 100_000_000:
        raise ValueError("Amount outside supported review bounds")
    if result.as_tuple().exponent < -2:
        raise ValueError("Money has more than two decimal places")
    return result


def normalize_monthly(
    record: SalaryObservation, factors: dict[str, str | int | bool],
) -> dict:
    if record.currency != "PHP":
        raise ValueError("Only PHP salaries can be compared without approved FX rules")
    if record.tax_basis != "gross" or record.pay_component != "total":
        raise ValueError("Only documented gross total-pay observations are comparable")
    if record.thirteenth_month_included is not False:
        raise ValueError("13th-month inclusion must be known and excluded")
    if record.allowances_included is None:
        raise ValueError("Allowance basis is unknown")
    factor_names = {
        "hourly": "paid_hours_per_month",
        "daily": "paid_days_per_month",
        "weekly": "paid_weeks_per_month",
        "annual": "months_per_year",
    }
    if record.original_period == "monthly":
        multiplier = Decimal("1")
    elif record.original_period in factor_names:
        field = factor_names[record.original_period]
        if field not in factors:
            raise ValueError(f"Missing approved factor {field}")
        value = factors[field]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Invalid conversion factor")
        multiplier = _money(str(value))
        if record.original_period == "annual":
            multiplier = Decimal("1") / multiplier
    else:
        raise ValueError("Unknown or unsupported pay period")

    if record.record_type == "advertised_salary":
        if factors.get("posted_target") != "midpoint":
            raise ValueError("Advertised range needs explicit midpoint target choice")
        maximum = record.original_amount_max
        minimum = record.original_amount_min
        original_target = (minimum + maximum) / 2 if maximum is not None else minimum
        calculation = "advertised_midpoint_not_employee_earnings"
    else:
        if record.original_amount_max is not None:
            raise ValueError("Employee record requires one observed salary, not a range")
        original_target = record.original_amount_min
        calculation = "observed_worker_salary"

    final = (original_target * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if final <= 0 or final > 100_000_000:
        raise ValueError("Converted monthly target outside review bounds")
    return {
        "monthly_php": str(final),
        "cohort": record.record_type,
        "calculation": calculation,
        "assumptions": {
            "original_period": record.original_period,
            "factor": str(multiplier),
            "allowances_included": record.allowances_included,
            "thirteenth_month_included": record.thirteenth_month_included,
            "tax_basis": record.tax_basis,
            "pay_component": record.pay_component,
        },
    }


def parse_salary_csv(path: Path, manifest: SourceManifest) -> Batch:
    if manifest.dataset_role not in {"individual_earnings", "advertised_salary"}:
        raise ValueError("Salary adapter needs a separately approved salary cohort")
    if manifest.file_format not in {"csv", "tsv"}:
        raise ValueError("Salary adapter requires CSV or TSV source")
    if not manifest.column_mapping:
        raise ValueError("A reviewed salary column mapping is required")
    if path.suffix.lower() != "." + manifest.file_format:
        raise ValueError("Salary file format mismatch")

    result = Batch()
    csv.field_size_limit(sys.maxsize)
    delimiter = "\t" if manifest.file_format == "tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("No salary dataset header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Repeated source CSV column names")
        if not set(manifest.column_mapping.values()).issubset(set(reader.fieldnames)):
            raise ValueError("Missing approved salary source columns")

        seen: set[str] = set()
        for number, values in enumerate(reader, 2):
            result.scanned += 1
            locator = f"csv-row:{number}"
            if None in values:
                result.reject(locator, "Extra CSV columns")
                continue
            mapped = {key: values[column] for key, column in manifest.column_mapping.items()}
            try:
                if mapped.get("record_type") != manifest.dataset_role:
                    raise ValueError("Mixed worker and job-advertised salary cohorts")
                identifier = mapped.get("observation_id", "")
                if identifier in seen:
                    raise ValueError("Repeated salary observation ID")
                seen.add(identifier)
                mapped["contract_version"] = 1
                for nullable in (
                    "original_amount_max", "occupation_psoc_code", "region_psgc_code",
                    "derivation_notes",
                ):
                    if mapped.get(nullable) == "":
                        mapped[nullable] = None
                for field in ("thirteenth_month_included", "allowances_included"):
                    mapped[field] = {"true": True, "false": False, "": None}.get(
                        mapped.get(field, "").strip().lower(), "invalid"
                    )
                observed_fields = (
                    "occupation_psoc_code", "region_psgc_code",
                    "years_experience", "education_level", "normalized_skills",
                )
                completeness = {}
                for name in observed_fields:
                    observed = mapped.pop("observed_" + name, "")
                    if observed not in {"true", "false"}:
                        raise ValueError(f"Predictor evidence flag unknown: {name}")
                    completeness[name] = observed == "true"
                mapped["predictor_completeness"] = completeness
                source = SalaryObservation.from_dict(mapped)
                _money(str(source.original_amount_min))
                if source.original_amount_max is not None:
                    _money(str(source.original_amount_max))
                if source.source_id != manifest.source_id:
                    raise ValueError("Source ID does not match approved manifest")
                normal = normalize_monthly(source, dict(manifest.parser_options))
                result.accept({
                    "salary_observation": source.to_dict(),
                    "normalized": normal,
                }, locator)
            except (ContractError, ValueError, TypeError, KeyError) as exc:
                result.reject(locator, f"Rejected salary row: {exc}")
    return result
