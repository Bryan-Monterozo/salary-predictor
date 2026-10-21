from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .common import (
    CONTRACT_VERSION,
    EDUCATION_LEVELS,
    FEATURE_SCHEMA_VERSION,
    PAY_COMPONENTS,
    PERIODS,
    RECORD_TYPES,
    SOURCE_ROLES,
    TAX_BASES,
    ContractError,
    amount_range,
    as_of,
    check_fields,
    check_version,
    choice,
    decimal,
    decimal_json,
    identifier,
    ph_code,
    text,
    text_list,
)

PROFILE_FIELDS = {
    "feature_schema_version",
    "occupation_psoc_code",
    "region_psgc_code",
    "years_experience",
    "education_level",
    "normalized_skills",
}
PREDICTOR_FIELDS = frozenset(PROFILE_FIELDS - {"feature_schema_version"})
FILE_FORMATS = frozenset({"csv", "tsv", "xlsx", "jsonl", "parquet", "pdf"})
BENCHMARK_DIMENSIONS = frozenset({"occupation", "industry", "region"})


@dataclass(frozen=True, slots=True)
class UserProfile:
    occupation_psoc_code: str | None
    region_psgc_code: str | None
    years_experience: Decimal | None
    education_level: str | None
    normalized_skills: tuple[str, ...]
    industry_code: str | None = None
    employment_context: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> UserProfile:
        data = check_fields(data, PROFILE_FIELDS, {"industry_code", "employment_context"}, "profile")
        check_version(data["feature_schema_version"], "feature_schema_version")
        years = decimal(data["years_experience"], "years_experience", nullable=True, zero_ok=True)
        if years is not None and years > 70:
            raise ContractError("years_experience", "cannot exceed 70 years")
        return cls(
            occupation_psoc_code=ph_code(
                data["occupation_psoc_code"], "occupation_psoc_code", nullable=True
            ),
            region_psgc_code=ph_code(
                data["region_psgc_code"], "region_psgc_code", nullable=True
            ),
            years_experience=years,
            education_level=choice(
                data["education_level"], "education_level", EDUCATION_LEVELS, nullable=True
            ),
            normalized_skills=text_list(data["normalized_skills"], "normalized_skills"),
            industry_code=identifier(data.get("industry_code"), "industry_code", nullable=True),
            employment_context=choice(
                data.get("employment_context"),
                "employment_context",
                frozenset({"full_time", "part_time", "contract", "self_employed", "unknown"}),
                nullable=True,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "occupation_psoc_code": self.occupation_psoc_code,
            "region_psgc_code": self.region_psgc_code,
            "years_experience": decimal_json(self.years_experience),
            "education_level": self.education_level,
            "normalized_skills": list(self.normalized_skills),
            "industry_code": self.industry_code,
            "employment_context": self.employment_context,
        }


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source_id: str
    source_version: str
    source_url: str
    local_path: str
    file_sha256: str
    dataset_role: str
    file_format: str
    rights_status: str
    approved_uses: tuple[str, ...]
    reference_date: str
    observation_unit: str
    population: str
    compensation_basis: str | None
    column_mapping: Mapping[str, str]
    parser_options: Mapping[str, str | int | bool]
    table_id: str | None = None
    sheet_name: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> SourceManifest:
        fields = {
            "contract_version", "source_id", "source_version", "source_url", "local_path",
            "file_sha256", "dataset_role", "file_format", "rights_status", "approved_uses",
            "reference_date", "observation_unit", "population", "compensation_basis",
            "column_mapping", "parser_options",
        }
        data = check_fields(data, fields, {"table_id", "sheet_name"}, "source_manifest")
        check_version(data["contract_version"])
        path = text(data["local_path"], "local_path")
        assert path is not None
        normalized = PurePosixPath(path)
        if (
            path.startswith("/")
            or "\\" in path
            or ".." in normalized.parts
            or normalized.parts[:2] != ("data", "raw")
        ):
            raise ContractError("local_path", "must stay under data/raw")
        sha = text(data["file_sha256"], "file_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
            raise ContractError("file_sha256", "expected lowercase SHA-256")
        mapping = data["column_mapping"]
        if not isinstance(mapping, dict):
            raise ContractError("column_mapping", "expected a field mapping object")
        for key, value in mapping.items():
            text(key, "column_mapping key")
            text(value, f"column_mapping.{key}")
        options = data["parser_options"]
        if not isinstance(options, dict) or any(
            not isinstance(key, str)
            or not key
            or type(value) not in (str, int, bool)
            for key, value in options.items()
        ):
            raise ContractError("parser_options", "expected simple named parser settings")
        source_url = text(data["source_url"], "source_url")
        if not source_url.startswith(("https://", "http://")):
            raise ContractError("source_url", "expected an HTTP(S) link")
        return cls(
            source_id=identifier(data["source_id"], "source_id"),
            source_version=text(data["source_version"], "source_version"),
            source_url=source_url,
            local_path=path,
            file_sha256=sha,
            dataset_role=choice(data["dataset_role"], "dataset_role", SOURCE_ROLES),
            file_format=choice(data["file_format"], "file_format", FILE_FORMATS),
            rights_status=text(data["rights_status"], "rights_status"),
            approved_uses=text_list(data["approved_uses"], "approved_uses"),
            reference_date=as_of(data["reference_date"], "reference_date"),
            observation_unit=text(data["observation_unit"], "observation_unit"),
            population=text(data["population"], "population"),
            compensation_basis=text(
                data["compensation_basis"], "compensation_basis", nullable=True
            ),
            column_mapping=MappingProxyType(dict(mapping)),
            parser_options=MappingProxyType(dict(options)),
            table_id=text(data.get("table_id"), "table_id", nullable=True),
            sheet_name=text(data.get("sheet_name"), "sheet_name", nullable=True),
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "file_sha256": self.file_sha256,
            "dataset_role": self.dataset_role,
            "file_format": self.file_format,
            "rights_status": self.rights_status,
            "approved_uses": list(self.approved_uses),
            "reference_date": self.reference_date,
            "observation_unit": self.observation_unit,
            "population": self.population,
            "compensation_basis": self.compensation_basis,
            "column_mapping": dict(self.column_mapping),
            "parser_options": dict(self.parser_options),
            "table_id": self.table_id,
            "sheet_name": self.sheet_name,
        }


@dataclass(frozen=True, slots=True)
class SalaryObservation:
    observation_id: str
    source_id: str
    record_type: str
    currency: str
    original_amount_min: Decimal
    original_amount_max: Decimal | None
    original_period: str
    target_scope: str
    pay_component: str
    tax_basis: str
    thirteenth_month_included: bool | None
    allowances_included: bool | None
    observation_date: str
    occupation_psoc_code: str | None
    region_psgc_code: str | None
    predictor_completeness: Mapping[str, bool]
    derivation_notes: str | None

    @classmethod
    def from_dict(cls, data: Any) -> SalaryObservation:
        fields = {
            "contract_version", "observation_id", "source_id", "record_type", "currency",
            "original_amount_min", "original_amount_max", "original_period", "target_scope",
            "pay_component", "tax_basis", "thirteenth_month_included",
            "allowances_included", "observation_date", "occupation_psoc_code",
            "region_psgc_code", "predictor_completeness", "derivation_notes",
        }
        data = check_fields(data, fields, set(), "salary_observation")
        check_version(data["contract_version"])
        record_type = choice(data["record_type"], "record_type", RECORD_TYPES)
        scope = choice(
            data["target_scope"], "target_scope",
            frozenset({"individual_worker", "job_posting"}),
        )
        if (record_type == "individual_earnings") != (scope == "individual_worker"):
            raise ContractError("target_scope", "must match the salary record type")
        currency = text(data["currency"], "currency")
        if currency != "unknown" and not re.fullmatch(r"[A-Z]{3}", currency or ""):
            raise ContractError("currency", "expected uppercase ISO currency or unknown")
        completeness = data["predictor_completeness"]
        if not isinstance(completeness, dict) or set(completeness) != PREDICTOR_FIELDS:
            raise ContractError("predictor_completeness", "must list all five profile predictors")
        if any(type(value) is not bool for value in completeness.values()):
            raise ContractError("predictor_completeness", "expected true/false for each predictor")
        for name in ("thirteenth_month_included", "allowances_included"):
            if data[name] is not None and type(data[name]) is not bool:
                raise ContractError(name, "expected true, false or null")
        low, high = amount_range(
            data["original_amount_min"], data["original_amount_max"],
            "original_amount_min", "original_amount_max",
        )
        return cls(
            observation_id=identifier(data["observation_id"], "observation_id"),
            source_id=identifier(data["source_id"], "source_id"),
            record_type=record_type,
            currency=currency,
            original_amount_min=low,
            original_amount_max=high,
            original_period=choice(data["original_period"], "original_period", PERIODS),
            target_scope=scope,
            pay_component=choice(data["pay_component"], "pay_component", PAY_COMPONENTS),
            tax_basis=choice(data["tax_basis"], "tax_basis", TAX_BASES),
            thirteenth_month_included=data["thirteenth_month_included"],
            allowances_included=data["allowances_included"],
            observation_date=as_of(data["observation_date"], "observation_date"),
            occupation_psoc_code=ph_code(
                data["occupation_psoc_code"], "occupation_psoc_code", nullable=True
            ),
            region_psgc_code=ph_code(
                data["region_psgc_code"], "region_psgc_code", nullable=True
            ),
            predictor_completeness=MappingProxyType(dict(completeness)),
            derivation_notes=text(data["derivation_notes"], "derivation_notes", nullable=True),
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "record_type": self.record_type,
            "currency": self.currency,
            "original_amount_min": decimal_json(self.original_amount_min),
            "original_amount_max": decimal_json(self.original_amount_max),
            "original_period": self.original_period,
            "target_scope": self.target_scope,
            "pay_component": self.pay_component,
            "tax_basis": self.tax_basis,
            "thirteenth_month_included": self.thirteenth_month_included,
            "allowances_included": self.allowances_included,
            "observation_date": self.observation_date,
            "occupation_psoc_code": self.occupation_psoc_code,
            "region_psgc_code": self.region_psgc_code,
            "predictor_completeness": dict(self.predictor_completeness),
            "derivation_notes": self.derivation_notes,
        }
