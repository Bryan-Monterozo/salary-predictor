from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .common import (
    CONTRACT_VERSION,
    PAY_COMPONENTS,
    RESULT_MODES,
    TAX_BASES,
    ContractError,
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

RANGE_BASES = frozenset({"published_source_range", "calibrated_model_interval"})
BENCHMARK_DIMENSIONS = frozenset({"occupation", "industry", "region"})
BENCHMARK_METRICS = frozenset(
    {"mean_monthly_wage", "median_basic_pay", "median_allowance"}
)
SPECIFICITY = frozenset({"exact", "group", "national", "unknown"})
RESULT_BASIS = frozenset(
    {"model_prediction", "published_average_monthly_wage", "published_median_basic_pay", "none"}
)


@dataclass(frozen=True, slots=True)
class TaxonomyConcept:
    source_id: str
    concept_uri: str
    kind: str
    label: str
    code: str | None

    @classmethod
    def from_dict(cls, data: Any) -> TaxonomyConcept:
        data = check_fields(
            data,
            {"contract_version", "source_id", "concept_uri", "kind", "label", "code"},
            set(),
            "taxonomy_concept",
        )
        check_version(data["contract_version"])
        return cls(
            source_id=identifier(data["source_id"], "source_id"),
            concept_uri=text(data["concept_uri"], "concept_uri"),
            kind=choice(
                data["kind"], "kind",
                frozenset({"occupation", "skill", "occupation_group", "skill_group"}),
            ),
            label=text(data["label"], "label"),
            code=text(data["code"], "code", nullable=True),
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "source_id": self.source_id,
            "concept_uri": self.concept_uri,
            "kind": self.kind,
            "label": self.label,
            "code": self.code,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    observation_id: str
    source_id: str
    source_version: str
    reference_date: str
    source_table: str
    source_page: int | None
    population: str
    published_dimensions: tuple[str, ...]
    occupation_psoc_code: str | None
    occupation_label: str | None
    region_psgc_code: str | None
    region_label: str | None
    industry_code: str | None
    industry_label: str | None
    metric: str
    pay_component: str
    tax_basis: str
    amount_php_monthly: Decimal

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkObservation:
        required = {
            "contract_version", "observation_id", "source_id", "source_version", "reference_date",
            "source_table", "source_page", "population", "published_dimensions",
            "occupation_psoc_code", "occupation_label", "region_psgc_code", "region_label",
            "industry_code", "industry_label", "metric", "pay_component", "tax_basis",
            "currency", "period", "amount_php_monthly",
        }
        data = check_fields(data, required, set(), "benchmark_observation")
        check_version(data["contract_version"])
        if data["currency"] != "PHP" or data["period"] != "monthly":
            raise ContractError("currency/period", "benchmark must use published monthly PHP")
        dimensions = text_list(data["published_dimensions"], "published_dimensions")
        if not dimensions or not set(dimensions).issubset(BENCHMARK_DIMENSIONS):
            raise ContractError("published_dimensions", "unsupported table dimensions")

        occupation = ph_code(data["occupation_psoc_code"], "occupation_psoc_code", nullable=True)
        occupation_label = text(data["occupation_label"], "occupation_label", nullable=True)
        region = ph_code(data["region_psgc_code"], "region_psgc_code", nullable=True)
        region_label = text(data["region_label"], "region_label", nullable=True)
        industry = text(data["industry_code"], "industry_code", nullable=True)
        industry_label = text(data["industry_label"], "industry_label", nullable=True)
        for dimension, present in (
            ("occupation", occupation is not None or occupation_label is not None),
            ("region", region is not None or region_label is not None),
            ("industry", industry is not None or industry_label is not None),
        ):
            if (dimension in dimensions) != present:
                raise ContractError(
                    "published_dimensions", f"{dimension} must match the source table dimensions"
                )

        expected_component = {
            "mean_monthly_wage": "total",
            "median_basic_pay": "basic",
            "median_allowance": "allowance",
        }
        if data["pay_component"] != expected_component.get(data["metric"]):
            raise ContractError("pay_component", "does not match the published wage measure")

        page = data["source_page"]
        if page is not None and (type(page) is not int or page < 1):
            raise ContractError("source_page", "expected a positive page number or null")
        return cls(
            observation_id=identifier(data["observation_id"], "observation_id"),
            source_id=identifier(data["source_id"], "source_id"),
            source_version=text(data["source_version"], "source_version"),
            reference_date=as_of(data["reference_date"], "reference_date"),
            source_table=text(data["source_table"], "source_table"),
            source_page=page,
            population=text(data["population"], "population"),
            published_dimensions=dimensions,
            occupation_psoc_code=occupation,
            occupation_label=occupation_label,
            region_psgc_code=region,
            region_label=region_label,
            industry_code=industry,
            industry_label=industry_label,
            metric=choice(data["metric"], "metric", BENCHMARK_METRICS),
            pay_component=choice(data["pay_component"], "pay_component", PAY_COMPONENTS),
            tax_basis=choice(data["tax_basis"], "tax_basis", TAX_BASES),
            amount_php_monthly=decimal(data["amount_php_monthly"], "amount_php_monthly"),
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "reference_date": self.reference_date,
            "source_table": self.source_table,
            "source_page": self.source_page,
            "population": self.population,
            "published_dimensions": list(self.published_dimensions),
            "occupation_psoc_code": self.occupation_psoc_code,
            "occupation_label": self.occupation_label,
            "region_psgc_code": self.region_psgc_code,
            "region_label": self.region_label,
            "industry_code": self.industry_code,
            "industry_label": self.industry_label,
            "metric": self.metric,
            "pay_component": self.pay_component,
            "tax_basis": self.tax_basis,
            "currency": "PHP",
            "period": "monthly",
            "amount_php_monthly": decimal_json(self.amount_php_monthly),
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    mode: str
    central_php_monthly: Decimal | None
    lower_php_monthly: Decimal | None
    upper_php_monthly: Decimal | None
    range_basis: str | None
    basis: str
    data_as_of: str | None
    occupation_specificity: str
    region_specificity: str
    explanation: str
    limitations: tuple[str, ...]
    reliability_flags: tuple[str, ...]
    source_id: str | None
    source_version: str | None
    artifact_version: str | None

    @classmethod
    def from_dict(cls, data: Any) -> PredictionResult:
        fields = {
            "contract_version", "mode", "central_php_monthly", "lower_php_monthly",
            "upper_php_monthly", "range_basis", "basis", "data_as_of",
            "occupation_specificity", "region_specificity", "explanation",
            "limitations", "reliability_flags", "source_id", "source_version",
            "artifact_version", "currency", "period",
        }
        data = check_fields(data, fields, set(), "prediction_result")
        check_version(data["contract_version"])
        if data["currency"] != "PHP" or data["period"] != "monthly":
            raise ContractError("currency/period", "result must be monthly PHP")
        mode = choice(data["mode"], "mode", RESULT_MODES)
        basis = choice(data["basis"], "basis", RESULT_BASIS)
        central = decimal(data["central_php_monthly"], "central_php_monthly", nullable=True)
        lower = decimal(data["lower_php_monthly"], "lower_php_monthly", nullable=True)
        upper = decimal(data["upper_php_monthly"], "upper_php_monthly", nullable=True)
        range_basis = choice(data["range_basis"], "range_basis", RANGE_BASES, nullable=True)
        data_as_of = as_of(data["data_as_of"], "data_as_of", nullable=True)
        source = identifier(data["source_id"], "source_id", nullable=True)
        source_version = text(data["source_version"], "source_version", nullable=True)
        artifact = text(data["artifact_version"], "artifact_version", nullable=True)

        if (lower is None) != (upper is None):
            raise ContractError("range", "both ends must be provided together")
        if lower is not None:
            if central is None or not (lower <= central <= upper):
                raise ContractError("range", "must contain the central value")
            if range_basis is None:
                raise ContractError("range_basis", "range needs a documented basis")
        elif range_basis is not None:
            raise ContractError("range_basis", "no range supplied")

        if mode == "UNAVAILABLE":
            if (
                any(
                    value is not None
                    for value in (central, source, source_version, artifact, data_as_of)
                )
                or lower is not None or basis != "none"
            ):
                raise ContractError("mode", "unavailable cannot show a salary or claim a source")
        elif mode == "MARKET_BENCHMARK":
            if (
                central is None or source is None or source_version is None
                or data_as_of is None or artifact is not None
                or basis not in {"published_average_monthly_wage", "published_median_basic_pay"}
                or range_basis == "calibrated_model_interval"
            ):
                raise ContractError("mode", "benchmark needs a date, source and published pay basis")
        elif (
            central is None or artifact is None or data_as_of is None
            or source is not None or source_version is not None
            or basis != "model_prediction" or range_basis == "published_source_range"
        ):
            raise ContractError("mode", "model result needs a validated artifact, date and amount")
        return cls(
            mode=mode,
            central_php_monthly=central,
            lower_php_monthly=lower,
            upper_php_monthly=upper,
            range_basis=range_basis,
            basis=basis,
            data_as_of=data_as_of,
            occupation_specificity=choice(
                data["occupation_specificity"], "occupation_specificity", SPECIFICITY
            ),
            region_specificity=choice(data["region_specificity"], "region_specificity", SPECIFICITY),
            explanation=text(data["explanation"], "explanation"),
            limitations=text_list(data["limitations"], "limitations"),
            reliability_flags=text_list(data["reliability_flags"], "reliability_flags"),
            source_id=source,
            source_version=source_version,
            artifact_version=artifact,
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "central_php_monthly": decimal_json(self.central_php_monthly),
            "lower_php_monthly": decimal_json(self.lower_php_monthly),
            "upper_php_monthly": decimal_json(self.upper_php_monthly),
            "range_basis": self.range_basis,
            "basis": self.basis,
            "data_as_of": self.data_as_of,
            "occupation_specificity": self.occupation_specificity,
            "region_specificity": self.region_specificity,
            "explanation": self.explanation,
            "limitations": list(self.limitations),
            "reliability_flags": list(self.reliability_flags),
            "source_id": self.source_id,
            "source_version": self.source_version,
            "artifact_version": self.artifact_version,
            "currency": "PHP",
            "period": "monthly",
        }


@dataclass(frozen=True, slots=True)
class StatutoryFloorObservation:
    observation_id: str
    source_id: str
    source_version: str
    effective_date: str
    region_psgc_code: str | None
    region_label: str
    sector: str
    coverage: str
    amount_php_daily: Decimal
    exclusions: str | None

    @classmethod
    def from_dict(cls, data: Any) -> StatutoryFloorObservation:
        fields = {
            "contract_version", "observation_id", "source_id", "source_version",
            "effective_date", "region_psgc_code", "region_label", "sector", "coverage",
            "amount_php_daily", "exclusions", "currency", "period",
        }
        data = check_fields(data, fields, set(), "statutory_floor")
        check_version(data["contract_version"])
        if data["currency"] != "PHP" or data["period"] != "daily":
            raise ContractError("currency/period", "statutory floor must remain daily PHP")
        return cls(
            observation_id=identifier(data["observation_id"], "observation_id"),
            source_id=identifier(data["source_id"], "source_id"),
            source_version=text(data["source_version"], "source_version"),
            effective_date=as_of(data["effective_date"], "effective_date"),
            region_psgc_code=ph_code(
                data["region_psgc_code"], "region_psgc_code", nullable=True
            ),
            region_label=text(data["region_label"], "region_label"),
            sector=text(data["sector"], "sector"),
            coverage=text(data["coverage"], "coverage"),
            amount_php_daily=decimal(data["amount_php_daily"], "amount_php_daily"),
            exclusions=text(data["exclusions"], "exclusions", nullable=True),
        )

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "effective_date": self.effective_date,
            "region_psgc_code": self.region_psgc_code,
            "region_label": self.region_label,
            "sector": self.sector,
            "coverage": self.coverage,
            "amount_php_daily": decimal_json(self.amount_php_daily),
            "exclusions": self.exclusions,
            "currency": "PHP",
            "period": "daily",
        }
