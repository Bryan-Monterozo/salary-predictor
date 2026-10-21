from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

FEATURE_SCHEMA_VERSION = 1
CONTRACT_VERSION = 1

EDUCATION_LEVELS = frozenset(
    {"none", "primary", "secondary", "vocational", "associate", "bachelor", "master", "doctorate"}
)
SOURCE_ROLES = frozenset(
    {"taxonomy", "aggregate_benchmark", "individual_earnings", "advertised_salary", "statutory_minimum"}
)
PERIODS = frozenset({"hourly", "daily", "weekly", "monthly", "annual", "unknown"})
PAY_COMPONENTS = frozenset({"basic", "allowance", "total", "unknown"})
TAX_BASES = frozenset({"gross", "net", "unknown"})
RECORD_TYPES = frozenset({"individual_earnings", "advertised_salary"})
RESULT_MODES = frozenset({"MODEL_ESTIMATE", "MARKET_BENCHMARK", "UNAVAILABLE"})


class ContractError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def check_fields(data: Any, required: set[str], optional: set[str], name: str) -> dict:
    if not isinstance(data, dict):
        raise ContractError(name, "expected an object")
    missing = required - data.keys()
    extra = data.keys() - required - optional
    if missing:
        raise ContractError(name, f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ContractError(name, f"unknown fields: {', '.join(sorted(extra))}")
    return data


def check_version(value: Any, field: str = "contract_version") -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ContractError(field, f"expected version {CONTRACT_VERSION}")
    return value


def text(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(field, "expected non-empty text without surrounding spaces")
    return value


def choice(value: Any, field: str, allowed: frozenset[str], *, nullable: bool = False) -> str | None:
    result = text(value, field, nullable=nullable)
    if result is not None and result not in allowed:
        raise ContractError(field, f"unsupported value: {result}")
    return result


def identifier(value: Any, field: str, *, nullable: bool = False) -> str | None:
    result = text(value, field, nullable=nullable)
    if result is not None and not re.fullmatch(r"[A-Za-z0-9_.:/#-]+", result):
        raise ContractError(field, "invalid identifier")
    return result


def ph_code(value: Any, field: str, *, nullable: bool = False) -> str | None:
    result = text(value, field, nullable=nullable)
    if result is not None and not re.fullmatch(r"[0-9]{1,10}", result):
        raise ContractError(field, "expected a code of 1 to 10 digits")
    return result


def as_of(value: Any, field: str, *, nullable: bool = False) -> str | None:
    result = text(value, field, nullable=nullable)
    if result is None:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", result):
        raise ContractError(field, "expected YYYY-MM or YYYY-MM-DD")
    try:
        date.fromisoformat(result + "-01" if len(result) == 7 else result)
    except ValueError as exc:
        raise ContractError(field, "invalid calendar date") from exc
    return result


def decimal(value: Any, field: str, *, nullable: bool = False, zero_ok: bool = False) -> Decimal | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"(0|[1-9]\d*)(\.\d{1,2})?", value):
        raise ContractError(field, "expected a plain decimal string with at most two places")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ContractError(field, "invalid decimal") from exc
    if result < 0 or (result == 0 and not zero_ok):
        raise ContractError(field, "must be positive" if not zero_ok else "cannot be negative")
    return result


def decimal_json(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(field, "expected a list")
    entries = tuple(text(item, f"{field}[{index}]") for index, item in enumerate(value))
    if len(entries) != len(set(entries)):
        raise ContractError(field, "duplicate values")
    return entries


def amount_range(
    minimum: Any, maximum: Any, low_field: str, high_field: str
) -> tuple[Decimal, Decimal | None]:
    low = decimal(minimum, low_field)
    high = decimal(maximum, high_field, nullable=True)
    assert low is not None
    if high is not None and high < low:
        raise ContractError(high_field, "cannot be less than the minimum")
    return low, high
