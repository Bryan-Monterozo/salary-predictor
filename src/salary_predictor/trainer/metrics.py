from __future__ import annotations

from decimal import Decimal, InvalidOperation
from statistics import median


def _amount(value: str | int | Decimal, *, observed: bool) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Salary or prediction must be a positive finite amount")
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric salary or prediction") from exc
    if not amount.is_finite() or (observed and amount <= 0):
        raise ValueError("Observed salary must be positive; predictions must be finite")
    return amount


def _quantile(sorted_errors: list[Decimal], fraction: Decimal) -> Decimal:
    if not sorted_errors:
        raise ValueError("Cannot score an empty cohort")
    position = Decimal(len(sorted_errors) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_errors) - 1)
    return (
        sorted_errors[lower]
        + (sorted_errors[upper] - sorted_errors[lower]) * (position - lower)
    )


def _value(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def absolute_error_metrics(
    observed: list[str | int | Decimal],
    predicted: list[str | int | Decimal],
) -> dict:
    """Pure arithmetic for synthetic contract tests, not evidence of model accuracy."""
    if not observed or len(observed) != len(predicted):
        raise ValueError("Observed and predicted arrays must have identical nonzero lengths")
    errors = sorted(
        abs(_amount(actual, observed=True) - _amount(prediction, observed=False))
        for actual, prediction in zip(observed, predicted, strict=True)
    )
    return {
        "count": len(errors),
        "mae_php": _value(sum(errors) / len(errors)),
        "median_absolute_error_php": _value(Decimal(median(errors))),
        "absolute_error_q90_php": _value(_quantile(errors, Decimal("0.9"))),
        "absolute_error_q95_php": _value(_quantile(errors, Decimal("0.95"))),
        "basis": "caller_supplied_paired_monthly_PHP_values",
    }


def subgroup_error_metrics(
    observed: list[str | int | Decimal],
    predicted: list[str | int | Decimal],
    groups: list[str | None],
    *,
    min_support: int,
) -> dict:
    if (
        not isinstance(min_support, int) or isinstance(min_support, bool)
        or min_support < 2
    ):
        raise ValueError("Support threshold must be at least two")
    if len(groups) != len(observed) or len(observed) != len(predicted):
        raise ValueError("Subgroup field must align with salary values")
    _ = absolute_error_metrics(observed, predicted)
    indexed: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        if group is None or not isinstance(group, str) or not group.strip():
            continue
        indexed.setdefault(group, []).append(index)
    reported = {}
    suppressed = {}
    for group, indices in sorted(indexed.items()):
        if len(indices) < min_support:
            suppressed[group] = len(indices)
            continue
        reported[group] = absolute_error_metrics(
            [observed[i] for i in indices],
            [predicted[i] for i in indices],
        )
    return {
        "min_support": min_support,
        "reported": reported,
        "suppressed_group_counts": suppressed,
        "missing_group_count": len(groups) - sum(len(x) for x in indexed.values()),
    }
