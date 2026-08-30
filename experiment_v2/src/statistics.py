"""Seed-level paired inference for the frozen comparison family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from typing import Iterable, Mapping, Sequence

import numpy as np


def _paired_arrays(
    first: Sequence[float], second: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired samples must be non-empty and have equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must be finite")
    return a, b


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided exact randomisation p-value for paired differences.

    Zero differences are removed because changing their sign creates duplicate
    assignments.  Ten formal training seeds require only 2^10 enumerations.
    """

    differences = np.asarray(differences, dtype=np.float64).reshape(-1)
    if not np.isfinite(differences).all() or differences.size == 0:
        raise ValueError("differences must be a non-empty finite vector")
    nonzero = differences[~np.isclose(differences, 0.0, atol=1e-12, rtol=0.0)]
    if nonzero.size == 0:
        return 1.0
    if nonzero.size > 20:
        raise ValueError(
            "exact enumeration is restricted to <=20 non-zero pairs; "
            "predeclare a Monte Carlo approximation for larger samples"
        )
    observed = abs(float(nonzero.mean()))
    extreme = 0
    total = 1 << int(nonzero.size)
    tolerance = 1e-12
    for bits in range(total):
        signs = np.fromiter(
            (1.0 if bits & (1 << index) else -1.0 for index in range(nonzero.size)),
            dtype=np.float64,
            count=nonzero.size,
        )
        statistic = abs(float(np.mean(nonzero * signs)))
        extreme += int(statistic >= observed - tolerance)
    return float(extreme / total)


def mean_confidence_interval(
    values: Sequence[float], *, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided Student-t confidence interval for an independent seed mean."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    try:
        from scipy.stats import t

        critical = float(t.ppf((1.0 + confidence) / 2.0, values.size - 1))
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        critical = 1.959963984540054
    margin = critical * standard_error
    return mean - margin, mean + margin


def paired_effect_size(differences: Sequence[float]) -> float:
    """Paired standardised mean difference (Cohen's dz)."""

    differences = np.asarray(differences, dtype=np.float64).reshape(-1)
    if differences.size < 2:
        return float("nan")
    standard_deviation = float(differences.std(ddof=1))
    mean = float(differences.mean())
    if np.isclose(standard_deviation, 0.0):
        if np.isclose(mean, 0.0):
            return 0.0
        return float(np.copysign(np.inf, mean))
    return mean / standard_deviation


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original input order."""

    values = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    family_size = values.size
    for rank, original_index in enumerate(order):
        candidate = (family_size - rank) * values[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


@dataclass(frozen=True)
class PairedComparison:
    method_a: str
    method_b: str
    metric: str
    n_pairs: int
    mean_a: float
    sd_a: float
    mean_b: float
    sd_b: float
    mean_difference: float
    ci95_low: float
    ci95_high: float
    effect_dz: float
    p_exact: float
    p_holm: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def paired_comparison(
    first: Sequence[float],
    second: Sequence[float],
    *,
    method_a: str,
    method_b: str,
    metric: str,
) -> PairedComparison:
    """Summarise ``method_a - method_b`` using seed as the unit."""

    a, b = _paired_arrays(first, second)
    differences = a - b
    low, high = mean_confidence_interval(differences)
    return PairedComparison(
        method_a=str(method_a),
        method_b=str(method_b),
        metric=str(metric),
        n_pairs=int(a.size),
        mean_a=float(a.mean()),
        sd_a=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        mean_b=float(b.mean()),
        sd_b=float(b.std(ddof=1)) if b.size > 1 else 0.0,
        mean_difference=float(differences.mean()),
        ci95_low=float(low),
        ci95_high=float(high),
        effect_dz=float(paired_effect_size(differences)),
        p_exact=float(exact_sign_flip_pvalue(differences)),
    )


def apply_holm(
    comparisons: Iterable[PairedComparison],
) -> list[PairedComparison]:
    """Return comparisons with Holm correction applied as one declared family."""

    from dataclasses import replace

    comparisons = list(comparisons)
    if not comparisons:
        return []
    adjusted = holm_adjust([comparison.p_exact for comparison in comparisons])
    return [
        replace(comparison, p_holm=float(p_holm))
        for comparison, p_holm in zip(comparisons, adjusted)
    ]


__all__ = [
    "exact_sign_flip_pvalue",
    "mean_confidence_interval",
    "paired_effect_size",
    "holm_adjust",
    "PairedComparison",
    "paired_comparison",
    "apply_holm",
]
