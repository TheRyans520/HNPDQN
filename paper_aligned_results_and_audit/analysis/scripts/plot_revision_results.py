#!/usr/bin/env python3
"""Build publication-ready revision figures from already frozen CSV artifacts.

The module has no result-directory discovery and no training code.  It reads
only the four CSV files explicitly supplied by the caller.  Learned-method
uncertainty is computed across ten independently trained seeds; deterministic
and privileged references are shown as fixed-trajectory means without a fake
training-seed confidence interval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


EXPECTED_TRAIN_SEEDS = tuple(range(1, 11))
EXPECTED_EVAL_SEEDS = tuple(range(1001, 1021))
JAMMER_ORDER = ("sweeping", "random")
CONDITION_ORDER = (
    "within_condition_pilot",
    "distance_shift_40cm_10dBm",
    "power_shift_20cm_5dBm",
)
CONDITION_LABELS = {
    "within_condition_pilot": "Pilot\n20 cm / 10 dBm",
    "distance_shift_40cm_10dBm": "Distance shift\n40 cm / 10 dBm",
    "power_shift_20cm_5dBm": "Power shift\n20 cm / 5 dBm",
}
CONDITION_SHORT = {
    "within_condition_pilot": "Pilot",
    "distance_shift_40cm_10dBm": "Distance shift",
    "power_shift_20cm_5dBm": "Power shift",
}

# Okabe-Ito colors plus marker redundancy for grayscale reproduction.
METHOD_STYLES: dict[str, dict[str, Any]] = {
    "hnp": {
        "label": "HNP-DQN",
        "color": "#0072B2",
        "marker": "o",
        "role": "learned",
        "hollow": False,
    },
    "matched": {
        "label": "Capacity-matched MLP",
        "color": "#D55E00",
        "marker": "s",
        "role": "learned",
        "hollow": False,
    },
    "threshold": {
        "label": "Threshold / hysteresis",
        "color": "#009E73",
        "marker": "^",
        "role": "ordinary",
        "hollow": False,
    },
    "max_quality": {
        "label": "Maximum measured quality",
        "color": "#E69F00",
        "marker": "v",
        "role": "ordinary",
        "hollow": False,
    },
    "stay": {
        "label": "Stay",
        "color": "#56B4E9",
        "marker": "<",
        "role": "ordinary",
        "hollow": False,
    },
    "random": {
        "label": "Random",
        "color": "#CC79A7",
        "marker": ">",
        "role": "ordinary",
        "hollow": False,
    },
    "schedule_sweep": {
        "label": "Schedule-aware (sweep only)",
        "color": "#000000",
        "marker": "X",
        "role": "schedule-aware",
        "hollow": False,
    },
    "jammer_greedy": {
        "label": "Jammer-aware greedy (privileged)",
        "color": "#009E73",
        "marker": "D",
        "role": "privileged-current-label",
        "hollow": True,
    },
    "clairvoyant_oracle": {
        "label": "Clairvoyant DP (privileged upper bound)",
        "color": "#000000",
        "marker": "*",
        "role": "privileged-full-sequence",
        "hollow": True,
    },
}
METHOD_ORDER = tuple(METHOD_STYLES)
LEARNED_METHODS = ("hnp", "matched")
COMMON_REFERENCES = (
    "threshold",
    "max_quality",
    "stay",
    "random",
    "jammer_greedy",
    "clairvoyant_oracle",
)
ABLATION_ORDER = (
    "no_polynomial",
    "no_layernorm",
    "no_dueling",
    "hnp_gamma0",
)
ABLATION_LABELS = {
    "no_polynomial": "No polynomial",
    "no_layernorm": "No LayerNorm",
    "no_dueling": "No dueling",
    "hnp_gamma0": r"HNP, $\gamma=0$",
}
OKABE_VARIANT_COLORS = {
    "no_polynomial": "#E69F00",
    "no_layernorm": "#009E73",
    "no_dueling": "#CC79A7",
    "hnp_gamma0": "#0072B2",
}
VARIANT_MARKERS = {
    "no_polynomial": "o",
    "no_layernorm": "s",
    "no_dueling": "^",
    "hnp_gamma0": "D",
}


class ValidationError(RuntimeError):
    """Raised when an input cannot support the declared figure."""


@dataclass(frozen=True)
class FigureArtifacts:
    main_source: pd.DataFrame
    primary_source: pd.DataFrame
    ablation_source: pd.DataFrame


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValidationError(f"{label} does not exist: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValidationError(f"{label} is missing columns: {missing}")


def _mean_t_ci(values: Sequence[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 2 or not np.isfinite(data).all():
        raise ValidationError("a learned-method CI needs at least two finite seeds")
    mean = float(data.mean())
    sd = float(data.std(ddof=1))
    margin = float(student_t.ppf(0.975, data.size - 1)) * sd / math.sqrt(data.size)
    return mean, mean - margin, mean + margin


def _condition_metadata(summary: pd.DataFrame, condition: str) -> dict[str, Any]:
    subset = summary[summary["condition"] == condition]
    if subset.empty:
        raise ValidationError(f"seed summary lacks condition {condition}")
    result: dict[str, Any] = {"condition": condition}
    for column in ("condition_role", "distance_cm", "power_dbm"):
        values = subset[column].drop_duplicates()
        if len(values) != 1:
            raise ValidationError(f"inconsistent {column} for {condition}")
        result[column] = values.iloc[0]
    return result


def _reference_expected(jammer: str) -> tuple[str, ...]:
    if jammer == "sweeping":
        return (*COMMON_REFERENCES[:4], "schedule_sweep", *COMMON_REFERENCES[4:])
    return COMMON_REFERENCES


def prepare_main_source_data(
    seed_summary: pd.DataFrame, evaluation_episodes: pd.DataFrame
) -> pd.DataFrame:
    """Create the source table used by the 2 x 3 performance figure."""

    summary_required = {
        "condition",
        "condition_role",
        "distance_cm",
        "power_dbm",
        "jammer_mode",
        "method",
        "train_seed",
        "return",
        "collision_rate",
        "switch_rate",
    }
    episode_required = {
        "condition",
        "condition_role",
        "distance_cm",
        "power_dbm",
        "jammer_mode",
        "method",
        "train_seed",
        "eval_seed",
        "return",
        "collision_rate",
        "switch_rate",
    }
    _require_columns(seed_summary, summary_required, "seed_summary")
    _require_columns(evaluation_episodes, episode_required, "evaluation_episodes")
    if set(seed_summary["condition"]) != set(CONDITION_ORDER):
        raise ValidationError("seed_summary must contain exactly the three declared conditions")
    if set(seed_summary["jammer_mode"]) != set(JAMMER_ORDER):
        raise ValidationError("seed_summary must contain sweeping and random modes")
    unknown = set(seed_summary["method"]) - set(METHOD_ORDER)
    if unknown:
        raise ValidationError(f"unrecognized methods would be silently omitted: {sorted(unknown)}")
    forbidden_schedule = seed_summary[
        (seed_summary["method"] == "schedule_sweep")
        & (seed_summary["jammer_mode"] != "sweeping")
    ]
    if not forbidden_schedule.empty:
        raise ValidationError("schedule_sweep must not appear outside sweeping mode")

    output_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        metadata = _condition_metadata(seed_summary, condition)
        for jammer in JAMMER_ORDER:
            setting_summary = seed_summary[
                (seed_summary["condition"] == condition)
                & (seed_summary["jammer_mode"] == jammer)
            ]
            for method in LEARNED_METHODS:
                learned = setting_summary[
                    (setting_summary["method"] == method)
                    & setting_summary["train_seed"].notna()
                ].copy()
                learned["train_seed"] = learned["train_seed"].astype(int)
                learned = learned.sort_values("train_seed")
                seeds = tuple(int(value) for value in learned["train_seed"])
                if seeds != EXPECTED_TRAIN_SEEDS:
                    raise ValidationError(
                        f"{condition}/{jammer}/{method} needs seeds 1--10, got {seeds}"
                    )
                if learned.duplicated("train_seed").any():
                    raise ValidationError(f"duplicate seed for {condition}/{jammer}/{method}")
                mean, low, high = _mean_t_ci(learned["return"])
                style = METHOD_STYLES[method]
                for row in learned.to_dict(orient="records"):
                    output_rows.append(
                        {
                            **metadata,
                            "jammer_mode": jammer,
                            "method": method,
                            "method_label": style["label"],
                            "role": style["role"],
                            "point_type": "individual_training_seed",
                            "train_seed": int(row["train_seed"]),
                            "n_training_seeds": 10,
                            "n_fixed_trajectories_per_seed": 20,
                            "return": float(row["return"]),
                            "ci95_low": np.nan,
                            "ci95_high": np.nan,
                            "return_sd_descriptive": np.nan,
                            "collision_rate": float(row["collision_rate"]),
                            "switch_rate": float(row["switch_rate"]),
                            "color": style["color"],
                            "marker": style["marker"],
                            "hollow": bool(style["hollow"]),
                        }
                    )
                output_rows.append(
                    {
                        **metadata,
                        "jammer_mode": jammer,
                        "method": method,
                        "method_label": style["label"],
                        "role": style["role"],
                        "point_type": "training_seed_mean_t95ci",
                        "train_seed": np.nan,
                        "n_training_seeds": 10,
                        "n_fixed_trajectories_per_seed": 20,
                        "return": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "return_sd_descriptive": float(
                            learned["return"].std(ddof=1)
                        ),
                        "collision_rate": float(learned["collision_rate"].mean()),
                        "switch_rate": float(learned["switch_rate"].mean()),
                        "color": style["color"],
                        "marker": style["marker"],
                        "hollow": bool(style["hollow"]),
                    }
                )

            expected_references = _reference_expected(jammer)
            actual_reference_rows = setting_summary[
                setting_summary["train_seed"].isna()
            ]
            actual_references = tuple(
                method for method in expected_references if method in set(actual_reference_rows["method"])
            )
            if actual_references != expected_references or set(
                actual_reference_rows["method"]
            ) != set(expected_references):
                raise ValidationError(
                    f"{condition}/{jammer} reference methods differ; expected "
                    f"{expected_references}, got {sorted(set(actual_reference_rows['method']))}"
                )

            for method in expected_references:
                episodes = evaluation_episodes[
                    (evaluation_episodes["condition"] == condition)
                    & (evaluation_episodes["jammer_mode"] == jammer)
                    & (evaluation_episodes["method"] == method)
                    & evaluation_episodes["train_seed"].isna()
                ].copy()
                if len(episodes) != 20:
                    raise ValidationError(
                        f"{condition}/{jammer}/{method} has {len(episodes)} fixed trajectories; expected 20"
                    )
                seeds = tuple(sorted(int(value) for value in episodes["eval_seed"]))
                if seeds != EXPECTED_EVAL_SEEDS:
                    raise ValidationError(
                        f"{condition}/{jammer}/{method} does not contain eval seeds 1001--1020"
                    )
                if not np.isfinite(
                    episodes[["return", "collision_rate", "switch_rate"]].to_numpy(float)
                ).all():
                    raise ValidationError(f"non-finite reference result for {condition}/{jammer}/{method}")
                summary_row = actual_reference_rows[
                    actual_reference_rows["method"] == method
                ]
                if len(summary_row) != 1:
                    raise ValidationError(
                        f"{condition}/{jammer}/{method} needs one baseline seed-summary row"
                    )
                episode_means = episodes[["return", "collision_rate", "switch_rate"]].mean()
                if not np.allclose(
                    summary_row[["return", "collision_rate", "switch_rate"]].to_numpy(float)[0],
                    episode_means.to_numpy(float),
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise ValidationError(
                        f"seed_summary and evaluation episodes disagree for {condition}/{jammer}/{method}"
                    )
                style = METHOD_STYLES[method]
                output_rows.append(
                    {
                        **metadata,
                        "jammer_mode": jammer,
                        "method": method,
                        "method_label": style["label"],
                        "role": style["role"],
                        "point_type": "fixed_trajectory_mean_no_seed_ci",
                        "train_seed": np.nan,
                        "n_training_seeds": 0,
                        "n_fixed_trajectories_per_seed": 20,
                        "return": float(episode_means["return"]),
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "return_sd_descriptive": float(episodes["return"].std(ddof=1)),
                        "collision_rate": float(episode_means["collision_rate"]),
                        "switch_rate": float(episode_means["switch_rate"]),
                        "color": style["color"],
                        "marker": style["marker"],
                        "hollow": bool(style["hollow"]),
                    }
                )

    output = pd.DataFrame(output_rows)
    output["condition_order"] = output["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    output["jammer_order"] = output["jammer_mode"].map(
        {name: index for index, name in enumerate(JAMMER_ORDER)}
    )
    output["method_order"] = output["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    output["point_order"] = output["point_type"].map(
        {
            "individual_training_seed": 0,
            "training_seed_mean_t95ci": 1,
            "fixed_trajectory_mean_no_seed_ci": 1,
        }
    )
    return output.sort_values(
        ["jammer_order", "condition_order", "method_order", "point_order", "train_seed"],
        na_position="last",
    ).reset_index(drop=True)


def _validate_comparison_numeric(frame: pd.DataFrame, label: str) -> None:
    columns = ["mean_difference", "ci95_low", "ci95_high", "effect_dz", "p_holm"]
    _require_columns(frame, columns, label)
    finite_columns = ["mean_difference", "ci95_low", "ci95_high", "p_holm"]
    if not np.isfinite(frame[finite_columns].to_numpy(float)).all():
        raise ValidationError(f"{label} contains non-finite estimates, intervals, or p-values")
    if ((frame["p_holm"] < 0) | (frame["p_holm"] > 1)).any():
        raise ValidationError(f"{label} has Holm p-values outside [0,1]")
    if (frame["ci95_low"] > frame["ci95_high"]).any():
        raise ValidationError(f"{label} has reversed confidence intervals")


def _p_label(value: float) -> str:
    if value < 0.001:
        return f"Holm p={value:.1e}"
    return f"Holm p={value:.3f}"


def prepare_primary_forest_data(primary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "condition",
        "jammer_mode",
        "method_a",
        "method_b",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "effect_dz",
        "p_exact",
        "p_holm",
    }
    _require_columns(primary, required, "primary_comparisons")
    _validate_comparison_numeric(primary, "primary_comparisons")
    if len(primary) != 12:
        raise ValidationError(f"primary forest requires 12 rows, got {len(primary)}")
    if set(primary["condition"]) != set(CONDITION_ORDER):
        raise ValidationError("primary comparisons have an unexpected condition set")
    if set(primary["jammer_mode"]) != set(JAMMER_ORDER):
        raise ValidationError("primary comparisons have an unexpected jammer set")
    if set(primary["method_a"]) != {"hnp"} or set(primary["metric"]) != {"return"}:
        raise ValidationError("primary comparisons must be HNP return differences")
    if set(primary["n_pairs"].astype(int)) != {10}:
        raise ValidationError("every primary comparison must contain 10 paired seeds")

    expected = {
        (condition, jammer, comparator)
        for condition in CONDITION_ORDER
        for jammer in JAMMER_ORDER
        for comparator in (
            ("schedule_sweep", "matched")
            if jammer == "sweeping"
            else ("threshold", "matched")
        )
    }
    actual = set(
        primary[["condition", "jammer_mode", "method_b"]].itertuples(
            index=False, name=None
        )
    )
    if actual != expected:
        raise ValidationError(
            f"primary comparison identities differ; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    if primary.duplicated(["condition", "jammer_mode", "method_b"]).any():
        raise ValidationError("duplicate primary comparison row")

    result = primary.copy()
    comparator_labels = {
        "schedule_sweep": "Simple rule",
        "threshold": "Simple rule",
        "matched": "Matched MLP",
    }
    result["condition_label"] = result["condition"].map(CONDITION_SHORT)
    result["comparator_label"] = result["method_b"].map(comparator_labels)
    result["row_label"] = result["condition_label"] + " · " + result["comparator_label"]
    result["holm_label"] = result["p_holm"].map(_p_label)
    result["color"] = result["method_b"].map(
        {"schedule_sweep": "#000000", "threshold": "#009E73", "matched": "#D55E00"}
    )
    result["marker"] = result["method_b"].map(
        {"schedule_sweep": "X", "threshold": "^", "matched": "s"}
    )
    result["condition_order"] = result["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    result["comparator_order"] = result["method_b"].map(
        {"schedule_sweep": 0, "threshold": 0, "matched": 1}
    )
    result["jammer_order"] = result["jammer_mode"].map(
        {name: index for index, name in enumerate(JAMMER_ORDER)}
    )
    return result.sort_values(
        ["jammer_order", "condition_order", "comparator_order"]
    ).reset_index(drop=True)


def prepare_ablation_forest_data(ablation: pd.DataFrame) -> pd.DataFrame:
    required = {
        "condition",
        "jammer_mode",
        "method_a",
        "method_b",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "effect_dz",
        "p_exact",
        "p_holm",
        "holm_family",
    }
    _require_columns(ablation, required, "ablation_comparisons")
    _validate_comparison_numeric(ablation, "ablation_comparisons")
    if len(ablation) != 24:
        raise ValidationError(f"ablation forest requires 24 rows, got {len(ablation)}")
    if set(ablation["condition"]) != set(CONDITION_ORDER):
        raise ValidationError("ablation comparisons have an unexpected condition set")
    if set(ablation["jammer_mode"]) != set(JAMMER_ORDER):
        raise ValidationError("ablation comparisons have an unexpected jammer set")
    if set(ablation["method_a"]) != {"hnp"} or set(ablation["metric"]) != {"return"}:
        raise ValidationError("ablation comparisons must be HNP return differences")
    if set(ablation["method_b"]) != set(ABLATION_ORDER):
        raise ValidationError("ablation comparisons have an unexpected variant set")
    if set(ablation["n_pairs"].astype(int)) != {10}:
        raise ValidationError("every ablation comparison must contain 10 paired seeds")
    if ablation["holm_family"].nunique() != 1:
        raise ValidationError("all 24 ablations must share one exploratory Holm family")
    expected = {
        (condition, jammer, variant)
        for condition in CONDITION_ORDER
        for jammer in JAMMER_ORDER
        for variant in ABLATION_ORDER
    }
    actual = set(
        ablation[["condition", "jammer_mode", "method_b"]].itertuples(
            index=False, name=None
        )
    )
    if actual != expected or ablation.duplicated(
        ["condition", "jammer_mode", "method_b"]
    ).any():
        raise ValidationError("ablation comparison identities are incomplete or duplicated")

    result = ablation.copy()
    result["condition_label"] = result["condition"].map(CONDITION_SHORT)
    result["variant_label"] = result["method_b"].map(ABLATION_LABELS)
    result["row_label"] = result["condition_label"] + " · " + result["variant_label"]
    result["holm_label"] = result["p_holm"].map(_p_label)
    result["color"] = result["method_b"].map(OKABE_VARIANT_COLORS)
    result["marker"] = result["method_b"].map(VARIANT_MARKERS)
    result["condition_order"] = result["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    result["variant_order"] = result["method_b"].map(
        {name: index for index, name in enumerate(ABLATION_ORDER)}
    )
    result["jammer_order"] = result["jammer_mode"].map(
        {name: index for index, name in enumerate(JAMMER_ORDER)}
    )
    return result.sort_values(
        ["jammer_order", "condition_order", "variant_order"]
    ).reset_index(drop=True)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=2.5)


def _main_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="none",
                markersize=5.2 if method != "clairvoyant_oracle" else 7,
                markerfacecolor="none" if style["hollow"] else style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                label=style["label"],
            )
        )
    return handles


def plot_main_performance(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 6.1), sharex=True, sharey=True)
    y_map = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary_points = source[source["point_type"] != "individual_training_seed"]
    value_candidates = list(summary_points["return"].astype(float))
    value_candidates.extend(summary_points["ci95_low"].dropna().astype(float))
    value_candidates.extend(summary_points["ci95_high"].dropna().astype(float))
    low, high = min(value_candidates), max(value_candidates)
    pad = max((high - low) * 0.06, 0.5)

    panel_index = 0
    for row_index, jammer in enumerate(JAMMER_ORDER):
        for column_index, condition in enumerate(CONDITION_ORDER):
            ax = axes[row_index, column_index]
            panel = source[
                (source["jammer_mode"] == jammer)
                & (source["condition"] == condition)
            ]
            for method in METHOD_ORDER:
                method_rows = panel[panel["method"] == method]
                if method_rows.empty:
                    continue
                style = METHOD_STYLES[method]
                y = y_map[method]
                individuals = method_rows[
                    method_rows["point_type"] == "individual_training_seed"
                ].sort_values("train_seed")
                if not individuals.empty:
                    jitter = np.linspace(-0.13, 0.13, len(individuals))
                    ax.scatter(
                        individuals["return"],
                        y + jitter,
                        s=9,
                        marker=style["marker"],
                        color=style["color"],
                        alpha=0.42,
                        edgecolors="none",
                        zorder=2,
                    )
                summary = method_rows[
                    method_rows["point_type"] != "individual_training_seed"
                ]
                if len(summary) != 1:
                    raise ValidationError(
                        f"plot source needs one summary row for {condition}/{jammer}/{method}"
                    )
                point = summary.iloc[0]
                face = "none" if bool(point["hollow"]) else style["color"]
                if pd.notna(point["ci95_low"]):
                    lower = float(point["return"] - point["ci95_low"])
                    upper = float(point["ci95_high"] - point["return"])
                    ax.errorbar(
                        float(point["return"]),
                        y,
                        xerr=np.array([[lower], [upper]]),
                        fmt=style["marker"],
                        color=style["color"],
                        markerfacecolor=face,
                        markeredgecolor=style["color"],
                        markeredgewidth=0.8,
                        markersize=4.8,
                        capsize=2.2,
                        elinewidth=1.0,
                        zorder=4,
                    )
                else:
                    ax.plot(
                        float(point["return"]),
                        y,
                        marker=style["marker"],
                        linestyle="none",
                        color=style["color"],
                        markerfacecolor=face,
                        markeredgecolor=style["color"],
                        markeredgewidth=1.0,
                        markersize=5.2 if method != "clairvoyant_oracle" else 7,
                        zorder=4,
                    )

            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(len(METHOD_ORDER) - 0.4, -0.6)
            ax.set_yticks(range(len(METHOD_ORDER)))
            if column_index == 0:
                ax.set_yticklabels([METHOD_STYLES[name]["label"] for name in METHOD_ORDER])
            else:
                ax.tick_params(labelleft=False)
            ax.set_title(f"{CONDITION_LABELS[condition]} · {jammer.capitalize()}")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
            ax.text(
                0.01,
                0.98,
                chr(ord("A") + panel_index),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
            _despine(ax)
            panel_index += 1

    for ax in axes[-1, :]:
        ax.set_xlabel("100-step return")
    fig.legend(
        handles=_main_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
        ncol=3,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.255, right=0.99, top=0.95, bottom=0.20, wspace=0.18, hspace=0.28)
    return fig


def _forest_xlim(frame: pd.DataFrame) -> tuple[float, float]:
    bound = float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        frame["ci95_low"].to_numpy(float),
                        frame["ci95_high"].to_numpy(float),
                        frame["mean_difference"].to_numpy(float),
                    ]
                )
            )
        )
    )
    bound = max(bound, 0.5) * 1.65
    return -bound, bound


def _forest_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    title: str,
    panel_letter: str,
) -> None:
    panel = panel.reset_index(drop=True)
    y = np.arange(len(panel))
    for index, row in panel.iterrows():
        lower = float(row["mean_difference"] - row["ci95_low"])
        upper = float(row["ci95_high"] - row["mean_difference"])
        ax.errorbar(
            float(row["mean_difference"]),
            index,
            xerr=np.array([[lower], [upper]]),
            fmt=str(row["marker"]),
            color=str(row["color"]),
            markerfacecolor=str(row["color"]),
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=4.8,
            capsize=2.2,
            elinewidth=1.0,
            zorder=3,
        )
        ax.text(
            0.985,
            index,
            str(row["holm_label"]),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
        )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(panel["row_label"])
    ax.set_ylim(len(panel) - 0.45, -0.55)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta$ return (HNP $-$ comparator)")
    ax.text(
        0.01,
        0.98,
        panel_letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    _despine(ax)


def plot_primary_forest(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), sharex=True)
    limits = _forest_xlim(source)
    for index, jammer in enumerate(JAMMER_ORDER):
        panel = source[source["jammer_mode"] == jammer]
        _forest_panel(
            axes[index],
            panel,
            title=f"{jammer.capitalize()} jammer",
            panel_letter=chr(ord("A") + index),
        )
        axes[index].set_xlim(*limits)
        if index > 0:
            axes[index].tick_params(labelleft=False)
    # Reserve enough canvas for the longest two-part row label (notably
    # "Distance shift · Matched MLP") in raster and vector backends alike.
    fig.subplots_adjust(left=0.20, right=0.99, top=0.91, bottom=0.16, wspace=0.30)
    return fig


def plot_ablation_forest(source: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.6), sharex=True)
    limits = _forest_xlim(source)
    for index, jammer in enumerate(JAMMER_ORDER):
        panel = source[source["jammer_mode"] == jammer]
        _forest_panel(
            axes[index],
            panel,
            title=f"{jammer.capitalize()} jammer · exploratory family",
            panel_letter=chr(ord("A") + index),
        )
        axes[index].set_xlim(*limits)
        if index > 0:
            axes[index].tick_params(labelleft=False)
    fig.subplots_adjust(left=0.185, right=0.99, top=0.93, bottom=0.11, wspace=0.35)
    return fig


def _save_figure(fig: plt.Figure, base: Path) -> None:
    pdf_metadata = {
        "Creator": "plot_revision_results.py",
        "Title": base.name,
        "Subject": "Frozen round-2 revision result visualization",
    }
    svg_metadata = {
        "Creator": "plot_revision_results.py",
        "Title": base.name,
        "Description": "Frozen round-2 revision result visualization",
    }
    fig.savefig(base.with_suffix(".pdf"), format="pdf", metadata=pdf_metadata)
    fig.savefig(base.with_suffix(".svg"), format="svg", metadata=svg_metadata)
    fig.savefig(
        base.with_suffix(".png"),
        format="png",
        dpi=600,
        metadata={"Software": "plot_revision_results.py"},
    )
    plt.close(fig)


def _captions() -> str:
    return """# Figure captions

## Main performance

**Main performance under the frozen scan-stratified protocol.** Columns show
the transparent 20 cm/10 dBm pilot, the 40 cm/10 dBm distance shift, and the
20 cm/5 dBm power shift; rows show sweeping and random jammer modes. For
HNP-DQN and the capacity-matched MLP-DDQN, faint symbols are the 10 independent
training-seed estimates after averaging 20 fixed trajectories per seed, and
the larger symbol/error bar is the seed mean with a two-sided 95% Student-t
confidence interval. Ordinary deterministic references are shown only as
their means over the same 20 fixed trajectories and are not assigned a
training-seed confidence interval. The schedule-aware rule appears only under
sweeping. Hollow diamond/star symbols identify references with current-label
or full-sequence privileged jammer information; the latter is the
clairvoyant finite-horizon dynamic-programming upper bound. Pilot scans 8 and
9 occur ten times each; every transfer scan 0--9 occurs twice. Colors use the
Okabe-Ito palette, with redundant marker shapes for grayscale reproduction.

Source: `main_performance_source_data.csv`.

## Primary comparison forest

**Predeclared primary paired comparisons.** Points show the mean difference in
100-step return, HNP-DQN minus comparator, across 10 paired training seeds;
horizontal bars are two-sided 95% Student-t confidence intervals for the seed
differences. The simple comparator is the schedule-aware rule under sweeping
and threshold/hysteresis under random jamming; the capacity-matched MLP-DDQN
comparison is shown under both modes. The dashed line denotes no difference.
Numeric labels are Holm-adjusted p-values for the single 12-comparison primary
family; asterisks are not used. The pilot is transparently distinguished from
the two cross-configuration evaluations.

Source: `primary_forest_source_data.csv`.

## Exploratory ablation forest

**Exploratory paired ablation comparisons.** Points show the mean difference
in 100-step return, full HNP-DQN minus variant, across 10 paired training
seeds; horizontal bars are two-sided 95% Student-t confidence intervals. The
dashed line denotes no difference. Numeric labels are Holm-adjusted p-values
for the separate 24-comparison exploratory family spanning four variants,
three physical conditions, and two jammer modes. This family is not mixed
with the primary HNP-versus-simple-rule/capacity-matched family, and
nonsignificant intervals are interpreted as inconclusive rather than as
equivalence.

Source: `ablation_forest_source_data.csv`.
"""


def prepare_figure_artifacts(
    *,
    seed_summary: pd.DataFrame,
    evaluation_episodes: pd.DataFrame,
    primary_comparisons: pd.DataFrame,
    ablation_comparisons: pd.DataFrame,
) -> FigureArtifacts:
    return FigureArtifacts(
        main_source=prepare_main_source_data(seed_summary, evaluation_episodes),
        primary_source=prepare_primary_forest_data(primary_comparisons),
        ablation_source=prepare_ablation_forest_data(ablation_comparisons),
    )


def build_revision_figures(
    *,
    seed_summary_path: Path,
    evaluation_episodes_path: Path,
    primary_comparisons_path: Path,
    ablation_comparisons_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> FigureArtifacts:
    seed_summary = _read_csv(seed_summary_path, "seed_summary")
    evaluation = _read_csv(evaluation_episodes_path, "evaluation_episodes")
    primary = _read_csv(primary_comparisons_path, "primary_comparisons")
    ablation = _read_csv(ablation_comparisons_path, "ablation_comparisons")
    artifacts = prepare_figure_artifacts(
        seed_summary=seed_summary,
        evaluation_episodes=evaluation,
        primary_comparisons=primary,
        ablation_comparisons=ablation,
    )

    output = Path(output_dir).expanduser().resolve()
    names = (
        "main_performance.pdf",
        "main_performance.png",
        "main_performance.svg",
        "primary_forest.pdf",
        "primary_forest.png",
        "primary_forest.svg",
        "ablation_forest.pdf",
        "ablation_forest.png",
        "ablation_forest.svg",
        "main_performance_source_data.csv",
        "primary_forest_source_data.csv",
        "ablation_forest_source_data.csv",
        "captions.md",
    )
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not overwrite:
        raise ValidationError(
            f"refusing to overwrite {len(existing)} existing figure artifact(s); use --overwrite"
        )

    # All input validation and source-table construction occur before writing.
    output.mkdir(parents=True, exist_ok=True)
    artifacts.main_source.to_csv(output / "main_performance_source_data.csv", index=False)
    artifacts.primary_source.to_csv(output / "primary_forest_source_data.csv", index=False)
    artifacts.ablation_source.to_csv(output / "ablation_forest_source_data.csv", index=False)
    (output / "captions.md").write_text(_captions(), encoding="utf-8")

    _configure_style()
    _save_figure(plot_main_performance(artifacts.main_source), output / "main_performance")
    _save_figure(plot_primary_forest(artifacts.primary_source), output / "primary_forest")
    _save_figure(plot_ablation_forest(artifacts.ablation_source), output / "ablation_forest")
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create frozen round-2 performance and forest figures."
    )
    parser.add_argument("--seed-summary", type=Path, required=True)
    parser.add_argument("--evaluation-episodes", type=Path, required=True)
    parser.add_argument("--primary-comparisons", type=Path, required=True)
    parser.add_argument("--ablation-comparisons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = build_revision_figures(
            seed_summary_path=args.seed_summary,
            evaluation_episodes_path=args.evaluation_episodes,
            primary_comparisons_path=args.primary_comparisons,
            ablation_comparisons_path=args.ablation_comparisons,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "wrote publication figures with "
        f"{len(artifacts.main_source)} main-source rows, "
        f"{len(artifacts.primary_source)} primary rows, and "
        f"{len(artifacts.ablation_source)} ablation rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
