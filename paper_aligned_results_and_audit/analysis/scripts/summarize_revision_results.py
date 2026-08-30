#!/usr/bin/env python3
"""Validate frozen revision artifacts and prepare auditable result text.

This is a post-processing gate, not an experiment runner.  It accepts one
explicit canonical primary-run directory, one explicit directory produced by
``build_revision_artifacts.py``, and one explicit isolated cost-profile
directory.  It does not discover runs.  It validates every input before it
creates either output file.

The generated prose is deliberately conservative.  Mechanical labels report
only whether the paired confidence interval and Holm-adjusted test jointly
support an advantage, support a disadvantage, or are inconclusive.  Practical
importance, narrative emphasis, and the final language/caption audit remain
human decisions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import build_revision_artifacts as gate


SCHEMA_VERSION = 1
EXPECTED_MODELS = ("hnp", "matched")
EXPECTED_CONDITIONS = gate.EXPECTED_CONDITIONS
EXPECTED_JAMMERS = gate.EXPECTED_JAMMERS
EXPECTED_TRAIN_SEEDS = gate.EXPECTED_TRAIN_SEEDS
EXPECTED_EVAL_SEEDS = gate.EXPECTED_EVAL_SEEDS
EXPECTED_PRIMARY_COLUMNS = (
    "condition",
    "jammer_mode",
    "method_a",
    "method_b",
    "metric",
    "n_pairs",
    "mean_a",
    "sd_a",
    "mean_b",
    "sd_b",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "effect_dz",
    "p_exact",
    "p_holm",
)
EXPECTED_VARIANTS = (
    "no_polynomial",
    "no_layernorm",
    "no_dueling",
    "hnp_gamma0",
)
EXPECTED_BASELINES = {
    "sweeping": (
        "threshold",
        "max_quality",
        "stay",
        "random",
        "schedule_sweep",
        "jammer_greedy",
        "clairvoyant_oracle",
    ),
    "random": (
        "threshold",
        "max_quality",
        "stay",
        "random",
        "jammer_greedy",
        "clairvoyant_oracle",
    ),
}
ORDINARY_BASELINES = {
    "sweeping": ("threshold", "max_quality", "stay", "random", "schedule_sweep"),
    "random": ("threshold", "max_quality", "stay", "random"),
}
METHOD_LABELS = {
    "hnp": "HNP-DQN",
    "matched": "capacity-matched MLP-DDQN",
    "threshold": "threshold/hysteresis",
    "max_quality": "maximum measured quality",
    "stay": "stay",
    "random": "random selection",
    "schedule_sweep": "schedule-aware sweep rule",
    "jammer_greedy": "jammer-aware greedy reference (privileged)",
    "clairvoyant_oracle": "clairvoyant DP upper bound (privileged)",
    "no_polynomial": "No-Polynomial",
    "no_layernorm": "No-LayerNorm",
    "no_dueling": "No-Dueling",
    "hnp_gamma0": r"HNP with $\gamma=0$",
}
CONDITION_LABELS = {
    "within_condition_pilot": "20 cm/10 dBm transparent pilot",
    "distance_shift_40cm_10dBm": "40 cm/10 dBm distance transfer",
    "power_shift_20cm_5dBm": "20 cm/5 dBm power transfer",
}
MANUSCRIPT_PATTERN = re.compile(r"\[VERIFIED RESULT:\s*(.*?)\]", re.DOTALL)
RESPONSE_PATTERN = re.compile(r"\[\[VERIFIED_RESULT:(.*?)\]\]", re.DOTALL)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(gate.ValidationError):
    """Raised before any output is written when an artifact is not canonical."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValidationError(f"required {label} is missing: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc


def _load_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise ValidationError(f"required {label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValidationError(f"{label} is missing columns: {missing}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalised_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _equal_scalar(first: Any, second: Any) -> bool:
    first = _normalised_scalar(first)
    second = _normalised_scalar(second)
    if pd.isna(first) and pd.isna(second):
        return True
    if isinstance(first, (bool, np.bool_)) or isinstance(second, (bool, np.bool_)):
        return str(first).lower() == str(second).lower()
    if isinstance(first, (int, float, np.number)) and isinstance(
        second, (int, float, np.number)
    ):
        return bool(
            np.isclose(float(first), float(second), rtol=1e-10, atol=1e-10, equal_nan=True)
        )
    return str(first) == str(second)


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
    exact_columns: bool = True,
) -> None:
    """Compare two keyed tables, including every reported statistic."""

    if exact_columns and set(actual.columns) != set(expected.columns):
        raise ValidationError(
            f"{label} column set differs; missing={sorted(set(expected.columns)-set(actual.columns))}, "
            f"extra={sorted(set(actual.columns)-set(expected.columns))}"
        )
    _require_columns(actual, keys, label)
    _require_columns(expected, keys, f"recomputed {label}")
    if actual.duplicated(list(keys)).any() or expected.duplicated(list(keys)).any():
        raise ValidationError(f"{label} has duplicate comparison keys")
    if len(actual) != len(expected):
        raise ValidationError(
            f"{label} has {len(actual)} rows but recomputation has {len(expected)}"
        )
    left = actual.sort_values(list(keys)).reset_index(drop=True)
    right = expected.sort_values(list(keys)).reset_index(drop=True)
    columns = list(expected.columns)
    for row_index in range(len(right)):
        for column in columns:
            if not _equal_scalar(left.iloc[row_index][column], right.iloc[row_index][column]):
                key = tuple(right.iloc[row_index][item] for item in keys)
                raise ValidationError(
                    f"{label} mismatch for {key}, column {column}: "
                    f"reported={left.iloc[row_index][column]!r}, "
                    f"recomputed={right.iloc[row_index][column]!r}"
                )


def _learned_rows(summary: pd.DataFrame, method: str, condition: str, jammer: str) -> pd.DataFrame:
    rows = summary[
        (summary["condition"] == condition)
        & (summary["jammer_mode"] == jammer)
        & (summary["method"] == method)
        & summary["train_seed"].notna()
    ].copy()
    rows["train_seed"] = rows["train_seed"].astype(int)
    rows = rows.sort_values("train_seed")
    seeds = tuple(int(value) for value in rows["train_seed"])
    if seeds != EXPECTED_TRAIN_SEEDS or rows.duplicated("train_seed").any():
        raise ValidationError(
            f"{condition}/{jammer}/{method} requires exactly seeds {EXPECTED_TRAIN_SEEDS}; "
            f"got {seeds}"
        )
    return rows


def _primary_paired_row(
    first: np.ndarray,
    second: np.ndarray,
    *,
    condition: str,
    jammer: str,
    comparator: str,
) -> dict[str, Any]:
    """Mirror ``experiment_v2/src/statistics.py`` for an independent check.

    The primary runner uses NumPy's default ``isclose`` tolerance for the
    zero-variance Cohen-dz edge case, whereas the exploratory artifact builder
    intentionally has its own tighter helper.  Keeping that distinction here
    prevents a legitimate primary CSV from being rejected only in a nearly
    constant-difference edge case.
    """

    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    if first.shape != second.shape or first.size != 10:
        raise ValidationError(f"primary comparison {comparator} needs 10 paired seeds")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValidationError(f"primary comparison {comparator} has non-finite values")
    differences = first - second
    mean_difference = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    margin = (
        float(gate.student_t.ppf(0.975, differences.size - 1))
        * sd_difference
        / math.sqrt(differences.size)
    )
    if np.isclose(sd_difference, 0.0):
        effect = 0.0 if np.isclose(mean_difference, 0.0) else float(
            np.copysign(np.inf, mean_difference)
        )
    else:
        effect = mean_difference / sd_difference
    return {
        "condition": condition,
        "jammer_mode": jammer,
        "method_a": "hnp",
        "method_b": comparator,
        "metric": "return",
        "n_pairs": 10,
        "mean_a": float(first.mean()),
        "sd_a": float(first.std(ddof=1)),
        "mean_b": float(second.mean()),
        "sd_b": float(second.std(ddof=1)),
        "mean_difference": mean_difference,
        "ci95_low": mean_difference - margin,
        "ci95_high": mean_difference + margin,
        "effect_dz": effect,
        "p_exact": gate._exact_sign_flip_pvalue(differences),
        "p_holm": float("nan"),
    }


def _recompute_primary_comparisons(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute the 12 declared rows from seed-level endpoints."""

    _require_columns(
        seed_summary,
        {
            "condition",
            "jammer_mode",
            "method",
            "train_seed",
            "return",
        },
        "primary seed_summary",
    )
    rows: list[dict[str, Any]] = []
    for jammer in EXPECTED_JAMMERS:
        heuristic = "schedule_sweep" if jammer == "sweeping" else "threshold"
        for condition in EXPECTED_CONDITIONS:
            hnp = _learned_rows(seed_summary, "hnp", condition, jammer)
            for comparator in (heuristic, "matched"):
                if comparator == "matched":
                    other = _learned_rows(seed_summary, comparator, condition, jammer)
                    joined = hnp[["train_seed", "return"]].merge(
                        other[["train_seed", "return"]],
                        on="train_seed",
                        how="outer",
                        validate="one_to_one",
                        suffixes=("_hnp", "_other"),
                        indicator=True,
                    )
                    if set(joined["_merge"]) != {"both"}:
                        raise ValidationError(
                            f"unpaired HNP/matched seeds for {condition}/{jammer}"
                        )
                    joined = joined.sort_values("train_seed")
                    first = joined["return_hnp"].to_numpy(dtype=float)
                    second = joined["return_other"].to_numpy(dtype=float)
                else:
                    baseline = seed_summary[
                        (seed_summary["condition"] == condition)
                        & (seed_summary["jammer_mode"] == jammer)
                        & (seed_summary["method"] == comparator)
                        & seed_summary["train_seed"].isna()
                    ]
                    if len(baseline) != 1:
                        raise ValidationError(
                            f"{condition}/{jammer}/{comparator} requires one fixed-trajectory mean; "
                            f"got {len(baseline)}"
                        )
                    first = hnp["return"].to_numpy(dtype=float)
                    second = np.full(first.shape, float(baseline.iloc[0]["return"]))
                row = _primary_paired_row(
                    first,
                    second,
                    condition=condition,
                    jammer=jammer,
                    comparator=comparator,
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 12:
        raise ValidationError(f"primary comparison recomputation produced {len(frame)} rows")
    frame["p_holm"] = gate._holm_adjust(frame["p_exact"].to_numpy(dtype=float))
    return frame[list(EXPECTED_PRIMARY_COLUMNS)]


def _validate_primary_comparisons(
    reported: pd.DataFrame, seed_summary: pd.DataFrame
) -> pd.DataFrame:
    if set(reported.columns) != set(EXPECTED_PRIMARY_COLUMNS):
        raise ValidationError(
            "primary_comparisons.csv does not have the exact declared schema; "
            f"expected={list(EXPECTED_PRIMARY_COLUMNS)}, got={list(reported.columns)}"
        )
    expected = _recompute_primary_comparisons(seed_summary)
    _assert_frame_matches(
        reported,
        expected,
        keys=("condition", "jammer_mode", "method_b"),
        label="primary_comparisons.csv",
    )
    return _canonical_primary_order(reported)


def _canonical_primary_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_jammer"] = result["jammer_mode"].map(
        {name: index for index, name in enumerate(EXPECTED_JAMMERS)}
    )
    result["_condition"] = result["condition"].map(
        {name: index for index, name in enumerate(EXPECTED_CONDITIONS)}
    )
    result["_comparator"] = [
        1 if method == "matched" else 0 for method in result["method_b"]
    ]
    return (
        result.sort_values(["_jammer", "_condition", "_comparator"])
        .drop(columns=["_jammer", "_condition", "_comparator"])
        .reset_index(drop=True)
    )


def _validate_primary_run(primary_dir: Path) -> tuple[gate.ValidatedRun, pd.DataFrame]:
    run = gate._validate_one_run(
        gate.RunSpec("primary", Path(primary_dir), EXPECTED_MODELS, 0.95)
    )
    comparisons = _load_csv(run.spec.path / "primary_comparisons.csv", "primary comparisons")
    return run, _validate_primary_comparisons(comparisons, run.seed_summary)


def _manifest_run_matches(record: Mapping[str, Any], run: gate.ValidatedRun) -> None:
    expected_scalars = {
        "path": str(run.spec.path),
        "expected_models": list(run.spec.expected_models),
        "gamma": float(run.config["gamma"]),
        "config_sha256": run.hashes["config_sha256"],
        "core_code_bundle_sha256": run.hashes["core_code_bundle_sha256"],
        "predeclared_schedule_sha256": run.hashes["predeclared_schedule_sha256"],
        "normalizer_sha256": run.hashes["normalizer_sha256"],
        "input_artifact_sha256": run.hashes["input_artifact_sha256"],
        "checkpoint_sha256": run.hashes["checkpoint_sha256"],
        "validation_counts": run.validation_counts,
    }
    for key, expected in expected_scalars.items():
        if _json_safe(record.get(key)) != _json_safe(expected):
            raise ValidationError(
                f"result_manifest run {run.spec.label} has stale or mismatched {key}"
            )


def _validate_ablation_directory(
    ablation_dir: Path, primary: gate.ValidatedRun
) -> tuple[pd.DataFrame, dict[str, gate.ValidatedRun], dict[str, Any]]:
    root = Path(ablation_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"ablation artifact directory does not exist: {root}")
    manifest_path = root / "result_manifest.json"
    manifest = _load_json(manifest_path, "ablation result manifest")
    if not isinstance(manifest, Mapping):
        raise ValidationError("ablation result_manifest.json must contain an object")
    if manifest.get("schema_version") != gate.SCHEMA_VERSION or manifest.get("status") != "validated":
        raise ValidationError("ablation manifest is not a validated schema-v1 artifact")
    run_records = manifest.get("runs")
    if not isinstance(run_records, Mapping) or set(run_records) != {
        "primary",
        "gamma0",
        "no_polynomial",
        "no_layernorm",
        "no_dueling",
    }:
        raise ValidationError("ablation manifest must contain exactly the five declared run roles")
    specs = {
        "primary": gate.RunSpec("primary", Path(run_records["primary"]["path"]), ("hnp", "matched"), 0.95),
        "gamma0": gate.RunSpec(
            "gamma0", Path(run_records["gamma0"]["path"]), ("hnp",), 0.0,
            variant_method="hnp", variant_label="hnp_gamma0"
        ),
        "no_polynomial": gate.RunSpec(
            "no_polynomial", Path(run_records["no_polynomial"]["path"]), ("no_polynomial",), 0.95,
            variant_method="no_polynomial", variant_label="no_polynomial"
        ),
        "no_layernorm": gate.RunSpec(
            "no_layernorm", Path(run_records["no_layernorm"]["path"]), ("no_layernorm",), 0.95,
            variant_method="no_layernorm", variant_label="no_layernorm"
        ),
        "no_dueling": gate.RunSpec(
            "no_dueling", Path(run_records["no_dueling"]["path"]), ("no_dueling",), 0.95,
            variant_method="no_dueling", variant_label="no_dueling"
        ),
    }
    if specs["primary"].path.expanduser().resolve() != primary.spec.path:
        raise ValidationError(
            "ablation manifest primary path is not the explicitly supplied canonical primary run"
        )
    runs: dict[str, gate.ValidatedRun] = {"primary": primary}
    for label in ("gamma0", "no_polynomial", "no_layernorm", "no_dueling"):
        runs[label] = gate._validate_one_run(specs[label])
    cross = gate._validate_cross_run_provenance(list(runs.values()))
    if _json_safe(manifest.get("cross_run_provenance")) != _json_safe(cross):
        raise ValidationError("ablation manifest cross-run provenance is stale")
    for label, run in runs.items():
        record = run_records.get(label)
        if not isinstance(record, Mapping):
            raise ValidationError(f"ablation manifest run record {label} is malformed")
        _manifest_run_matches(record, run)

    output_record = manifest.get("outputs", {}).get("ablation_comparisons")
    if not isinstance(output_record, Mapping):
        raise ValidationError("ablation manifest lacks the comparison output record")
    if output_record.get("path") != "ablation_comparisons.csv":
        raise ValidationError("ablation comparison path must be ablation_comparisons.csv")
    comparison_path = root / "ablation_comparisons.csv"
    if str(output_record.get("sha256", "")).lower() != _sha256(comparison_path):
        raise ValidationError("ablation comparison hash does not match the manifest")
    reported = _load_csv(comparison_path, "ablation comparisons")
    if int(output_record.get("rows", -1)) != len(reported) or list(
        output_record.get("columns", [])
    ) != list(reported.columns):
        raise ValidationError("ablation comparison row/column manifest is stale")
    expected = gate._build_ablation_comparisons(
        runs["primary"],
        [runs["gamma0"], runs["no_polynomial"], runs["no_layernorm"], runs["no_dueling"]],
    )
    _assert_frame_matches(
        reported,
        expected,
        keys=("condition", "jammer_mode", "method_b"),
        label="ablation_comparisons.csv",
    )
    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "train_seeds": list(EXPECTED_TRAIN_SEEDS),
        "eval_seeds": list(EXPECTED_EVAL_SEEDS),
        "trajectories_per_method_training_seed_condition_jammer": 20,
        "conditions": list(EXPECTED_CONDITIONS),
        "jammer_modes": list(EXPECTED_JAMMERS),
        "metric": "return",
        "holm_family": gate.EXPLORATORY_FAMILY,
        "holm_family_size": 24,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ValidationError(f"ablation manifest protocol field {key} is invalid")
    return expected.reset_index(drop=True), runs, dict(manifest)


def _records_match_csv(frame: pd.DataFrame, records: Sequence[Mapping[str, Any]]) -> None:
    union = set().union(*(record.keys() for record in records)) if records else set()
    if set(frame.columns) != union:
        raise ValidationError(
            f"cost CSV/JSON column union differs; csv-only={sorted(set(frame.columns)-union)}, "
            f"json-only={sorted(union-set(frame.columns))}"
        )
    if len(frame) != len(records):
        raise ValidationError("cost CSV and JSON have different row counts")
    for index, record in enumerate(records):
        for column in frame.columns:
            expected = record.get(column)
            actual = frame.iloc[index][column]
            if not _equal_scalar(actual, expected):
                raise ValidationError(
                    f"cost CSV/JSON mismatch at row {index}, column {column}: "
                    f"csv={actual!r}, json={expected!r}"
                )


def _finite_nonnegative(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    _require_columns(frame, columns, label)
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValidationError(f"{label} contains non-finite or negative values in {columns}")


def _int_set(values: Iterable[Any], label: str) -> set[int]:
    try:
        converted = {int(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must contain integers") from exc
    return converted


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValidationError(f"expected Boolean value, got {value!r}")


def _primary_checkpoint_records(primary: gate.ValidatedRun) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in primary.freeze.get("checkpoints", []):
        key = (str(record["model"]), str(record["jammer_mode"]), int(record["train_seed"]))
        path = primary.spec.path / str(record["path"])
        records[key] = {
            "sha256": str(record["sha256"]).lower(),
            "bytes": path.stat().st_size,
        }
    return records


def _validate_cost_profile(
    cost_dir: Path, primary: gate.ValidatedRun
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    root = Path(cost_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"cost profile directory does not exist: {root}")
    csv_path = root / "isolated_model_costs.csv"
    json_path = root / "isolated_model_costs.json"
    frame = _load_csv(csv_path, "isolated model cost CSV")
    payload = _load_json(json_path, "isolated model cost JSON")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValidationError("isolated cost JSON must contain metadata")
    training_records = payload.get("isolated_training")
    inference_records = payload.get("canonical_checkpoint_inference")
    if not isinstance(training_records, list) or not isinstance(inference_records, list):
        raise ValidationError("isolated cost JSON record arrays are missing")
    records = [*training_records, *inference_records]
    if not all(isinstance(record, Mapping) for record in records):
        raise ValidationError("isolated cost JSON contains a malformed record")
    _records_match_csv(frame, records)
    training = frame[frame["record_type"] == "isolated_training_wall_time"].copy()
    inference = frame[frame["record_type"] == "canonical_checkpoint_inference"].copy()
    if len(training) != 6 or len(inference) != 40 or len(frame) != 46:
        raise ValidationError(
            f"isolated cost profile requires 6 training and 40 inference rows; "
            f"got {len(training)} and {len(inference)}"
        )
    if set(frame["record_type"]) != {
        "isolated_training_wall_time",
        "canonical_checkpoint_inference",
    }:
        raise ValidationError("isolated cost profile contains an unknown record type")

    metadata = dict(payload["metadata"])
    if metadata.get("purpose") != "isolated engineering cost comparison; no policy-performance inference":
        raise ValidationError("cost metadata purpose does not preserve non-inferential scope")
    if _int_set(metadata.get("profiling_seeds", ()), "profiling_seeds") != {91001, 91002, 91003}:
        raise ValidationError("cost metadata must contain profiling seeds 91001--91003")
    expected_training_order = [
        [91001, "hnp"],
        [91001, "matched"],
        [91002, "matched"],
        [91002, "hnp"],
        [91003, "hnp"],
        [91003, "matched"],
    ]
    if metadata.get("training_order") != expected_training_order:
        raise ValidationError("cost metadata does not contain the predeclared counterbalanced order")
    profiling_seed_role = str(metadata.get("profiling_seed_role", ""))
    if "not units for policy-performance inference" not in profiling_seed_role:
        raise ValidationError("cost metadata does not label profiling seeds as non-inferential")
    if str(metadata.get("data_scope", "")) != (
        "development 20cm/10dBm train scans 0-6; validation scan 7 loaded only for provenance"
    ):
        raise ValidationError("cost metadata development-only data scope is invalid")
    if "pilot scans 8-9" not in str(metadata.get("excluded_data", "")) or "cross-configuration" not in str(
        metadata.get("excluded_data", "")
    ):
        raise ValidationError("cost metadata does not explicitly exclude pilot and transfer data")
    if int(metadata.get("latency_warmup", -1)) != 200 or int(
        metadata.get("latency_repetitions", -1)
    ) != 1000:
        raise ValidationError("cost latency protocol must use 200 warm-ups and 1000 repetitions")
    if int(metadata.get("cpu_intraop_threads", -1)) != 1 or int(
        metadata.get("cpu_interop_threads", -1)
    ) != 1:
        raise ValidationError("cost profile must use one intra-op and one inter-op CPU thread")
    memory_scope = str(metadata.get("memory_scope", ""))
    if "excludes activations" not in memory_scope or "allocator overhead" not in memory_scope:
        raise ValidationError("cost metadata memory scope is incomplete")

    training_keys = set(
        (str(row.model), int(row.profiling_seed))
        for row in training[["model", "profiling_seed"]].itertuples(index=False)
    )
    expected_training = {(model, seed) for model in EXPECTED_MODELS for seed in (91001, 91002, 91003)}
    if training_keys != expected_training or training.duplicated(["model", "profiling_seed"]).any():
        raise ValidationError("isolated training profile is incomplete or duplicated")
    if _int_set(training["profile_order"], "training profile_order") != set(range(6)):
        raise ValidationError("isolated training profile order must be exactly 0--5")
    ordered_training = training.sort_values("profile_order")
    observed_training_order = [
        [int(row.profiling_seed), str(row.model)]
        for row in ordered_training[["profiling_seed", "model"]].itertuples(index=False)
    ]
    if observed_training_order != expected_training_order:
        raise ValidationError("training rows do not follow the predeclared counterbalanced order")
    for row in training.itertuples(index=False):
        if str(row.jammer_mode) != "sweeping":
            raise ValidationError("isolated training timing must use only sweeping mode")
        if int(row.train_episodes) != 200 or int(row.episode_length) != 100 or int(row.environment_steps) != 20000:
            raise ValidationError("isolated training timing does not use the exact 200x100 formal budget")
        if int(row.optimizer_updates) <= 0:
            raise ValidationError("isolated training timing has no optimizer updates")
        if int(row.cpu_intraop_threads) != 1 or int(row.cpu_interop_threads) != 1:
            raise ValidationError("isolated training row is not single-threaded")
        if str(row.training_data_scope) != "20cm/10dBm train scans 0-6 only" or not _bool_value(
            row.validation_loaded_not_used_for_training
        ):
            raise ValidationError("isolated training row has an invalid data scope")
        if str(row.profiling_seed_role) != profiling_seed_role:
            raise ValidationError("profiling seed was not labelled non-inferential")
        if str(row.inference_persistent_tensor_bytes_scope) != memory_scope:
            raise ValidationError("training-row persistent-tensor scope differs from metadata")
    _finite_nonnegative(
        training,
        (
            "agent_initialization_wall_seconds",
            "training_loop_wall_seconds",
            "agent_init_plus_training_wall_seconds",
            "trainable_parameters",
            "parameter_bytes",
            "registered_buffer_bytes",
            "inference_persistent_tensor_bytes",
        ),
        "isolated training rows",
    )
    if not np.allclose(
        training["agent_initialization_wall_seconds"].to_numpy(float)
        + training["training_loop_wall_seconds"].to_numpy(float),
        training["agent_init_plus_training_wall_seconds"].to_numpy(float),
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValidationError("initialization + training wall time does not add up")

    checkpoint_records = _primary_checkpoint_records(primary)
    inference_keys = set(
        (str(row.model), str(row.jammer_mode), int(row.canonical_train_seed))
        for row in inference[["model", "jammer_mode", "canonical_train_seed"]].itertuples(index=False)
    )
    if inference_keys != set(checkpoint_records) or inference.duplicated(
        ["model", "jammer_mode", "canonical_train_seed"]
    ).any():
        raise ValidationError("cost inference rows do not cover all 40 canonical primary checkpoints")
    if _int_set(inference["profile_order"], "inference profile_order") != set(range(40)):
        raise ValidationError("inference profile order must be exactly 0--39")
    cpu_columns = (
        "batch1_cpu_latency_ms_median",
        "batch1_cpu_latency_ms_mean",
        "batch1_cpu_latency_ms_p05",
        "batch1_cpu_latency_ms_p95",
    )
    _finite_nonnegative(
        inference,
        (*cpu_columns, "trainable_parameters", "parameter_bytes", "registered_buffer_bytes", "inference_persistent_tensor_bytes", "serialized_state_bytes", "checkpoint_bytes"),
        "canonical inference rows",
    )
    cuda_available = _bool_value(metadata.get("cuda_available"))
    gpu_columns = (
        "batch1_gpu_latency_ms_median",
        "batch1_gpu_latency_ms_mean",
        "batch1_gpu_latency_ms_p05",
        "batch1_gpu_latency_ms_p95",
    )
    if cuda_available:
        _finite_nonnegative(inference, gpu_columns, "canonical GPU inference rows")
    elif not inference[list(gpu_columns)].isna().all().all():
        raise ValidationError("GPU latency is present although metadata says CUDA was unavailable")
    if (inference["batch1_cpu_latency_ms_p05"] > inference["batch1_cpu_latency_ms_median"]).any() or (
        inference["batch1_cpu_latency_ms_median"] > inference["batch1_cpu_latency_ms_p95"]
    ).any():
        raise ValidationError("CPU latency quantiles are reversed")
    if cuda_available and (
        (inference["batch1_gpu_latency_ms_p05"] > inference["batch1_gpu_latency_ms_median"]).any()
        or (inference["batch1_gpu_latency_ms_median"] > inference["batch1_gpu_latency_ms_p95"]).any()
    ):
        raise ValidationError("GPU latency quantiles are reversed")

    for row in inference.itertuples(index=False):
        key = (str(row.model), str(row.jammer_mode), int(row.canonical_train_seed))
        expected = checkpoint_records[key]
        if str(row.checkpoint_sha256).lower() != expected["sha256"] or int(row.checkpoint_bytes) != expected["bytes"]:
            raise ValidationError(f"cost checkpoint provenance mismatch for {key}")
        if int(row.latency_warmup) != 200 or int(row.latency_repetitions) != 1000:
            raise ValidationError(f"cost latency schedule mismatch for {key}")
        if str(row.profiling_seed_role) != profiling_seed_role:
            raise ValidationError(f"cost inference row has an invalid profiling-seed role for {key}")
        if str(row.latency_input) != "batch-1 all-zero normalized 48D observation":
            raise ValidationError(f"cost latency input mismatch for {key}")
        expected_synchronization = (
            "per-forward torch.cuda.synchronize" if cuda_available else "CPU call return"
        )
        if str(row.latency_synchronization) != expected_synchronization:
            raise ValidationError(f"cost latency synchronization mismatch for {key}")
        if str(row.inference_persistent_tensor_bytes_scope) != memory_scope:
            raise ValidationError(f"cost memory scope mismatch for {key}")
        if str(row.serialized_state_bytes_role) != "storage size, not memory" or str(
            row.checkpoint_bytes_role
        ) != "storage size, not memory":
            raise ValidationError(f"storage bytes are mislabelled for {key}")
        if int(row.parameter_bytes) + int(row.registered_buffer_bytes) != int(
            row.inference_persistent_tensor_bytes
        ):
            raise ValidationError(f"persistent tensor byte accounting mismatch for {key}")

    for model in EXPECTED_MODELS:
        model_rows = inference[inference["model"] == model]
        for column in (
            "trainable_parameters",
            "parameter_bytes",
            "registered_buffer_bytes",
            "inference_persistent_tensor_bytes",
            "serialized_state_bytes",
        ):
            if model_rows[column].nunique(dropna=False) != 1:
                raise ValidationError(f"{model} {column} varies across canonical checkpoints")
        training_model = training[training["model"] == model]
        if set(training_model["trainable_parameters"].astype(int)) != set(
            model_rows["trainable_parameters"].astype(int)
        ):
            raise ValidationError(f"{model} parameter count differs between training and inference profiles")
    hnp_parameters = int(inference[inference["model"] == "hnp"].iloc[0]["trainable_parameters"])
    matched_parameters = int(inference[inference["model"] == "matched"].iloc[0]["trainable_parameters"])
    if abs(hnp_parameters - matched_parameters) / hnp_parameters > 0.01:
        raise ValidationError("the named capacity-matched MLP differs from HNP by more than 1% in parameters")

    return training, inference, metadata, {
        "isolated_model_costs.csv": _sha256(csv_path),
        "isolated_model_costs.json": _sha256(json_path),
    }


def _classification(row: Mapping[str, Any]) -> str:
    low = float(row["ci95_low"])
    high = float(row["ci95_high"])
    p_holm = float(row["p_holm"])
    if low > 0.0 and p_holm < 0.05:
        return "supported_hnp_advantage"
    if high < 0.0 and p_holm < 0.05:
        return "supported_hnp_disadvantage"
    return "inconclusive"


def _number(value: float, digits: int = 2) -> str:
    value = float(value)
    if math.isinf(value):
        return r"+\infty" if value > 0 else r"-\infty"
    if math.isnan(value):
        return "NA"
    rounded = 0.0 if abs(value) < 0.5 * (10 ** -digits) else value
    return f"{rounded:.{digits}f}"


def _p_tex(value: float) -> str:
    value = float(value)
    if value < 0.001:
        return r"$p_{\mathrm{Holm}}<0.001$"
    return rf"$p_{{\mathrm{{Holm}}}}={value:.3f}$"


def _comparison_record(row: Mapping[str, Any], family: str) -> dict[str, Any]:
    classification = _classification(row)
    condition = str(row["condition"])
    jammer = str(row["jammer_mode"])
    comparator = str(row["method_b"])
    difference_tex = (
        rf"${_number(row['mean_difference'])}$ "
        rf"[95\% CI ${_number(row['ci95_low'])}$, ${_number(row['ci95_high'])}$]"
    )
    effect_tex = rf"Cohen's $d_z={_number(row['effect_dz'])}$"
    inference_tex = f"{_p_tex(row['p_holm'])}; {classification.replace('_', ' ')}"
    direction = {
        "supported_hnp_advantage": "supported an HNP-DQN advantage",
        "supported_hnp_disadvantage": f"supported an advantage for {METHOD_LABELS[comparator]}",
        "inconclusive": "was inconclusive (not evidence of equivalence)",
    }[classification]
    sentence = (
        f"For the {CONDITION_LABELS[condition]} under {jammer} jamming, the paired "
        f"HNP-DQN minus {METHOD_LABELS[comparator]} return difference was "
        f"{_number(row['mean_difference'])} (95% CI {_number(row['ci95_low'])} to "
        f"{_number(row['ci95_high'])}; Cohen's dz={_number(row['effect_dz'])}; "
        f"Holm-adjusted p={float(row['p_holm']):.3f}), which {direction}."
    )
    # Keep non-finite effect sizes available to the renderer as +/- infinity;
    # the final JSON serializer converts them to null while the adjacent
    # ``effect_tex`` retains the explicit human-readable value.
    raw = dict(row)
    return {
        "family": family,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "jammer_mode": jammer,
        "method_b": comparator,
        "method_b_label": METHOD_LABELS[comparator],
        "classification": classification,
        "raw": raw,
        "difference_ci_tex": difference_tex,
        "effect_tex": effect_tex,
        "inference_tex": inference_tex,
        "english_sentence": sentence,
    }


def _summarise_classifications(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        label: sum(record["classification"] == label for record in records)
        for label in (
            "supported_hnp_advantage",
            "supported_hnp_disadvantage",
            "inconclusive",
        )
    }


def _descriptive_performance(run: gate.ValidatedRun) -> list[dict[str, Any]]:
    summary = run.seed_summary
    evaluation = run.evaluation
    output: list[dict[str, Any]] = []
    for condition in EXPECTED_CONDITIONS:
        for jammer in EXPECTED_JAMMERS:
            setting = summary[
                (summary["condition"] == condition) & (summary["jammer_mode"] == jammer)
            ]
            expected_methods = set(EXPECTED_MODELS) | set(EXPECTED_BASELINES[jammer])
            if set(setting["method"]) != expected_methods:
                raise ValidationError(
                    f"{condition}/{jammer} method set differs; "
                    f"missing={sorted(expected_methods-set(setting['method']))}, "
                    f"extra={sorted(set(setting['method'])-expected_methods)}"
                )
            methods: list[dict[str, Any]] = []
            for method in (*EXPECTED_MODELS, *EXPECTED_BASELINES[jammer]):
                rows = setting[setting["method"] == method]
                if method in EXPECTED_MODELS:
                    learned = _learned_rows(summary, method, condition, jammer)
                    record = {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "role": "learned",
                        "n_training_seeds": 10,
                    }
                    for metric in ("return", "collision_rate", "switch_rate"):
                        values = learned[metric].to_numpy(dtype=float)
                        record[metric] = {
                            "mean": float(values.mean()),
                            "sd_across_training_seeds": float(values.std(ddof=1)),
                        }
                else:
                    baseline = rows[rows["train_seed"].isna()]
                    if len(baseline) != 1:
                        raise ValidationError(
                            f"{condition}/{jammer}/{method} needs one descriptive baseline row"
                        )
                    episode_rows = evaluation[
                        (evaluation["condition"] == condition)
                        & (evaluation["jammer_mode"] == jammer)
                        & (evaluation["method"] == method)
                        & evaluation["train_seed"].isna()
                    ]
                    if len(episode_rows) != 20:
                        raise ValidationError(
                            f"{condition}/{jammer}/{method} needs 20 fixed trajectories"
                        )
                    record = {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "role": (
                            "privileged"
                            if method in {"jammer_greedy", "clairvoyant_oracle"}
                            else "ordinary_or_schedule_aware"
                        ),
                        "n_fixed_trajectories": 20,
                    }
                    for metric in ("return", "collision_rate", "switch_rate"):
                        values = episode_rows[metric].to_numpy(dtype=float)
                        record[metric] = {
                            "mean": float(values.mean()),
                            "sd_across_fixed_trajectories_descriptive_only": float(values.std(ddof=1)),
                        }
                methods.append(record)
            return_tex = "; ".join(
                f"{item['method_label']}: ${_number(item['return']['mean'])}"
                + (
                    rf" \pm {_number(item['return']['sd_across_training_seeds'])}$"
                    if item["role"] == "learned"
                    else "$ (fixed-trajectory mean)"
                )
                for item in methods
            )
            rates_tex = "; ".join(
                f"{item['method_label']}: collision ${_number(100*item['collision_rate']['mean'])}\\%$, "
                f"switch ${_number(100*item['switch_rate']['mean'])}\\%$"
                for item in methods
            )
            output.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "jammer_mode": jammer,
                    "methods": methods,
                    "returns_tex": return_tex,
                    "collision_switch_tex": rates_tex,
                    "uncertainty_note": (
                        "Learned entries are mean +/- SD across 10 training seeds; "
                        "fixed references are 20-trajectory point summaries and are not inferential replicates."
                    ),
                }
            )
    return output


def _strongest_ordinary(performance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for setting in performance:
        allowed = set(ORDINARY_BASELINES[setting["jammer_mode"]])
        candidates = [item for item in setting["methods"] if item["method"] in allowed]
        best_value = max(float(item["return"]["mean"]) for item in candidates)
        winners = [item for item in candidates if math.isclose(float(item["return"]["mean"]), best_value)]
        hnp = next(item for item in setting["methods"] if item["method"] == "hnp")
        predeclared = "schedule_sweep" if setting["jammer_mode"] == "sweeping" else "threshold"
        winner = winners[0]
        output.append(
            {
                "condition": setting["condition"],
                "jammer_mode": setting["jammer_mode"],
                "posthoc_best_observed_methods": [item["method"] for item in winners],
                "predeclared_simple_comparator": predeclared,
                "posthoc_winner_matches_predeclared": predeclared in {item["method"] for item in winners},
                "hnp_minus_best_observed_return": float(hnp["return"]["mean"] - best_value),
                "hnp_minus_best_observed_collision_percentage_points": float(
                    100 * (hnp["collision_rate"]["mean"] - winner["collision_rate"]["mean"])
                ),
                "hnp_minus_best_observed_switch_percentage_points": float(
                    100 * (hnp["switch_rate"]["mean"] - winner["switch_rate"]["mean"])
                ),
                "warning": (
                    "This is a descriptive post-hoc ranking. It must not inherit the adjusted p-value "
                    "of the predeclared schedule-aware/threshold comparison."
                ),
            }
        )
    return output


def _ablation_supporting_endpoints(
    runs: Mapping[str, gate.ValidatedRun]
) -> list[dict[str, Any]]:
    primary = gate._learned_seed_rows(runs["primary"], "hnp")
    variant_specs = (
        ("gamma0", "hnp", "hnp_gamma0"),
        ("no_polynomial", "no_polynomial", "no_polynomial"),
        ("no_layernorm", "no_layernorm", "no_layernorm"),
        ("no_dueling", "no_dueling", "no_dueling"),
    )
    rows: list[dict[str, Any]] = []
    keys = ["condition", "jammer_mode", "train_seed"]
    for run_label, method, variant_label in variant_specs:
        variant = gate._learned_seed_rows(runs[run_label], method)
        joined = primary[keys + ["collision_rate", "switch_rate"]].merge(
            variant[keys + ["collision_rate", "switch_rate"]],
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_hnp", "_variant"),
        )
        if set(joined["_merge"]) != {"both"}:
            raise ValidationError(f"supporting endpoint seed mismatch for {variant_label}")
        for (condition, jammer), group in joined.groupby(["condition", "jammer_mode"], sort=False):
            if tuple(sorted(group["train_seed"].astype(int))) != EXPECTED_TRAIN_SEEDS:
                raise ValidationError(f"supporting endpoint schedule incomplete for {variant_label}")
            rows.append(
                {
                    "condition": str(condition),
                    "jammer_mode": str(jammer),
                    "method_b": variant_label,
                    "mean_hnp_minus_variant_collision_percentage_points": float(
                        100 * (group["collision_rate_hnp"] - group["collision_rate_variant"]).mean()
                    ),
                    "mean_hnp_minus_variant_switch_percentage_points": float(
                        100 * (group["switch_rate_hnp"] - group["switch_rate_variant"]).mean()
                    ),
                    "scope": "descriptive paired-seed supporting endpoints; not added to the return Holm family",
                }
            )
    return rows


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValidationError("cost summary requires finite non-empty values")
    return {
        "n": int(data.size),
        "median": float(np.median(data)),
        "q1": float(np.quantile(data, 0.25)),
        "q3": float(np.quantile(data, 0.75)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _median_iqr_tex(summary: Mapping[str, Any], unit: str) -> str:
    return (
        rf"${_number(summary['median'], 3)}$ [{_number(summary['q1'], 3)}, "
        rf"{_number(summary['q3'], 3)}] {unit}"
    )


def _cost_summary(
    training: pd.DataFrame, inference: pd.DataFrame, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        train = training[training["model"] == model]
        infer = inference[inference["model"] == model]
        first = infer.iloc[0]
        gpu_values = infer["batch1_gpu_latency_ms_median"].dropna().to_numpy(float)
        model_record = {
            "trainable_parameters": int(first["trainable_parameters"]),
            "parameter_bytes": int(first["parameter_bytes"]),
            "registered_buffer_bytes": int(first["registered_buffer_bytes"]),
            "inference_persistent_tensor_bytes": int(first["inference_persistent_tensor_bytes"]),
            "inference_persistent_tensor_bytes_scope": str(first["inference_persistent_tensor_bytes_scope"]),
            "serialized_state_bytes": _distribution(infer["serialized_state_bytes"]),
            "checkpoint_bytes": _distribution(infer["checkpoint_bytes"]),
            "batch1_cpu_latency_ms_median_across_checkpoints": _distribution(
                infer["batch1_cpu_latency_ms_median"]
            ),
            "batch1_gpu_latency_ms_median_across_checkpoints": (
                _distribution(gpu_values) if gpu_values.size else None
            ),
            "training_loop_wall_seconds_across_engineering_seeds": _distribution(
                train["training_loop_wall_seconds"]
            ),
        }
        persistent_kib = model_record["inference_persistent_tensor_bytes"] / 1024
        state = model_record["serialized_state_bytes"]
        checkpoint = model_record["checkpoint_bytes"]
        cpu = model_record["batch1_cpu_latency_ms_median_across_checkpoints"]
        gpu = model_record["batch1_gpu_latency_ms_median_across_checkpoints"]
        wall = model_record["training_loop_wall_seconds_across_engineering_seeds"]
        model_record["capacity_tex"] = (
            f"{model_record['trainable_parameters']:,} trainable parameters; "
            rf"${_number(persistent_kib)}$ KiB persistent parameter/buffer tensors"
        )
        model_record["storage_tex"] = (
            rf"state ${_number(state['median']/1024)}$ [{_number(state['q1']/1024)}, "
            rf"{_number(state['q3']/1024)}] KiB; checkpoint "
            rf"${_number(checkpoint['median']/1024)}$ [{_number(checkpoint['q1']/1024)}, "
            rf"{_number(checkpoint['q3']/1024)}] KiB"
        )
        latency_parts = [f"CPU {_median_iqr_tex(cpu, 'ms')}"]
        latency_parts.append(
            f"GPU {_median_iqr_tex(gpu, 'ms')}" if gpu else "GPU not available on the profiling system"
        )
        latency_parts.append(f"training {_median_iqr_tex(wall, 's')}")
        model_record["latency_training_tex"] = "; ".join(latency_parts)
        models[model] = model_record
    hnp_n = models["hnp"]["trainable_parameters"]
    matched_n = models["matched"]["trainable_parameters"]
    gap = matched_n - hnp_n
    relative = 100 * gap / hnp_n
    models["matched"]["capacity_gap_from_hnp"] = {
        "absolute_parameters": gap,
        "percent_of_hnp": relative,
        "tex": f"{gap:+,} parameters ({relative:+.2f}\\% relative to HNP-DQN)",
    }
    return {
        "models": models,
        "protocol": {
            "training_seed_role": metadata["profiling_seed_role"],
            "training_timing_n_per_model": 3,
            "inference_checkpoint_n_per_model": 20,
            "latency_warmup": 200,
            "latency_repetitions": 1000,
            "cpu_threads": 1,
            "cuda_available": _bool_value(metadata["cuda_available"]),
            "memory_scope": metadata["memory_scope"],
        },
    }


def _stability_summary(run: gate.ValidatedRun) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for method in EXPECTED_MODELS:
        for condition in EXPECTED_CONDITIONS:
            for jammer in EXPECTED_JAMMERS:
                rows = _learned_rows(run.seed_summary, method, condition, jammer)
                records.append(
                    {
                        "method": method,
                        "condition": condition,
                        "jammer_mode": jammer,
                        "return_sd_across_training_seeds": float(rows["return"].std(ddof=1)),
                    }
                )
    ranges = {}
    for method in EXPECTED_MODELS:
        values = [item["return_sd_across_training_seeds"] for item in records if item["method"] == method]
        ranges[method] = {"min": min(values), "max": max(values)}
    sentence = (
        f"All 10 planned canonical training seeds were present for every learned-model and jammer-mode "
        f"combination. Across the six held-out settings, seed-level return SD ranged from "
        f"{_number(ranges['hnp']['min'])} to {_number(ranges['hnp']['max'])} for HNP-DQN and from "
        f"{_number(ranges['matched']['min'])} to {_number(ranges['matched']['max'])} for the matched MLP. "
        "The frozen artifacts establish completeness of the canonical runs but do not enumerate failed "
        "exploratory or pre-canonical attempts; no convergence/AUC endpoint or cross-model TD-loss "
        "stability claim is made."
    )
    return {"setting_records": records, "ranges": ranges, "english_sentence": sentence}


def _placeholder_inventory(path: Path, pattern: re.Pattern[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"{label} file is missing: {path}")
    text = path.read_text(encoding="utf-8")
    matches = list(pattern.finditer(text))
    return [
        {
            "index": index,
            "line": text.count("\n", 0, match.start()) + 1,
            "raw": match.group(0),
            "label": " ".join(match.group(1).split()),
        }
        for index, match in enumerate(matches, start=1)
    ]


def _placeholder_map(
    inventory: Sequence[Mapping[str, Any]],
    performance: Sequence[Mapping[str, Any]],
    primary_records: Sequence[Mapping[str, Any]],
    ablation_records: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Any],
    stability: Mapping[str, Any],
    primary_summary_sentence: str,
    ablation_summary_sentence: str,
) -> list[dict[str, Any]]:
    if len(inventory) != 99:
        raise ValidationError(
            f"current manuscript mapping requires exactly 99 VERIFIED RESULT markers; got {len(inventory)}"
        )
    mapping: list[dict[str, Any]] = []
    auto: dict[int, tuple[str, str | None]] = {
        1: ("primary/composite", None),
        2: ("primary matched rows + cost", None),
        3: ("all validated outcomes", None),
        4: ("validated protocol", "10 independently trained seeds and 20 fixed evaluation trajectories per condition and jammer mode"),
        5: ("validated protocol", "10"),
        6: (
            "validated protocol",
            "10 training seeds; 20 fixed trajectories per setting; two-sided 95\\% Student-$t$ interval for paired seed differences; Cohen's $d_z$; two-sided exact sign-flip enumeration; one primary Holm family over 12 rows",
        ),
        7: ("manual figure insertion", None),
        20: ("predeclared and descriptive comparator audit", None),
        57: ("all 12 primary rows", primary_summary_sentence),
        58: ("validated seed dispersion", stability["english_sentence"]),
        65: ("matched performance + cost", None),
        96: ("24 exploratory rows + supporting endpoints", ablation_summary_sentence),
        97: ("separate checkpoint weight-audit artifact", None),
        98: ("all primary rows", primary_summary_sentence),
        99: ("all ablation rows", ablation_summary_sentence),
    }
    # Table 4: two placeholders for each setting in manuscript order.
    for offset, setting in enumerate(performance):
        auto[8 + 2 * offset] = ("seed_summary/evaluation_episodes", setting["returns_tex"])
        auto[9 + 2 * offset] = ("seed_summary/evaluation_episodes", setting["collision_switch_tex"])
    # Table 5: difference, effect, inference for 12 rows.
    for row_index, record in enumerate(primary_records):
        start = 21 + 3 * row_index
        auto[start] = ("primary_comparisons", record["difference_ci_tex"])
        auto[start + 1] = ("primary_comparisons", record["effect_tex"])
        auto[start + 2] = ("primary_comparisons", record["inference_tex"])
    hnp = costs["models"]["hnp"]
    matched = costs["models"]["matched"]
    auto[59] = ("isolated cost profile", hnp["capacity_tex"])
    auto[60] = ("isolated cost profile", hnp["storage_tex"])
    auto[61] = ("isolated cost profile", hnp["latency_training_tex"])
    auto[62] = (
        "isolated cost profile",
        matched["capacity_tex"] + "; " + matched["capacity_gap_from_hnp"]["tex"],
    )
    auto[63] = ("isolated cost profile", matched["storage_tex"])
    auto[64] = ("isolated cost profile", matched["latency_training_tex"])
    # Table 6 order: variant, jammer, then the three conditions.  Matched rows
    # come from the primary family; all other variants use the exploratory family.
    primary_index = {
        (item["method_b"], item["jammer_mode"], item["condition"]): item
        for item in primary_records
    }
    ablation_index = {
        (item["method_b"], item["jammer_mode"], item["condition"]): item
        for item in ablation_records
    }
    table6_variants = ("no_polynomial", "no_layernorm", "no_dueling", "matched", "hnp_gamma0")
    position = 66
    for variant in table6_variants:
        for jammer in EXPECTED_JAMMERS:
            for condition in EXPECTED_CONDITIONS:
                index = primary_index if variant == "matched" else ablation_index
                record = index[(variant, jammer, condition)]
                family = "primary" if variant == "matched" else "exploratory ablation"
                auto[position] = (family, record["difference_ci_tex"])
                position += 1
    for item in inventory:
        source, suggestion = auto[item["index"]]
        mapping.append(
            {
                **dict(item),
                "source": source,
                "status": "auto_formatted" if suggestion is not None else "human_review_required",
                "suggested_tex": suggestion,
            }
        )
    return mapping


def _records_sentence(records: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(record["english_sentence"]) for record in records)


def _summary_sentence(records: Sequence[Mapping[str, Any]], label: str) -> str:
    counts = _summarise_classifications(records)
    return (
        f"Across the {len(records)} {label}, {counts['supported_hnp_advantage']} supported an "
        f"HNP-DQN advantage, {counts['supported_hnp_disadvantage']} supported the comparator/variant, "
        f"and {counts['inconclusive']} were inconclusive under the joint CI-plus-Holm rule. "
        "This count is a mechanical audit, not a practical-significance judgment; every setting-specific "
        "estimate and null or adverse result must remain visible."
    )


def _response_clauses(
    primary_records: Sequence[Mapping[str, Any]],
    ablation_records: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> dict[str, str]:
    simple = [item for item in primary_records if item["method_b"] != "matched"]
    matched = [item for item in primary_records if item["method_b"] == "matched"]
    gamma0 = [item for item in ablation_records if item["method_b"] == "hnp_gamma0"]
    primary_summary = _summary_sentence(primary_records, "predeclared primary comparisons")
    ablation_summary = _summary_sentence(ablation_records, "exploratory ablation comparisons")
    simple_summary = _summary_sentence(simple, "predeclared simple-comparator comparisons")
    matched_summary = _summary_sentence(matched, "capacity-matched comparisons")
    gamma_summary = _summary_sentence(gamma0, "HNP-minus-gamma-zero comparisons")
    hnp = costs["models"]["hnp"]
    mlp = costs["models"]["matched"]
    cost_sentence = (
        f"HNP-DQN used {hnp['trainable_parameters']:,} trainable parameters and the matched MLP used "
        f"{mlp['trainable_parameters']:,} ({mlp['capacity_gap_from_hnp']['percent_of_hnp']:+.2f}% "
        f"relative to HNP). HNP cost: {hnp['capacity_tex']}; {hnp['storage_tex']}; "
        f"{hnp['latency_training_tex']}. Matched-MLP cost: {mlp['capacity_tex']}; "
        f"{mlp['storage_tex']}; {mlp['latency_training_tex']}. Timing summaries are engineering "
        "medians [IQR], not policy-performance inference, and persistent tensors are not peak memory."
    )
    return {
        "one_paragraph_overall_outcome_with_uncertainty": (
            primary_summary + " " + cost_sentence
        ),
        "myopic_vs_hnp_result_with_ci": _records_sentence(gamma0),
        "sequentiality_claim_consistent_with_outcome": gamma_summary,
        "hnp_vs_strongest_heuristic_effect_ci_adjusted_p": _records_sentence(simple),
        "heuristic_comparison_claim_consistent_with_outcome": simple_summary,
        "number_of_training_seeds": "10 independently trained seeds per learned method and jammer mode",
        "number_of_fixed_evaluation_trajectories": (
            "20 fixed trajectories per physical condition and jammer mode (10 replicates per pilot scan; "
            "two replicates per transfer scan)"
        ),
        "exact_test_details": (
            "paired training-seed differences, two-sided 95% Student-t confidence intervals, Cohen's dz, "
            "two-sided exact sign-flip enumeration, and Holm correction over the 12 predeclared primary rows"
        ),
        "primary_paired_statistics_holm": _records_sentence(primary_records),
        "ablation_paired_statistics_holm": (
            _records_sentence(ablation_records)
            + " These 24 return comparisons form one separate exploratory Holm family; the matched MLP remains in the primary family."
        ),
        "model_cost_summary": cost_sentence,
        "hnp_vs_matched_mlp_effect_ci_adjusted_p": _records_sentence(matched),
        "capacity_control_claim_consistent_with_outcome": matched_summary,
        "directly_comparable_stability_metrics_and_results": stability["english_sentence"],
        "language_and_caption_audit": (
            "MANUAL: complete only after inserting all values and inspecting every claim, table, caption, and compiled page."
        ),
        "r3_statistics_ablation_cost_summary": primary_summary + " " + ablation_summary + " " + cost_sentence,
    }


def _render_response_markdown(
    payload: Mapping[str, Any], clauses: Mapping[str, str]
) -> str:
    lines = [
        "# Validated revision-result fill sheet",
        "",
        "> Generated only after strict run, schedule, checkpoint, seed, comparison, manifest, and cost-profile validation.",
        "> Mechanical support labels require both a directionally excluding 95% CI and Holm-adjusted p<0.05.",
        "> Inconclusive does not mean equivalent. Practical significance and final narrative emphasis require human review.",
        "",
        "## Validation receipt",
        "",
        f"- Generated UTC: `{payload['generated_utc']}`",
        f"- Primary run: `{payload['provenance']['primary_dir']}`",
        f"- Ablation artifacts: `{payload['provenance']['ablation_dir']}`",
        f"- Cost profile: `{payload['provenance']['cost_profile_dir']}`",
        "- Inferential unit: one independently trained seed; 10 paired seeds per comparison.",
        "- Fixed evaluation cases: 20 scan-stratified trajectories per condition and jammer mode.",
        "- Primary multiplicity family: 12 predeclared return comparisons.",
        "- Exploratory multiplicity family: 24 HNP-minus-variant return comparisons, kept separate.",
        "",
        "## Response-placeholder clauses",
        "",
    ]
    for name, text in clauses.items():
        lines.extend([f"### `{name}`", "", text, ""])
    lines.extend(
        [
            "## Primary comparison audit (all 12 rows)",
            "",
            "| Condition | Jammer | Comparator | HNP-minus-comparator difference (95% CI) | dz | Holm p | Mechanical label |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for item in payload["primary_comparisons"]:
        raw = item["raw"]
        lines.append(
            f"| {item['condition_label']} | {item['jammer_mode']} | {item['method_b_label']} | "
            f"{_number(raw['mean_difference'])} [{_number(raw['ci95_low'])}, {_number(raw['ci95_high'])}] | "
            f"{_number(raw['effect_dz'])} | {float(raw['p_holm']):.3f} | {item['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Exploratory ablation audit (all 24 rows)",
            "",
            "| Condition | Jammer | Variant | HNP-minus-variant difference (95% CI) | dz | Holm p | Mechanical label |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for item in payload["ablation_comparisons"]:
        raw = item["raw"]
        lines.append(
            f"| {item['condition_label']} | {item['jammer_mode']} | {item['method_b_label']} | "
            f"{_number(raw['mean_difference'])} [{_number(raw['ci95_low'])}, {_number(raw['ci95_high'])}] | "
            f"{_number(raw['effect_dz'])} | {float(raw['p_holm']):.3f} | {item['classification']} |"
        )
    lines.extend(["", "## Required human decisions", ""])
    for item in payload["human_review_required"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def summarize_revision_results(
    *,
    primary_dir: Path,
    ablation_dir: Path,
    cost_profile_dir: Path,
    output_dir: Path,
    manuscript_template: Path,
    response_draft: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all inputs, then atomically prepare the two fill-sheet files."""

    target = Path(output_dir).expanduser().resolve()
    manuscript_output = target / "results_for_manuscript.json"
    response_output = target / "results_for_response.md"
    if not overwrite and (manuscript_output.exists() or response_output.exists()):
        raise ValidationError(
            f"refusing to overwrite an existing fill sheet in {target}; use --overwrite"
        )

    # All reads, recomputations, and prose construction happen before mkdir/write.
    primary, primary_frame = _validate_primary_run(Path(primary_dir))
    ablation_frame, runs, ablation_manifest = _validate_ablation_directory(
        Path(ablation_dir), primary
    )
    training_costs, inference_costs, cost_metadata, cost_hashes = _validate_cost_profile(
        Path(cost_profile_dir), primary
    )
    manuscript_inventory = _placeholder_inventory(
        Path(manuscript_template).expanduser().resolve(), MANUSCRIPT_PATTERN, "manuscript template"
    )
    response_inventory = _placeholder_inventory(
        Path(response_draft).expanduser().resolve(), RESPONSE_PATTERN, "response draft"
    )
    if len(response_inventory) != 17:
        raise ValidationError(
            f"current response mapping requires exactly 17 VERIFIED_RESULT markers; got {len(response_inventory)}"
        )
    response_names = {item["label"] for item in response_inventory}
    expected_response_names = {
        "...",
        "one_paragraph_overall_outcome_with_uncertainty",
        "myopic_vs_hnp_result_with_ci",
        "sequentiality_claim_consistent_with_outcome",
        "hnp_vs_strongest_heuristic_effect_ci_adjusted_p",
        "heuristic_comparison_claim_consistent_with_outcome",
        "number_of_training_seeds",
        "number_of_fixed_evaluation_trajectories",
        "exact_test_details",
        "primary_paired_statistics_holm",
        "ablation_paired_statistics_holm",
        "model_cost_summary",
        "hnp_vs_matched_mlp_effect_ci_adjusted_p",
        "capacity_control_claim_consistent_with_outcome",
        "directly_comparable_stability_metrics_and_results",
        "language_and_caption_audit",
        "r3_statistics_ablation_cost_summary",
    }
    if response_names != expected_response_names:
        raise ValidationError(
            f"response VERIFIED_RESULT names differ; missing={sorted(expected_response_names-response_names)}, "
            f"extra={sorted(response_names-expected_response_names)}"
        )

    primary_records = [
        _comparison_record(row, "primary_holm_12")
        for row in primary_frame.to_dict(orient="records")
    ]
    ablation_records = [
        _comparison_record(row, gate.EXPLORATORY_FAMILY)
        for row in ablation_frame.to_dict(orient="records")
    ]
    performance = _descriptive_performance(primary)
    strongest = _strongest_ordinary(performance)
    supporting = _ablation_supporting_endpoints(runs)
    costs = _cost_summary(training_costs, inference_costs, cost_metadata)
    stability = _stability_summary(primary)
    primary_summary_sentence = _summary_sentence(
        primary_records, "predeclared primary comparisons"
    )
    ablation_summary_sentence = _summary_sentence(
        ablation_records, "exploratory ablation comparisons"
    )
    manuscript_map = _placeholder_map(
        manuscript_inventory,
        performance,
        primary_records,
        ablation_records,
        costs,
        stability,
        primary_summary_sentence,
        ablation_summary_sentence,
    )
    clauses = _response_clauses(primary_records, ablation_records, costs, stability)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated_fill_sheet_not_final_author_interpretation",
        "provenance": {
            "primary_dir": str(primary.spec.path),
            "ablation_dir": str(Path(ablation_dir).expanduser().resolve()),
            "cost_profile_dir": str(Path(cost_profile_dir).expanduser().resolve()),
            "primary_input_artifact_sha256": primary.hashes["input_artifact_sha256"],
            "primary_comparisons_sha256": _sha256(primary.spec.path / "primary_comparisons.csv"),
            "ablation_manifest_sha256": _sha256(Path(ablation_dir).expanduser().resolve() / "result_manifest.json"),
            "ablation_comparisons_sha256": ablation_manifest["outputs"]["ablation_comparisons"]["sha256"],
            "cost_profile_sha256": cost_hashes,
        },
        "protocol": {
            "train_seeds": list(EXPECTED_TRAIN_SEEDS),
            "eval_seeds": list(EXPECTED_EVAL_SEEDS),
            "fixed_trajectories_per_setting": 20,
            "pilot_scan_allocation": "2 scans x 10 replicates",
            "transfer_scan_allocation": "10 scans x 2 replicates",
            "inferential_unit": "independently trained seed after averaging its 20 fixed trajectories",
            "primary_holm_family_size": 12,
            "exploratory_ablation_holm_family_size": 24,
        },
        "primary_comparisons": primary_records,
        "primary_classification_counts": _summarise_classifications(primary_records),
        "primary_summary_english": primary_summary_sentence,
        "descriptive_performance": performance,
        "posthoc_ordinary_comparator_audit": strongest,
        "ablation_comparisons": ablation_records,
        "ablation_classification_counts": _summarise_classifications(ablation_records),
        "ablation_summary_english": ablation_summary_sentence,
        "ablation_supporting_endpoints": supporting,
        "costs": costs,
        "stability": stability,
        "response_placeholder_clauses": clauses,
        "manuscript_placeholder_map": manuscript_map,
        "placeholder_inventory": {
            "manuscript": manuscript_inventory,
            "response": response_inventory,
        },
        "human_review_required": [
            "Choose the abstract's limited set of principal numerical results without significance cherry-picking.",
            "Judge practical importance; the script only classifies CI/Holm direction and never defines a smallest important effect.",
            "Decide the sequentiality claim from all six gamma-zero rows, retaining null or adverse settings.",
            "Decide component-necessity wording from all 24 exploratory rows; do not turn inconclusive results into proof of necessity or equivalence.",
            "If the descriptively best ordinary rule differs from the predeclared comparator, report that ranking without borrowing the predeclared adjusted p-value.",
            "Complete the manuscript-wide language and caption audit after all values and figures are inserted and the document is compiled.",
            "Insert the main figure and its final caption manually.",
            "The polynomial-branch weight subsection requires its separate frozen-checkpoint audit artifact; these three input directories do not supply it.",
            "No switching-cost sensitivity, MAC/FLOP, peak-memory, acquisition-session, new-site, or hardware-benefit claim can be generated from these artifacts.",
            "Cost timing is an engineering profile on the recorded hardware and is not a policy-performance inferential endpoint.",
        ],
    }
    markdown = _render_response_markdown(payload, clauses)
    target.mkdir(parents=True, exist_ok=True)
    manuscript_output.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    response_output.write_text(markdown, encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate canonical primary, ablation, and isolated cost artifacts, "
            "then build auditable manuscript/response fill sheets."
        )
    )
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--cost-profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manuscript-template", type=Path, default=project / "manuscript" / "template.tex"
    )
    parser.add_argument(
        "--response-draft", type=Path, default=project / "response" / "response_draft.md"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = summarize_revision_results(
            primary_dir=args.primary_dir,
            ablation_dir=args.ablation_dir,
            cost_profile_dir=args.cost_profile_dir,
            output_dir=args.output_dir,
            manuscript_template=args.manuscript_template,
            response_draft=args.response_draft,
            overwrite=args.overwrite,
        )
    except (ValidationError, gate.ValidationError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"validated {len(payload['primary_comparisons'])} primary and "
        f"{len(payload['ablation_comparisons'])} exploratory comparisons; wrote fill sheets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
