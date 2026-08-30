#!/usr/bin/env python3
"""Validate frozen runs and build revision-facing ablation statistics.

This module is deliberately a post-processing gate.  It does not train a
model, load a checkpoint into PyTorch, or discover result directories.  The
caller must explicitly provide one canonical primary run and four explicitly
labelled secondary runs.  No output is written until every input has passed
the provenance, schedule, seed-completeness, and scan-balance checks.

The exploratory Holm family contains only return comparisons between the
canonical full HNP model and the four requested variants.  The capacity-
matched MLP remains in the primary comparison family and is therefore not
duplicated here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scipy.stats import t as student_t


SCHEMA_VERSION = 1
EXPECTED_TRAIN_SEEDS = tuple(range(1, 11))
EXPECTED_EVAL_SEEDS = tuple(range(1001, 1021))
EXPECTED_JAMMERS = ("sweeping", "random")
EXPECTED_CONDITIONS = (
    "within_condition_pilot",
    "distance_shift_40cm_10dBm",
    "power_shift_20cm_5dBm",
)
EXPLORATORY_FAMILY = "exploratory_ablation_return_all_conditions_jammers"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(RuntimeError):
    """Raised when a run cannot support the declared revision analysis."""


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    expected_models: tuple[str, ...]
    expected_gamma: float
    variant_method: str | None = None
    variant_label: str | None = None


@dataclass
class ValidatedRun:
    spec: RunSpec
    config: dict[str, Any]
    freeze: dict[str, Any]
    conditions: list[dict[str, Any]]
    evaluation: pd.DataFrame
    seed_summary: pd.DataFrame
    hashes: dict[str, Any]
    validation_counts: dict[str, Any]


def _json_load(path: Path) -> Any:
    if not path.is_file():
        raise ValidationError(f"required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse JSON file {path}: {exc}") from exc


def _csv_load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValidationError(f"required file is missing: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parser exception types
        raise ValidationError(f"cannot parse CSV file {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_json(value: Any, *, runner_style: bool = False) -> str:
    if runner_style:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
    else:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValidationError(f"{name} is missing columns: {missing}")


def _normalise_ints(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must contain integers") from exc


def _validate_static_config(spec: RunSpec, config: Mapping[str, Any]) -> None:
    if config.get("mode") != "formal":
        raise ValidationError(f"{spec.label}: mode must be formal")
    if tuple(config.get("models", ())) != spec.expected_models:
        raise ValidationError(
            f"{spec.label}: models={config.get('models')} but expected "
            f"{list(spec.expected_models)}"
        )
    try:
        gamma = float(config["gamma"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{spec.label}: invalid or missing gamma") from exc
    if not math.isclose(gamma, spec.expected_gamma, rel_tol=0.0, abs_tol=1e-12):
        raise ValidationError(
            f"{spec.label}: gamma={gamma} but expected {spec.expected_gamma}"
        )
    train_seeds = _normalise_ints(config.get("train_seeds", ()), name="train_seeds")
    eval_seeds = _normalise_ints(config.get("eval_seeds", ()), name="eval_seeds")
    jammers = tuple(str(value) for value in config.get("jammer_modes", ()))
    if train_seeds != EXPECTED_TRAIN_SEEDS:
        raise ValidationError(
            f"{spec.label}: train_seeds must be {EXPECTED_TRAIN_SEEDS}, got {train_seeds}"
        )
    if eval_seeds != EXPECTED_EVAL_SEEDS:
        raise ValidationError(
            f"{spec.label}: eval_seeds must be {EXPECTED_EVAL_SEEDS}, got {eval_seeds}"
        )
    if jammers != EXPECTED_JAMMERS:
        raise ValidationError(
            f"{spec.label}: jammer_modes must be {EXPECTED_JAMMERS}, got {jammers}"
        )
    if int(config.get("evaluation_episodes_per_seed", -1)) != 1:
        raise ValidationError(
            f"{spec.label}: exactly one evaluation trajectory per eval seed is required"
        )
    if int(config.get("train_episodes", -1)) != 200:
        raise ValidationError(f"{spec.label}: formal train_episodes must equal 200")


def _schedule_config_hash(config: Mapping[str, Any]) -> str:
    """Hash all common protocol fields while allowing model and gamma changes."""

    comparable = {
        key: value
        for key, value in config.items()
        if key not in {"models", "gamma"}
    }
    return _sha256_json(comparable)


def _validate_freeze_and_hashes(
    spec: RunSpec, config: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    expected_config_hash = _sha256_json(dict(config), runner_style=True)
    recorded_config_hash = str(freeze.get("config_sha256", "")).lower()
    if recorded_config_hash != expected_config_hash:
        raise ValidationError(
            f"{spec.label}: config hash mismatch; recorded={recorded_config_hash}, "
            f"computed={expected_config_hash}"
        )

    code_map = freeze.get("core_code_sha256")
    if not isinstance(code_map, Mapping) or not code_map:
        raise ValidationError(
            f"{spec.label}: FROZEN_BEFORE_EVALUATION.json lacks core_code_sha256"
        )
    clean_code_map = {str(key): str(value).lower() for key, value in code_map.items()}
    bad_code_hashes = {
        key: value for key, value in clean_code_map.items() if not HEX64.fullmatch(value)
    }
    if bad_code_hashes:
        raise ValidationError(f"{spec.label}: invalid code hashes: {bad_code_hashes}")

    schedule_record = freeze.get("predeclared_evaluation_schedule")
    if not isinstance(schedule_record, Mapping):
        raise ValidationError(
            f"{spec.label}: missing predeclared_evaluation_schedule freeze record"
        )
    schedule_relative = schedule_record.get("path")
    schedule_hash = str(schedule_record.get("sha256", "")).lower()
    if not schedule_relative or not HEX64.fullmatch(schedule_hash):
        raise ValidationError(f"{spec.label}: invalid frozen schedule record")
    schedule_path = spec.path / str(schedule_relative)
    computed_schedule_hash = _sha256_file(schedule_path)
    if computed_schedule_hash != schedule_hash:
        raise ValidationError(
            f"{spec.label}: schedule hash mismatch; recorded={schedule_hash}, "
            f"computed={computed_schedule_hash}"
        )

    normalizer_path = spec.path / "normalizer.json"
    normalizer_hash = _sha256_file(normalizer_path)
    return {
        "config_sha256": expected_config_hash,
        "schedule_config_sha256": _schedule_config_hash(config),
        "predeclared_schedule_sha256": schedule_hash,
        "core_code_sha256": clean_code_map,
        "core_code_bundle_sha256": _sha256_json(clean_code_map),
        "normalizer_sha256": normalizer_hash,
    }


def _validate_checkpoints(
    spec: RunSpec, freeze: Mapping[str, Any]
) -> tuple[int, dict[str, str]]:
    records = freeze.get("checkpoints")
    if not isinstance(records, list):
        raise ValidationError(f"{spec.label}: checkpoint manifest is missing")
    expected = {
        (model, jammer, seed)
        for model in spec.expected_models
        for jammer in EXPECTED_JAMMERS
        for seed in EXPECTED_TRAIN_SEEDS
    }
    seen: dict[tuple[str, str, int], str] = {}
    path_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValidationError(f"{spec.label}: malformed checkpoint record")
        try:
            key = (
                str(record["model"]),
                str(record["jammer_mode"]),
                int(record["train_seed"]),
            )
            relative_path = str(record["path"])
            recorded_hash = str(record["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"{spec.label}: malformed checkpoint record") from exc
        if key in seen:
            raise ValidationError(f"{spec.label}: duplicate checkpoint key {key}")
        if not HEX64.fullmatch(recorded_hash):
            raise ValidationError(f"{spec.label}: invalid checkpoint hash for {key}")
        checkpoint = spec.path / relative_path
        computed_hash = _sha256_file(checkpoint)
        if computed_hash != recorded_hash:
            raise ValidationError(f"{spec.label}: checkpoint hash mismatch for {key}")
        seen[key] = relative_path
        path_hashes[relative_path] = computed_hash
    actual = set(seen)
    if actual != expected:
        raise ValidationError(
            f"{spec.label}: checkpoint combinations differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return len(seen), path_hashes


def _condition_scan_map(
    spec: RunSpec, conditions: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[int, ...]]:
    names = tuple(str(item.get("name")) for item in conditions)
    if names != EXPECTED_CONDITIONS:
        raise ValidationError(
            f"{spec.label}: conditions must be ordered as {EXPECTED_CONDITIONS}, got {names}"
        )
    result: dict[str, tuple[int, ...]] = {}
    for item in conditions:
        name = str(item["name"])
        scans = tuple(sorted(_normalise_ints(item.get("scan_ids", ()), name="scan_ids")))
        if not scans or len(scans) != len(set(scans)):
            raise ValidationError(f"{spec.label}: invalid scan_ids for {name}: {scans}")
        result[name] = scans
    return result


def _normalised_group_seed(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _trajectory_signature(frame: pd.DataFrame) -> tuple[tuple[Any, ...], ...]:
    signature_columns = [
        "eval_seed",
        "scan_id",
        "scan_replicate",
        "trajectory_seed",
    ]
    optional = ["start_index", "window_start"]
    signature_columns.extend(column for column in optional if column in frame.columns)
    ordered = frame.sort_values(["eval_seed", "scan_replicate", "scan_id"])
    return tuple(
        tuple(row)
        for row in ordered[signature_columns].itertuples(index=False, name=None)
    )


def _validate_evaluation(
    spec: RunSpec,
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "condition",
        "jammer_mode",
        "method",
        "train_seed",
        "eval_seed",
        "scan_id",
        "scan_replicate",
        "trajectory_seed",
        "return",
        "collisions",
        "collision_rate",
        "switches",
        "switch_rate",
    }
    _require_columns(frame, required, f"{spec.label}/evaluation_episodes.csv")
    if frame.empty:
        raise ValidationError(f"{spec.label}: evaluation_episodes.csv is empty")
    if not np.isfinite(
        frame[["return", "collisions", "collision_rate", "switches", "switch_rate"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValidationError(f"{spec.label}: evaluation metrics contain non-finite values")

    scan_map = _condition_scan_map(spec, conditions)
    observed_conditions = tuple(
        name for name in EXPECTED_CONDITIONS if name in set(frame["condition"])
    )
    if observed_conditions != EXPECTED_CONDITIONS or set(frame["condition"]) != set(
        EXPECTED_CONDITIONS
    ):
        raise ValidationError(f"{spec.label}: evaluation condition set is incomplete")
    if set(frame["jammer_mode"].astype(str)) != set(EXPECTED_JAMMERS):
        raise ValidationError(f"{spec.label}: evaluation jammer set is incomplete")

    expected_trajectories = len(EXPECTED_EVAL_SEEDS) * int(
        config["evaluation_episodes_per_seed"]
    )
    if expected_trajectories != 20:
        raise ValidationError(
            f"{spec.label}: revision protocol requires exactly 20 trajectories, "
            f"computed {expected_trajectories}"
        )

    group_columns = ["condition", "jammer_mode", "method", "train_seed"]
    signatures: dict[tuple[str, str], tuple[tuple[Any, ...], ...]] = {}
    group_count = 0
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        condition, jammer, method, train_seed_raw = keys
        train_seed = _normalised_group_seed(train_seed_raw)
        group_count += 1
        if len(group) != expected_trajectories:
            raise ValidationError(
                f"{spec.label}: {keys} has {len(group)} trajectories; expected 20"
            )
        eval_seeds = tuple(sorted(int(value) for value in group["eval_seed"]))
        if eval_seeds != EXPECTED_EVAL_SEEDS:
            raise ValidationError(
                f"{spec.label}: {keys} does not contain each fixed eval seed exactly once"
            )
        allowed_scans = scan_map[str(condition)]
        observed_scans = tuple(sorted(int(value) for value in group["scan_id"].unique()))
        if observed_scans != allowed_scans:
            raise ValidationError(
                f"{spec.label}: {keys} scan set {observed_scans} != {allowed_scans}"
            )
        counts = group["scan_id"].astype(int).value_counts().reindex(allowed_scans)
        if counts.isna().any() or int(counts.max() - counts.min()) > 1:
            raise ValidationError(f"{spec.label}: {keys} is not scan-balanced: {counts.to_dict()}")
        if group.duplicated(["eval_seed", "scan_replicate"]).any():
            raise ValidationError(f"{spec.label}: {keys} has duplicate trajectory slots")

        signature = _trajectory_signature(group)
        setting = (str(condition), str(jammer))
        reference = signatures.setdefault(setting, signature)
        if signature != reference:
            raise ValidationError(
                f"{spec.label}: methods do not share the same trajectories for {setting}"
            )
        if train_seed is not None and train_seed not in EXPECTED_TRAIN_SEEDS:
            raise ValidationError(f"{spec.label}: unexpected learned train seed {train_seed}")

    # Every declared learned model must have all 10 seeds in every setting.
    learned = frame[frame["train_seed"].notna()].copy()
    learned["train_seed"] = learned["train_seed"].astype(int)
    expected_learned = {
        (condition, jammer, model, seed)
        for condition in EXPECTED_CONDITIONS
        for jammer in EXPECTED_JAMMERS
        for model in spec.expected_models
        for seed in EXPECTED_TRAIN_SEEDS
    }
    actual_learned = set(
        learned[["condition", "jammer_mode", "method", "train_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual_learned != expected_learned:
        raise ValidationError(
            f"{spec.label}: learned evaluation combinations differ; "
            f"missing={sorted(expected_learned - actual_learned)}, "
            f"extra={sorted(actual_learned - expected_learned)}"
        )

    # A baseline may be jammer-specific, but if present for one jammer it must
    # cover all three conditions for that jammer.
    baseline = frame[frame["train_seed"].isna()]
    for (jammer, method), subset in baseline.groupby(
        ["jammer_mode", "method"], sort=False
    ):
        if set(subset["condition"]) != set(EXPECTED_CONDITIONS):
            raise ValidationError(
                f"{spec.label}: baseline {method}/{jammer} is missing a condition"
            )

    return {
        "evaluation_rows": int(len(frame)),
        "trajectory_groups": int(group_count),
        "trajectories_per_group": expected_trajectories,
        "learned_combinations": int(len(actual_learned)),
        "baseline_methods": sorted(str(value) for value in baseline["method"].unique()),
    }


def _validate_training_and_costs(
    spec: RunSpec, config: Mapping[str, Any], training: pd.DataFrame, costs: pd.DataFrame
) -> dict[str, int]:
    _require_columns(
        training,
        {"model", "jammer_mode", "train_seed", "episode"},
        f"{spec.label}/training_history.csv",
    )
    _require_columns(
        costs,
        {"model", "jammer_mode", "train_seed", "trainable_parameters"},
        f"{spec.label}/model_costs.csv",
    )
    expected_keys = {
        (model, jammer, seed)
        for model in spec.expected_models
        for jammer in EXPECTED_JAMMERS
        for seed in EXPECTED_TRAIN_SEEDS
    }
    training_keys: set[tuple[str, str, int]] = set()
    for key, group in training.groupby(["model", "jammer_mode", "train_seed"]):
        normalised = (str(key[0]), str(key[1]), int(key[2]))
        training_keys.add(normalised)
        episodes = tuple(sorted(int(value) for value in group["episode"]))
        expected_episodes = tuple(range(int(config["train_episodes"])))
        if episodes != expected_episodes:
            raise ValidationError(
                f"{spec.label}: incomplete/duplicate training episodes for {normalised}"
            )
    if training_keys != expected_keys:
        raise ValidationError(f"{spec.label}: training model/seed schedule is incomplete")

    cost_rows = costs.copy()
    cost_rows["train_seed"] = cost_rows["train_seed"].astype(int)
    if cost_rows.duplicated(["model", "jammer_mode", "train_seed"]).any():
        raise ValidationError(f"{spec.label}: duplicate model-cost rows")
    cost_keys = set(
        cost_rows[["model", "jammer_mode", "train_seed"]].itertuples(
            index=False, name=None
        )
    )
    if cost_keys != expected_keys:
        raise ValidationError(f"{spec.label}: model-cost schedule is incomplete")
    return {"training_rows": int(len(training)), "model_cost_rows": int(len(costs))}


def _validate_seed_summary(
    spec: RunSpec, evaluation: pd.DataFrame, summary: pd.DataFrame
) -> dict[str, int]:
    metrics = ["return", "collisions", "collision_rate", "switches", "switch_rate"]
    required = {
        "condition",
        "condition_role",
        "distance_cm",
        "power_dbm",
        "jammer_mode",
        "method",
        "train_seed",
        *metrics,
    }
    _require_columns(summary, required, f"{spec.label}/seed_summary.csv")
    keys = [
        "condition",
        "condition_role",
        "distance_cm",
        "power_dbm",
        "jammer_mode",
        "method",
        "train_seed",
    ]
    expected = (
        evaluation.groupby(keys, as_index=False, dropna=False)[metrics]
        .mean()
        .sort_values(keys, na_position="last")
        .reset_index(drop=True)
    )
    actual = summary[keys + metrics].sort_values(keys, na_position="last").reset_index(drop=True)
    if len(expected) != len(actual):
        raise ValidationError(
            f"{spec.label}: seed_summary row count does not match evaluation aggregation"
        )
    for key in keys[:-1]:
        if not expected[key].astype(str).equals(actual[key].astype(str)):
            raise ValidationError(f"{spec.label}: seed_summary key mismatch in {key}")
    exp_seed = expected["train_seed"].fillna(-1).astype(int)
    act_seed = actual["train_seed"].fillna(-1).astype(int)
    if not exp_seed.equals(act_seed):
        raise ValidationError(f"{spec.label}: seed_summary train_seed keys mismatch")
    if not np.allclose(
        expected[metrics].to_numpy(dtype=float),
        actual[metrics].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValidationError(f"{spec.label}: seed_summary values do not match evaluation")
    learned_rows = summary[summary["train_seed"].notna()]
    return {
        "seed_summary_rows": int(len(summary)),
        "learned_seed_summary_rows": int(len(learned_rows)),
    }


def _validate_one_run(spec: RunSpec) -> ValidatedRun:
    run_path = spec.path.expanduser().resolve()
    if not run_path.is_dir():
        raise ValidationError(f"run directory does not exist: {run_path}")
    spec = RunSpec(
        label=spec.label,
        path=run_path,
        expected_models=spec.expected_models,
        expected_gamma=spec.expected_gamma,
        variant_method=spec.variant_method,
        variant_label=spec.variant_label,
    )
    config = _json_load(run_path / "config.json")
    freeze = _json_load(run_path / "FROZEN_BEFORE_EVALUATION.json")
    conditions = _json_load(run_path / "evaluation_conditions.json")
    if not isinstance(config, dict) or not isinstance(freeze, dict):
        raise ValidationError(f"{spec.label}: config/freeze must be JSON objects")
    if not isinstance(conditions, list):
        raise ValidationError(f"{spec.label}: evaluation_conditions must be a list")

    _validate_static_config(spec, config)
    hashes = _validate_freeze_and_hashes(spec, config, freeze)
    checkpoint_count, checkpoint_hashes = _validate_checkpoints(spec, freeze)
    hashes["checkpoint_sha256"] = checkpoint_hashes

    evaluation = _csv_load(run_path / "evaluation_episodes.csv")
    seed_summary = _csv_load(run_path / "seed_summary.csv")
    training = _csv_load(run_path / "training_history.csv")
    costs = _csv_load(run_path / "model_costs.csv")
    counts: dict[str, Any] = {"checkpoints": checkpoint_count}
    counts.update(_validate_evaluation(spec, config, conditions, evaluation))
    counts.update(_validate_training_and_costs(spec, config, training, costs))
    counts.update(_validate_seed_summary(spec, evaluation, seed_summary))

    hashes["input_artifact_sha256"] = {
        name: _sha256_file(run_path / name)
        for name in (
            "config.json",
            "FROZEN_BEFORE_EVALUATION.json",
            "evaluation_conditions.json",
            "evaluation_episodes.csv",
            "seed_summary.csv",
            "training_history.csv",
            "model_costs.csv",
        )
    }
    return ValidatedRun(
        spec=spec,
        config=config,
        freeze=freeze,
        conditions=conditions,
        evaluation=evaluation,
        seed_summary=seed_summary,
        hashes=hashes,
        validation_counts=counts,
    )


def _validate_cross_run_provenance(runs: Sequence[ValidatedRun]) -> dict[str, str]:
    if not runs:
        raise ValidationError("no runs were provided")
    reference = runs[0]
    fields = (
        "schedule_config_sha256",
        "predeclared_schedule_sha256",
        "core_code_bundle_sha256",
        "normalizer_sha256",
    )
    for run in runs[1:]:
        for field in fields:
            if run.hashes[field] != reference.hashes[field]:
                raise ValidationError(
                    f"cross-run {field} mismatch: {reference.spec.label}="
                    f"{reference.hashes[field]}, {run.spec.label}={run.hashes[field]}"
                )
        if run.freeze.get("threshold") != reference.freeze.get("threshold"):
            raise ValidationError(
                f"cross-run frozen threshold mismatch: {run.spec.label}"
            )
    return {field: str(reference.hashes[field]) for field in fields}


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValidationError("paired differences must be finite and non-empty")
    nonzero = values[~np.isclose(values, 0.0, atol=1e-12, rtol=0.0)]
    if nonzero.size == 0:
        return 1.0
    if nonzero.size > 20:
        raise ValidationError("exact sign-flip enumeration is limited to 20 pairs")
    observed = abs(float(nonzero.mean()))
    total = 1 << int(nonzero.size)
    extreme = 0
    for bits in range(total):
        signs = np.fromiter(
            (1.0 if bits & (1 << index) else -1.0 for index in range(nonzero.size)),
            dtype=np.float64,
            count=nonzero.size,
        )
        extreme += int(abs(float(np.mean(nonzero * signs))) >= observed - 1e-12)
    return float(extreme / total)


def _cohen_dz(differences: np.ndarray) -> float:
    if differences.size < 2:
        return float("nan")
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    if math.isclose(sd, 0.0, abs_tol=1e-15):
        if math.isclose(mean, 0.0, abs_tol=1e-15):
            return 0.0
        return math.copysign(float("inf"), mean)
    return mean / sd


def _paired_row(
    first: np.ndarray,
    second: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    method_b: str,
) -> dict[str, Any]:
    if first.shape != second.shape or first.size != 10:
        raise ValidationError(
            f"paired comparison {method_b} requires exactly 10 matched seeds"
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValidationError(f"paired comparison {method_b} contains non-finite values")
    differences = first - second
    n_pairs = int(differences.size)
    mean_difference = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    critical = float(student_t.ppf(0.975, n_pairs - 1))
    margin = critical * sd_difference / math.sqrt(n_pairs)
    return {
        **metadata,
        "method_a": "hnp",
        "method_b": method_b,
        "metric": "return",
        "n_pairs": n_pairs,
        "mean_a": float(first.mean()),
        "sd_a": float(first.std(ddof=1)),
        "mean_b": float(second.mean()),
        "sd_b": float(second.std(ddof=1)),
        "mean_difference": mean_difference,
        "ci95_low": mean_difference - margin,
        "ci95_high": mean_difference + margin,
        "effect_dz": _cohen_dz(differences),
        "p_exact": _exact_sign_flip_pvalue(differences),
        "p_holm": float("nan"),
        "holm_family": EXPLORATORY_FAMILY,
    }


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValidationError("Holm adjustment requires finite p-values")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    family_size = values.size
    for rank, original_index in enumerate(order):
        running = max(running, (family_size - rank) * values[original_index])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def _learned_seed_rows(run: ValidatedRun, method: str) -> pd.DataFrame:
    rows = run.seed_summary[
        (run.seed_summary["method"] == method)
        & run.seed_summary["train_seed"].notna()
    ].copy()
    rows["train_seed"] = rows["train_seed"].astype(int)
    keys = ["condition", "jammer_mode", "train_seed"]
    if rows.duplicated(keys).any():
        raise ValidationError(f"{run.spec.label}: duplicate seed-summary rows for {method}")
    return rows


def _build_ablation_comparisons(
    primary: ValidatedRun, variants: Sequence[ValidatedRun]
) -> pd.DataFrame:
    full = _learned_seed_rows(primary, "hnp")
    rows: list[dict[str, Any]] = []
    condition_order = {name: index for index, name in enumerate(EXPECTED_CONDITIONS)}
    jammer_order = {name: index for index, name in enumerate(EXPECTED_JAMMERS)}
    variant_order = {run.spec.variant_label: index for index, run in enumerate(variants)}

    for variant_run in variants:
        if not variant_run.spec.variant_method or not variant_run.spec.variant_label:
            raise ValidationError(f"{variant_run.spec.label}: variant metadata is incomplete")
        variant = _learned_seed_rows(variant_run, variant_run.spec.variant_method)
        join_keys = [
            "condition",
            "condition_role",
            "distance_cm",
            "power_dbm",
            "jammer_mode",
            "train_seed",
        ]
        joined = full[join_keys + ["return"]].merge(
            variant[join_keys + ["return"]],
            on=join_keys,
            how="outer",
            suffixes=("_hnp", "_variant"),
            indicator=True,
            validate="one_to_one",
        )
        if set(joined["_merge"]) != {"both"}:
            bad = joined[joined["_merge"] != "both"]
            raise ValidationError(
                f"{variant_run.spec.label}: variant/full seed keys differ: "
                f"{bad[join_keys + ['_merge']].to_dict(orient='records')[:5]}"
            )
        for setting, group in joined.groupby(
            [
                "condition",
                "condition_role",
                "distance_cm",
                "power_dbm",
                "jammer_mode",
            ],
            sort=False,
        ):
            ordered = group.sort_values("train_seed")
            seeds = tuple(int(value) for value in ordered["train_seed"])
            if seeds != EXPECTED_TRAIN_SEEDS:
                raise ValidationError(
                    f"{variant_run.spec.label}/{setting}: paired seed schedule is incomplete"
                )
            condition, role, distance_cm, power_dbm, jammer = setting
            rows.append(
                _paired_row(
                    ordered["return_hnp"].to_numpy(dtype=float),
                    ordered["return_variant"].to_numpy(dtype=float),
                    metadata={
                        "condition": condition,
                        "condition_role": role,
                        "distance_cm": int(distance_cm),
                        "power_dbm": int(power_dbm),
                        "jammer_mode": jammer,
                        "variant_run": variant_run.spec.label,
                    },
                    method_b=variant_run.spec.variant_label,
                )
            )

    frame = pd.DataFrame(rows)
    expected_rows = len(variants) * len(EXPECTED_CONDITIONS) * len(EXPECTED_JAMMERS)
    if len(frame) != expected_rows:
        raise ValidationError(
            f"expected {expected_rows} exploratory comparison rows, got {len(frame)}"
        )
    frame["p_holm"] = _holm_adjust(frame["p_exact"].to_numpy(dtype=float))
    frame["_condition_order"] = frame["condition"].map(condition_order)
    frame["_jammer_order"] = frame["jammer_mode"].map(jammer_order)
    frame["_variant_order"] = frame["method_b"].map(variant_order)
    frame = frame.sort_values(
        ["_condition_order", "_jammer_order", "_variant_order"]
    ).drop(columns=["_condition_order", "_jammer_order", "_variant_order"])
    return frame.reset_index(drop=True)


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )


def build_revision_artifacts(
    *,
    primary_dir: Path,
    gamma0_dir: Path,
    no_polynomial_dir: Path,
    no_layernorm_dir: Path,
    no_dueling_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all runs, then write ablation comparisons and a manifest."""

    specs = [
        RunSpec("primary", Path(primary_dir), ("hnp", "matched"), 0.95),
        RunSpec(
            "gamma0",
            Path(gamma0_dir),
            ("hnp",),
            0.0,
            variant_method="hnp",
            variant_label="hnp_gamma0",
        ),
        RunSpec(
            "no_polynomial",
            Path(no_polynomial_dir),
            ("no_polynomial",),
            0.95,
            variant_method="no_polynomial",
            variant_label="no_polynomial",
        ),
        RunSpec(
            "no_layernorm",
            Path(no_layernorm_dir),
            ("no_layernorm",),
            0.95,
            variant_method="no_layernorm",
            variant_label="no_layernorm",
        ),
        RunSpec(
            "no_dueling",
            Path(no_dueling_dir),
            ("no_dueling",),
            0.95,
            variant_method="no_dueling",
            variant_label="no_dueling",
        ),
    ]
    resolved_inputs = [spec.path.expanduser().resolve() for spec in specs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValidationError("each run role must point to a distinct directory")

    target = Path(output_dir).expanduser().resolve()
    comparison_path = target / "ablation_comparisons.csv"
    manifest_path = target / "result_manifest.json"
    if not overwrite and (comparison_path.exists() or manifest_path.exists()):
        raise ValidationError(
            f"refusing to overwrite existing output in {target}; use --overwrite"
        )

    # Validation performs all reads before any output directory or file is made.
    runs = [_validate_one_run(spec) for spec in specs]
    cross_run_hashes = _validate_cross_run_provenance(runs)
    comparisons = _build_ablation_comparisons(runs[0], runs[1:])

    target.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(comparison_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated",
        "statement": (
            "All statistics were produced only after config, code, frozen "
            "trajectory schedule, checkpoints, seeds, trajectory counts, and "
            "scan balance passed validation. The Holm family is exploratory "
            "and separate from the primary HNP-versus-threshold/matched family."
        ),
        "protocol": {
            "train_seeds": list(EXPECTED_TRAIN_SEEDS),
            "eval_seeds": list(EXPECTED_EVAL_SEEDS),
            "trajectories_per_method_training_seed_condition_jammer": 20,
            "conditions": list(EXPECTED_CONDITIONS),
            "jammer_modes": list(EXPECTED_JAMMERS),
            "metric": "return",
            "ci": "two-sided 95% Student-t interval for paired seed differences",
            "effect_size": "Cohen's dz",
            "test": "two-sided exact sign-flip enumeration",
            "holm_family": EXPLORATORY_FAMILY,
            "holm_family_size": int(len(comparisons)),
        },
        "cross_run_provenance": cross_run_hashes,
        "runs": {
            run.spec.label: {
                "path": str(run.spec.path),
                "expected_models": list(run.spec.expected_models),
                "gamma": float(run.config["gamma"]),
                "config_sha256": run.hashes["config_sha256"],
                "core_code_bundle_sha256": run.hashes[
                    "core_code_bundle_sha256"
                ],
                "predeclared_schedule_sha256": run.hashes[
                    "predeclared_schedule_sha256"
                ],
                "normalizer_sha256": run.hashes["normalizer_sha256"],
                "input_artifact_sha256": run.hashes["input_artifact_sha256"],
                "checkpoint_sha256": run.hashes["checkpoint_sha256"],
                "validation_counts": run.validation_counts,
            }
            for run in runs
        },
        "outputs": {
            "ablation_comparisons": {
                "path": comparison_path.name,
                "sha256": _sha256_file(comparison_path),
                "rows": int(len(comparisons)),
                "columns": list(comparisons.columns),
            }
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one canonical primary run plus gamma=0/no-polynomial/"
            "no-LayerNorm/no-dueling runs, then build exploratory paired "
            "ablation statistics."
        )
    )
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--gamma0", type=Path, required=True)
    parser.add_argument("--no-polynomial", type=Path, required=True)
    parser.add_argument("--no-layernorm", type=Path, required=True)
    parser.add_argument("--no-dueling", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the two named generated artifacts if they exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_revision_artifacts(
            primary_dir=args.primary,
            gamma0_dir=args.gamma0,
            no_polynomial_dir=args.no_polynomial,
            no_layernorm_dir=args.no_layernorm,
            no_dueling_dir=args.no_dueling,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"validated {len(manifest['runs'])} runs; wrote "
        f"{manifest['outputs']['ablation_comparisons']['rows']} comparisons"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
