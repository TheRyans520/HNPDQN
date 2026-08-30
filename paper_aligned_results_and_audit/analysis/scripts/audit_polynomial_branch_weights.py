"""Checkpoint-only descriptive audit of HNP polynomial-branch weights.

The script reads only formal run configuration, freeze/hash manifests, and HNP
checkpoint tensors.  It never opens episode evaluations, seed summaries, or
cross-configuration policy outcomes.  Run it after the canonical
``formal_primary_v3`` checkpoints are complete; do not use it to select a run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REVISION_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REVISION_ROOT / "experiment_v2"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import ExperimentConfig  # noqa: E402
from src import models as models_module  # noqa: E402
from src.models import HNPQNetwork, build_model  # noqa: E402


EXPECTED_SEEDS: tuple[int, ...] = tuple(range(1, 11))
EXPECTED_JAMMER_MODES: tuple[str, ...] = ("sweeping", "random")
PROJECTION_WEIGHT_KEY = "projection.0.weight"
POLYNOMIAL_LAYERNORM_KEY = "polynomial.layer_norm.weight"
MODELS_SOURCE_PATH = Path(models_module.__file__).resolve()
MODELS_FREEZE_KEY = "src/models.py"
ALLOWED_EXECUTION_DEVICE = re.compile(r"^(?:auto|cpu|cuda(?::[0-9]+)?)$")
INTERPRETATION_LIMIT = (
    "Descriptive raw-weight magnitude only. It is scale- and reparameterization-"
    "dependent, is coupled to learned LayerNorm and upstream activation scales, "
    "ignores activation frequency and sign cancellation, and is not causal "
    "feature importance, branch contribution, robustness, or interpretability evidence."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def discover_frozen_run_directories(
    roots: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Find run directories by freeze manifest, never by result-table files."""

    run_directories: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            if root.name != "FROZEN_BEFORE_EVALUATION.json":
                raise ValueError(
                    "a file root must be FROZEN_BEFORE_EVALUATION.json"
                )
            run_directories.add(root.parent)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        direct_manifest = root / "FROZEN_BEFORE_EVALUATION.json"
        if direct_manifest.is_file():
            run_directories.add(root)
            continue
        run_directories.update(
            path.parent.resolve()
            for path in root.rglob("FROZEN_BEFORE_EVALUATION.json")
        )
    if not run_directories:
        raise FileNotFoundError("no frozen formal run manifest was found")
    return tuple(sorted(run_directories))


def _json_normalize(value: Any) -> Any:
    """Normalize tuple/list representation without silently dropping fields."""

    return json.loads(json.dumps(value, sort_keys=True))


def _parse_complete_config(
    payload: Mapping[str, Any], *, source: Path
) -> tuple[ExperimentConfig, dict[str, Any]]:
    formal_keys = set(ExperimentConfig.preset("formal").to_dict())
    payload_keys = set(payload)
    if payload_keys != formal_keys:
        raise ValueError(
            f"incomplete config fields in {source}; "
            f"missing={sorted(formal_keys - payload_keys)}, "
            f"extra={sorted(payload_keys - formal_keys)}"
        )
    try:
        config = ExperimentConfig(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid config in {source}: {exc}") from exc
    normalized_payload = _json_normalize(dict(payload))
    normalized_config = _json_normalize(config.to_dict())
    if normalized_payload != normalized_config:
        raise ValueError(
            f"config values in {source} require implicit coercion and are not exact"
        )
    return config, normalized_config


def _validate_primary_run_config(
    payload: Mapping[str, Any], *, source: Path
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Require the complete formal-primary preset, exempting only device choice."""

    config, normalized = _parse_complete_config(payload, source=source)
    formal = _json_normalize(ExperimentConfig.preset("formal").to_dict())
    drift = {
        field: (normalized[field], formal[field])
        for field in formal
        if field != "device" and normalized[field] != formal[field]
    }
    if drift:
        raise ValueError(f"noncanonical formal-primary run config in {source}: {drift}")
    device = normalized["device"]
    if not isinstance(device, str) or ALLOWED_EXECUTION_DEVICE.fullmatch(device) is None:
        raise ValueError(
            f"invalid execution device in {source}: {device!r}; device is the only "
            "field exempted from exact formal-primary preset equality"
        )
    return config, normalized


def _validate_models_source_hash(freeze: Mapping[str, Any], *, run_dir: Path) -> str:
    core_hashes = freeze.get("core_code_sha256")
    if not isinstance(core_hashes, Mapping):
        raise ValueError(f"freeze manifest has no core-code hashes: {run_dir}")
    recorded_hash = core_hashes.get(MODELS_FREEZE_KEY)
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise ValueError(
            f"freeze manifest has no {MODELS_FREEZE_KEY!r} hash: {run_dir}"
        )
    expected_source = (EXPERIMENT_ROOT / MODELS_FREEZE_KEY).resolve()
    if MODELS_SOURCE_PATH != expected_source:
        raise ValueError(
            "imported src.models does not resolve to the experiment source tree: "
            f"{MODELS_SOURCE_PATH} != {expected_source}"
        )
    actual_hash = _sha256(MODELS_SOURCE_PATH)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"{MODELS_FREEZE_KEY} source hash mismatch in {run_dir}: "
            f"frozen={recorded_hash}, imported={actual_hash}"
        )
    return actual_hash


def canonical_hnp_checkpoint_records(
    roots: Iterable[str | Path],
    *,
    jammer_mode: str,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
    required_run_label: str = "formal_primary_v3",
) -> list[dict[str, Any]]:
    """Verify freeze hashes, formal protocol, model identity and seed coverage."""

    if jammer_mode == "both":
        reusable_roots = tuple(roots)
        combined: list[dict[str, Any]] = []
        for single_mode in EXPECTED_JAMMER_MODES:
            combined.extend(
                canonical_hnp_checkpoint_records(
                    reusable_roots,
                    jammer_mode=single_mode,
                    expected_seeds=expected_seeds,
                    required_run_label=required_run_label,
                )
            )
        return combined
    if jammer_mode not in set(EXPECTED_JAMMER_MODES):
        raise ValueError("jammer_mode must be 'both', 'sweeping', or 'random'")
    if not str(required_run_label).strip():
        raise ValueError("required_run_label must be non-empty")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    records: dict[int, dict[str, Any]] = {}
    for run_dir in discover_frozen_run_directories(roots):
        if required_run_label.lower() not in run_dir.name.lower():
            continue
        freeze_path = run_dir / "FROZEN_BEFORE_EVALUATION.json"
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        freeze = _read_json(freeze_path)
        run_config_payload = _read_json(config_path)
        recorded_config_hash = freeze.get("config_sha256")
        actual_config_hash = _config_hash(run_config_payload)
        if recorded_config_hash != actual_config_hash:
            raise ValueError(f"config hash mismatch in {run_dir}")
        run_config, normalized_run_config = _validate_primary_run_config(
            run_config_payload, source=config_path
        )
        models_source_hash = _validate_models_source_hash(freeze, run_dir=run_dir)

        checkpoint_entries = freeze.get("checkpoints")
        if not isinstance(checkpoint_entries, list):
            raise ValueError(f"freeze manifest has no checkpoint list: {freeze_path}")
        for entry in checkpoint_entries:
            if not isinstance(entry, dict):
                raise ValueError(f"invalid checkpoint entry in {freeze_path}")
            if str(entry.get("model", "")).lower() != "hnp":
                continue
            if str(entry.get("jammer_mode", "")) != jammer_mode:
                continue
            seed = int(entry["train_seed"])
            relative_path = Path(str(entry["path"]))
            checkpoint = (run_dir / relative_path).resolve()
            try:
                checkpoint.relative_to(run_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"checkpoint escapes its frozen run directory: {checkpoint}"
                ) from exc
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            actual_hash = _sha256(checkpoint)
            if actual_hash != str(entry.get("sha256", "")):
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            if seed in records:
                raise ValueError(
                    f"duplicate HNP checkpoint for seed {seed}: "
                    f"{records[seed]['checkpoint_path']} and {checkpoint}"
                )

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if str(payload.get("model_name", "")).lower() != "hnp":
                raise ValueError(f"checkpoint is not HNP: {checkpoint}")
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"seed metadata mismatch: {checkpoint}")
            if str(payload.get("training_jammer_mode", "")) != jammer_mode:
                raise ValueError(f"jammer-mode metadata mismatch: {checkpoint}")
            checkpoint_config_payload = payload.get("config")
            if not isinstance(checkpoint_config_payload, Mapping):
                raise ValueError(f"checkpoint has no embedded config: {checkpoint}")
            checkpoint_config, normalized_checkpoint_config = _parse_complete_config(
                checkpoint_config_payload, source=checkpoint
            )
            if normalized_checkpoint_config != normalized_run_config:
                changed_fields = sorted(
                    field
                    for field in normalized_run_config
                    if normalized_checkpoint_config[field]
                    != normalized_run_config[field]
                )
                raise ValueError(
                    f"checkpoint embedded config differs from frozen run config "
                    f"in {checkpoint}; changed_fields={changed_fields}"
                )

            state_dict = payload.get("online_state_dict")
            if not isinstance(state_dict, Mapping):
                raise ValueError(f"checkpoint has no online state_dict: {checkpoint}")
            model = build_model(
                "hnp",
                observation_dim=checkpoint_config.input_dim,
                action_dim=checkpoint_config.n_actions,
            )
            if not isinstance(model, HNPQNetwork):
                raise TypeError("the registered 'hnp' factory no longer builds HNPQNetwork")
            model.load_state_dict(state_dict, strict=True)
            if PROJECTION_WEIGHT_KEY not in state_dict:
                raise KeyError(
                    f"missing {PROJECTION_WEIGHT_KEY!r} in {checkpoint}"
                )
            if POLYNOMIAL_LAYERNORM_KEY not in state_dict:
                raise KeyError(
                    f"missing {POLYNOMIAL_LAYERNORM_KEY!r} in {checkpoint}"
                )
            projection_weight = state_dict[PROJECTION_WEIGHT_KEY]
            if tuple(projection_weight.shape) != (128, 128):
                raise ValueError(
                    f"unexpected {PROJECTION_WEIGHT_KEY} shape "
                    f"{tuple(projection_weight.shape)} in {checkpoint}"
                )
            records[seed] = {
                "train_seed": seed,
                "jammer_mode": jammer_mode,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": actual_hash,
                "run_directory": str(run_dir),
                "run_config_sha256": actual_config_hash,
                "models_source_path": str(MODELS_SOURCE_PATH),
                "models_source_sha256": models_source_hash,
                "state_dict": state_dict,
            }

    actual_seed_set = set(records)
    missing = sorted(expected_seed_set - actual_seed_set)
    extra = sorted(actual_seed_set - expected_seed_set)
    if missing or extra:
        raise ValueError(
            f"HNP seed coverage mismatch for {jammer_mode}; missing={missing}, extra={extra}"
        )
    return [records[seed] for seed in sorted(records)]


def audit_polynomial_branch_weights(
    checkpoint_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute equal-sized branch summaries independently for every seed."""

    rows: list[dict[str, Any]] = []
    for record in checkpoint_records:
        state_dict = record["state_dict"]
        weight = (
            state_dict[PROJECTION_WEIGHT_KEY]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        output_features, input_features = weight.shape
        if input_features % 2:
            raise ValueError("projection input width cannot be split into [h, h^2]")
        branch_width = input_features // 2
        linear = weight[:, :branch_width]
        squared = weight[:, branch_width:]
        linear_mean_abs = float(linear.abs().mean())
        squared_mean_abs = float(squared.abs().mean())
        rows.append(
            {
                "train_seed": int(record["train_seed"]),
                "jammer_mode": str(record["jammer_mode"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                "run_config_sha256": str(record["run_config_sha256"]),
                "models_source_path": str(record["models_source_path"]),
                "models_source_sha256": str(record["models_source_sha256"]),
                "projection_weight_key": PROJECTION_WEIGHT_KEY,
                "projection_weight_shape": f"{output_features}x{input_features}",
                "branch_order": "columns 0:64 = LayerNorm([h,h^2]) positions for h; columns 64:128 = positions for h^2",
                "branch_width": int(branch_width),
                "connections_per_branch": int(linear.numel()),
                "linear_mean_absolute_outgoing_weight": linear_mean_abs,
                "squared_mean_absolute_outgoing_weight": squared_mean_abs,
                "squared_minus_linear_mean_absolute_weight": (
                    squared_mean_abs - linear_mean_abs
                ),
                "squared_to_linear_mean_absolute_ratio": (
                    squared_mean_abs / linear_mean_abs
                    if linear_mean_abs > 0
                    else None
                ),
                "linear_frobenius_norm": float(torch.linalg.vector_norm(linear)),
                "squared_frobenius_norm": float(torch.linalg.vector_norm(squared)),
                "interpretation_scope": INTERPRETATION_LIMIT,
            }
        )
    return rows


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and non-empty")
    return {
        "n_records": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _metric_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "linear_mean_absolute_outgoing_weight": _summary(
            [float(row["linear_mean_absolute_outgoing_weight"]) for row in rows]
        ),
        "squared_mean_absolute_outgoing_weight": _summary(
            [float(row["squared_mean_absolute_outgoing_weight"]) for row in rows]
        ),
        "paired_squared_minus_linear": _summary(
            [float(row["squared_minus_linear_mean_absolute_weight"]) for row in rows]
        ),
        "linear_frobenius_norm": _summary(
            [float(row["linear_frobenius_norm"]) for row in rows]
        ),
        "squared_frobenius_norm": _summary(
            [float(row["squared_frobenius_norm"]) for row in rows]
        ),
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]], *, jammer_mode: str
) -> dict[str, Any]:
    if not rows:
        raise ValueError("weight summary requires at least one checkpoint record")
    actual_modes = {str(row["jammer_mode"]) for row in rows}
    expected_modes = (
        set(EXPECTED_JAMMER_MODES) if jammer_mode == "both" else {jammer_mode}
    )
    if actual_modes != expected_modes:
        raise ValueError(
            f"jammer-mode coverage mismatch in weight rows; "
            f"expected={sorted(expected_modes)}, actual={sorted(actual_modes)}"
        )
    model_source_hashes = sorted({str(row["models_source_sha256"]) for row in rows})
    model_source_paths = sorted({str(row["models_source_path"]) for row in rows})
    run_config_hashes = sorted({str(row["run_config_sha256"]) for row in rows})
    if len(model_source_hashes) != 1 or len(model_source_paths) != 1:
        raise ValueError("weight rows do not share one verified src/models.py source")
    overall_metrics = _metric_summaries(rows)
    by_mode: dict[str, Any] = {}
    for mode in sorted(actual_modes):
        mode_rows = [row for row in rows if str(row["jammer_mode"]) == mode]
        mode_seeds = sorted({int(row["train_seed"]) for row in mode_rows})
        by_mode[mode] = {
            "unit": "one HNP checkpoint per training seed",
            "n_training_seeds": len(mode_seeds),
            "training_seeds": mode_seeds,
            **_metric_summaries(mode_rows),
        }
    checkpoint_manifest = sorted(
        (
            str(row["jammer_mode"]),
            int(row["train_seed"]),
            str(row["checkpoint_sha256"]),
        )
        for row in rows
    )
    checkpoint_set_sha256 = hashlib.sha256(
        json.dumps(checkpoint_manifest, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "descriptive checkpoint-weight audit only",
        "outcome_access": (
            "No evaluation episode table, seed summary, or OOD policy outcome is read."
        ),
        "model": "HNP-DQN",
        "jammer_mode": jammer_mode,
        "training_seeds": sorted({int(row["train_seed"]) for row in rows}),
        "checkpoint_record_count": len(rows),
        "state_dict_key": PROJECTION_WEIGHT_KEY,
        "verified_state_dict_shape": [128, 128],
        "provenance": {
            "models_source_path": model_source_paths[0],
            "models_source_sha256": model_source_hashes[0],
            "run_config_sha256": run_config_hashes,
            "device_is_only_formal_preset_exemption": True,
            "checkpoint_configs_equal_frozen_run_config": True,
            "checkpoint_set_sha256": checkpoint_set_sha256,
        },
        "branch_definition": {
            "linear_h": "projection.0.weight[:, 0:64]",
            "squared_h2": "projection.0.weight[:, 64:128]",
            "upstream_tensor": "LayerNorm(concat(h, h^2)); ordering is preserved but normalization couples/scales the branches",
        },
        "overall": {
            "unit": "one training-seed x jammer-mode HNP checkpoint",
            "n_seed_mode_records": len(rows),
            **overall_metrics,
        },
        "by_jammer_mode": by_mode,
        "inferential_tests": "none; no p-value or performance association is claimed",
        "interpretation_limit": INTERPRETATION_LIMIT,
    }
    # Keep direct metric keys for simple consumers; they are explicitly the
    # same descriptive overall summaries recorded above.
    summary.update(overall_metrics)
    return summary


def build_plot_source(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    long_rows: list[dict[str, Any]] = []
    for row in rows:
        for branch, mean_key, norm_key in (
            (
                "linear_h",
                "linear_mean_absolute_outgoing_weight",
                "linear_frobenius_norm",
            ),
            (
                "squared_h2",
                "squared_mean_absolute_outgoing_weight",
                "squared_frobenius_norm",
            ),
        ):
            long_rows.append(
                {
                    "train_seed": int(row["train_seed"]),
                    "jammer_mode": str(row["jammer_mode"]),
                    "branch": branch,
                    "mean_absolute_outgoing_weight": float(row[mean_key]),
                    "frobenius_norm": float(row[norm_key]),
                    "projection_weight_key": PROJECTION_WEIGHT_KEY,
                }
            )
    return pd.DataFrame(long_rows)


def write_outputs(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_csv = output_dir / "polynomial_branch_weight_audit_seed_level.csv"
    summary_json = output_dir / "polynomial_branch_weight_audit_summary.json"
    plot_csv = output_dir / "polynomial_branch_weight_plot_source.csv"
    if any(path.exists() for path in (seed_csv, summary_json, plot_csv)):
        raise FileExistsError(
            "refusing to overwrite an existing weight audit; choose a new output directory"
        )
    serializable_rows = [
        {key: value for key, value in row.items() if key != "state_dict"}
        for row in rows
    ]
    pd.DataFrame(serializable_rows).to_csv(seed_csv, index=False)
    summary_json.write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    build_plot_source(serializable_rows).to_csv(plot_csv, index=False)
    return seed_csv, summary_json, plot_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Descriptive checkpoint-only [h,h^2] projection-weight audit. "
            "It does not read evaluation or OOD policy outcomes."
        )
    )
    parser.add_argument(
        "--formal-root",
        action="append",
        required=True,
        help="formal_primary_v3 run directory or parent; repeat for split runs",
    )
    parser.add_argument(
        "--jammer-mode",
        choices=("both", "sweeping", "random"),
        default="both",
        help="default: audit both canonical HNP training jammer modes in one output",
    )
    parser.add_argument(
        "--run-label",
        default="formal_primary_v3",
        help="Required substring in each canonical run-directory name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "polynomial_weight_audit",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    records = canonical_hnp_checkpoint_records(
        args.formal_root,
        jammer_mode=args.jammer_mode,
        expected_seeds=EXPECTED_SEEDS,
        required_run_label=args.run_label,
    )
    rows = audit_polynomial_branch_weights(records)
    summary = build_summary(rows, jammer_mode=args.jammer_mode)
    paths = write_outputs(args.output_dir, rows, summary)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
