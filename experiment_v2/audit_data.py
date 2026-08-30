"""Generate the frozen data-integrity and feature-direction audit.

The script is intentionally independent of model training.  It records the
file-level split, window counts, SHA-256 hashes, finite-value checks, and the
empirical direction of the dataset field named ``snr``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data import CHANNELS, STATE_FEATURES, build_dataset, build_ood_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_summary(split) -> dict[str, object]:
    arrays = [trajectory for _, trajectory, _, _ in split.iter_trajectories()]
    values = split.observations
    return {
        "label": split.name,
        "distance_cm": split.distance_cm,
        "power_dbm": split.power_dbm,
        "scan_ids": list(split.scan_ids),
        "source_file_count": len(arrays),
        "trajectory_count": len(arrays),
        "window_count": int(sum(len(array) for array in arrays)),
        "observation_dimension": int(split.observation_size),
        "all_features_finite": bool(np.isfinite(values).all()),
        "minimum_trajectory_windows": int(min(map(len, arrays))),
        "maximum_trajectory_windows": int(max(map(len, arrays))),
    }


def snr_direction(split) -> dict[str, float | str]:
    jammed: list[np.ndarray] = []
    unjammed: list[np.ndarray] = []
    argmin_hits = 0
    argmax_hits = 0
    total = 0
    for key, trajectory, _, _ in split.iter_trajectories():
        snr = trajectory.reshape(len(trajectory), len(CHANNELS), len(STATE_FEATURES))[
            :, :, 0
        ]
        jammer_index = CHANNELS.index(key.jammer_channel)
        jammed.append(snr[:, jammer_index])
        unjammed.append(np.delete(snr, jammer_index, axis=1).reshape(-1))
        argmin_hits += int(np.sum(np.argmin(snr, axis=1) == jammer_index))
        argmax_hits += int(np.sum(np.argmax(snr, axis=1) == jammer_index))
        total += len(snr)
    jammed_values = np.concatenate(jammed)
    unjammed_values = np.concatenate(unjammed)
    return {
        "dataset_field": "snr_mean",
        "jammed_channel_mean": float(jammed_values.mean()),
        "unjammed_channel_mean": float(unjammed_values.mean()),
        "argmin_selects_jammer_rate": float(argmin_hits / total),
        "argmax_selects_jammer_rate": float(argmax_hits / total),
        "declared_quality_direction": "quality = -snr_mean (higher quality is better)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development = build_dataset(raw_dir, distance_cm=20, power_dbm=10)
    distance_shift = build_ood_dataset(raw_dir, distance_cm=40, power_dbm=10)
    power_shift = build_ood_dataset(raw_dir, distance_cm=20, power_dbm=5)

    split_sets = {
        "development_train": development.train,
        "development_validation": development.val,
        "within_condition_pilot": development.test,
        "distance_shift_evaluation": distance_shift["ood"],
        "power_shift_evaluation": power_shift["ood"],
    }

    file_owners: dict[str, str] = {}
    scan_owners: dict[str, list[int]] = {}
    manifest_rows: list[dict[str, object]] = []
    for label, split in split_sets.items():
        scan_owners[label] = list(split.scan_ids)
        for key, trajectory, starts, source in split.iter_trajectories():
            resolved = str(source.resolve())
            if resolved in file_owners:
                raise RuntimeError(
                    f"source file reused in {file_owners[resolved]} and {label}: {resolved}"
                )
            file_owners[resolved] = label
            manifest_rows.append(
                {
                    "set": label,
                    "distance_cm": split.distance_cm,
                    "power_dbm": split.power_dbm,
                    "scan_id": key.scan_id,
                    "jammer_channel_mhz": key.jammer_channel,
                    "windows": len(trajectory),
                    "first_window_start": int(starts[0]),
                    "last_window_start": int(starts[-1]),
                    "source_file": source.name,
                    "sha256": sha256(source),
                }
            )

    with (output_dir / "source_file_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    audit = {
        "protocol": {
            "window_size": development.window_size,
            "stride": development.stride,
            "rf_observation_dimension": development.train.observation_size,
            "previous_action_dimension": len(CHANNELS),
            "agent_state_dimension": development.train.observation_size + len(CHANNELS),
            "source_files_are_unique_across_declared_sets": True,
            "scan_owners": scan_owners,
        },
        "sets": {label: split_summary(split) for label, split in split_sets.items()},
        "snr_direction_train_only": snr_direction(development.train),
        "manifest": {
            "row_count": len(manifest_rows),
            "sha256": sha256(output_dir / "source_file_manifest.csv"),
        },
    }
    with (output_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
