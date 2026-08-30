"""Synthetic-checkpoint tests for the outcome-blind branch-weight audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig
from src.models import build_model


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "analysis"
    / "audit_polynomial_branch_weights.py"
)
SPEC = importlib.util.spec_from_file_location("audit_polynomial_branch_weights", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture()
def synthetic_formal_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "formal_primary_v3_synthetic"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    config = ExperimentConfig.preset("formal")
    config_payload = config.to_dict()
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_model = build_model("hnp", observation_dim=48, action_dim=8)
    checkpoint_entries = []
    for jammer_mode, scale in (("sweeping", 1.0), ("random", 3.0)):
        for seed in range(1, 11):
            state_dict = {
                key: tensor.detach().clone()
                for key, tensor in base_model.state_dict().items()
            }
            projection = state_dict[audit.PROJECTION_WEIGHT_KEY]
            projection[:, :64].fill_(scale * seed / 100.0)
            projection[:, 64:].fill_(scale * seed / 50.0)
            checkpoint = checkpoint_dir / f"hnp_{jammer_mode}_seed{seed}.pt"
            torch.save(
                {
                    "model_name": "hnp",
                    "seed": seed,
                    "training_jammer_mode": jammer_mode,
                    "config": config_payload,
                    "online_state_dict": state_dict,
                },
                checkpoint,
            )
            checkpoint_entries.append(
                {
                    "model": "hnp",
                    "jammer_mode": jammer_mode,
                    "train_seed": seed,
                    "path": str(checkpoint.relative_to(run_dir)),
                    "sha256": audit._sha256(checkpoint),
                }
            )
    (run_dir / "FROZEN_BEFORE_EVALUATION.json").write_text(
        json.dumps(
            {
                "config_sha256": audit._config_hash(config_payload),
                "core_code_sha256": {
                    audit.MODELS_FREEZE_KEY: audit._sha256(audit.MODELS_SOURCE_PATH)
                },
                "checkpoints": checkpoint_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # These sentinel outcome files must never be opened by the audit.
    (run_dir / "evaluation_episodes.csv").write_text("FORBIDDEN", encoding="utf-8")
    (run_dir / "seed_summary.csv").write_text("FORBIDDEN", encoding="utf-8")
    return run_dir


def test_ten_seed_audit_uses_verified_projection_key_without_outcomes(
    synthetic_formal_run: Path, monkeypatch
) -> None:
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name in {"evaluation_episodes.csv", "seed_summary.csv"}:
            raise AssertionError(f"outcome file was opened: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    records = audit.canonical_hnp_checkpoint_records(
        [synthetic_formal_run], jammer_mode="sweeping"
    )
    assert len(records) == 10
    rows = audit.audit_polynomial_branch_weights(records)
    assert [row["train_seed"] for row in rows] == list(range(1, 11))
    assert {row["projection_weight_key"] for row in rows} == {
        "projection.0.weight"
    }
    assert {row["projection_weight_shape"] for row in rows} == {"128x128"}
    for row in rows:
        seed = row["train_seed"]
        assert row["linear_mean_absolute_outgoing_weight"] == pytest.approx(
            seed / 100.0
        )
        assert row["squared_mean_absolute_outgoing_weight"] == pytest.approx(
            seed / 50.0
        )
        assert row["squared_to_linear_mean_absolute_ratio"] == pytest.approx(2.0)


def test_summary_and_plot_source_are_descriptive_and_reproducible(
    synthetic_formal_run: Path, tmp_path: Path
) -> None:
    records = audit.canonical_hnp_checkpoint_records(
        [synthetic_formal_run], jammer_mode="sweeping"
    )
    rows = audit.audit_polynomial_branch_weights(records)
    summary = audit.build_summary(rows, jammer_mode="sweeping")
    assert summary["state_dict_key"] == "projection.0.weight"
    assert summary["linear_mean_absolute_outgoing_weight"]["n_records"] == 10
    assert summary["provenance"]["models_source_sha256"] == audit._sha256(
        audit.MODELS_SOURCE_PATH
    )
    assert summary["provenance"]["checkpoint_configs_equal_frozen_run_config"]
    assert summary["inferential_tests"].startswith("none")
    assert "not causal" in summary["interpretation_limit"]

    output_dir = tmp_path / "audit_output"
    seed_csv, summary_json, plot_csv = audit.write_outputs(
        output_dir, rows, summary
    )
    assert len(pd.read_csv(seed_csv)) == 10
    plot = pd.read_csv(plot_csv)
    assert len(plot) == 20
    assert set(plot["branch"]) == {"linear_h", "squared_h2"}
    persisted = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "No evaluation episode table" in persisted["outcome_access"]
    with pytest.raises(FileExistsError):
        audit.write_outputs(output_dir, rows, summary)


def test_both_modes_produce_twenty_seed_mode_rows_and_stratified_summary(
    synthetic_formal_run: Path, tmp_path: Path
) -> None:
    records = audit.canonical_hnp_checkpoint_records(
        [synthetic_formal_run], jammer_mode="both"
    )
    assert len(records) == 20
    assert {
        mode: sum(record["jammer_mode"] == mode for record in records)
        for mode in audit.EXPECTED_JAMMER_MODES
    } == {"sweeping": 10, "random": 10}

    rows = audit.audit_polynomial_branch_weights(records)
    summary = audit.build_summary(rows, jammer_mode="both")
    assert summary["checkpoint_record_count"] == 20
    assert summary["overall"]["n_seed_mode_records"] == 20
    assert summary["overall"]["linear_mean_absolute_outgoing_weight"][
        "n_records"
    ] == 20
    assert set(summary["by_jammer_mode"]) == {"sweeping", "random"}
    assert all(
        value["n_training_seeds"] == 10
        for value in summary["by_jammer_mode"].values()
    )
    assert summary["inferential_tests"].startswith("none")
    assert len(audit.build_plot_source(rows)) == 40

    seed_csv, summary_json, plot_csv = audit.write_outputs(
        tmp_path / "both_output", rows, summary
    )
    assert len(pd.read_csv(seed_csv)) == 20
    assert json.loads(summary_json.read_text(encoding="utf-8"))[
        "jammer_mode"
    ] == "both"
    assert len(pd.read_csv(plot_csv)) == 40


def test_checkpoint_hash_tampering_is_rejected(synthetic_formal_run: Path) -> None:
    checkpoint = synthetic_formal_run / "checkpoints" / "hnp_sweeping_seed1.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        audit.canonical_hnp_checkpoint_records(
            [synthetic_formal_run], jammer_mode="sweeping"
        )


def test_noncanonical_run_config_is_rejected_even_with_updated_freeze_hash(
    synthetic_formal_run: Path,
) -> None:
    config_path = synthetic_formal_run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["eval_seeds"] = [1001]
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    freeze_path = synthetic_formal_run / "FROZEN_BEFORE_EVALUATION.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["config_sha256"] = audit._config_hash(config)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="noncanonical formal-primary run config"):
        audit.canonical_hnp_checkpoint_records(
            [synthetic_formal_run], jammer_mode="sweeping"
        )


def test_checkpoint_config_must_equal_frozen_run_config_field_by_field(
    synthetic_formal_run: Path,
) -> None:
    checkpoint = synthetic_formal_run / "checkpoints" / "hnp_sweeping_seed1.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config"] = dict(payload["config"])
    payload["config"]["device"] = "cpu"
    torch.save(payload, checkpoint)
    freeze_path = synthetic_formal_run / "FROZEN_BEFORE_EVALUATION.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for entry in freeze["checkpoints"]:
        if entry["train_seed"] == 1:
            entry["sha256"] = audit._sha256(checkpoint)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded config differs"):
        audit.canonical_hnp_checkpoint_records(
            [synthetic_formal_run], jammer_mode="sweeping"
        )


def test_frozen_models_source_hash_must_match_current_imported_source(
    synthetic_formal_run: Path,
) -> None:
    freeze_path = synthetic_formal_run / "FROZEN_BEFORE_EVALUATION.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["core_code_sha256"][audit.MODELS_FREEZE_KEY] = "0" * 64
    freeze_path.write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="models.py source hash mismatch"):
        audit.canonical_hnp_checkpoint_records(
            [synthetic_formal_run], jammer_mode="sweeping"
        )
