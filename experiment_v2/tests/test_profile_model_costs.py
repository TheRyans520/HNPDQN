"""Fast synthetic checks for the deferred isolated cost profiler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import profile_model_costs as profiler


def test_isolated_config_uses_fixed_three_noninferential_cpu_seeds() -> None:
    config = profiler.isolated_training_config()
    assert config.mode == "formal"
    assert config.models == ("hnp", "matched")
    assert config.train_seeds == (91001, 91002, 91003)
    assert config.train_episodes == 200
    assert config.episode_length == 100
    assert config.device == "cpu"
    assert config.jammer_modes == ("sweeping",)
    jobs = profiler.counterbalanced_training_jobs()
    assert len(jobs) == 6
    for seed in profiler.PROFILING_SEEDS:
        assert {model for job_seed, model in jobs if job_seed == seed} == {
            "hnp",
            "matched",
        }


def test_development_loader_never_constructs_pilot_or_ood(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_build_dataset(data_dir, **kwargs):
        captured["data_dir"] = data_dir
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(profiler, "build_dataset", fake_build_dataset)
    config = profiler.isolated_training_config()
    result = profiler.load_development_only_dataset("synthetic_raw", config)
    assert result is sentinel
    assert captured["split_scan_ids"] == {
        "train": tuple(range(0, 7)),
        "val": (7,),
    }
    flattened = {
        scan_id
        for split_ids in captured["split_scan_ids"].values()
        for scan_id in split_ids
    }
    assert flattened == set(range(0, 8))
    assert 8 not in flattened and 9 not in flattened
    assert captured["distance_cm"] == 20
    assert captured["power_dbm"] == 10


def test_fixed_latency_helper_is_small_and_deterministic_shape() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(48, 8), torch.nn.ReLU())
    result = profiler.fixed_batch1_latency_ms(
        model,
        device=torch.device("cpu"),
        warmup=2,
        repetitions=5,
    )
    assert set(result) == {"median_ms", "mean_ms", "p05_ms", "p95_ms"}
    assert all(np.isfinite(value) and value >= 0 for value in result.values())
    assert result["p05_ms"] <= result["median_ms"] <= result["p95_ms"]


def test_pacing_occurs_after_sync_timing_and_is_excluded_from_samples(
    monkeypatch,
) -> None:
    events: list[str] = []

    class RecordingModel(torch.nn.Module):
        def forward(self, value):
            events.append("forward")
            return value

    clock_values = iter((0, 1_000_000, 2_000_000, 3_000_000))

    def fake_clock() -> int:
        events.append("clock")
        return next(clock_values)

    def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    monkeypatch.setattr(profiler.time, "perf_counter_ns", fake_clock)
    monkeypatch.setattr(profiler.time, "sleep", fake_sleep)
    result = profiler.fixed_batch1_latency_ms(
        RecordingModel(),
        device=torch.device("cpu"),
        warmup=1,
        repetitions=2,
        pacing_sleep_seconds=0.005,
    )
    assert result["median_ms"] == pytest.approx(1.0)
    assert events == [
        "forward",
        "sleep:0.005",
        "clock",
        "forward",
        "clock",
        "sleep:0.005",
        "clock",
        "forward",
        "clock",
        "sleep:0.005",
    ]


def test_gpu_pacing_protocol_is_auditable_and_rejects_invalid_values() -> None:
    assert profiler.GPU_LATENCY_PACING_SECONDS == pytest.approx(0.005)
    assert "outside the recorded latency interval" in profiler.GPU_LATENCY_PACING_SCOPE
    assert "below 85%" in profiler.GPU_UTILIZATION_INTENT
    metadata = profiler._metadata()
    assert metadata["gpu_latency_pacing_seconds"] == pytest.approx(0.005)
    assert metadata["gpu_latency_pacing_scope"] == profiler.GPU_LATENCY_PACING_SCOPE
    model = torch.nn.Linear(48, 8)
    with pytest.raises(ValueError, match="finite and non-negative"):
        profiler.fixed_batch1_latency_ms(
            model,
            device=torch.device("cpu"),
            warmup=0,
            repetitions=1,
            pacing_sleep_seconds=-0.001,
        )


def test_manifest_filters_primary_models_and_rejects_protocol_drift(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    formal = profiler.isolated_training_config().with_overrides(
        train_seeds=(1,), jammer_modes=("sweeping",)
    )
    for model in ("hnp", "matched", "no_dueling"):
        torch.save(
            {
                "model_name": model,
                "seed": 1,
                "training_jammer_mode": "sweeping",
                "config": formal.to_dict(),
            },
            checkpoint_dir / f"{model}_sweeping_seed1.pt",
        )
    manifest = profiler.canonical_checkpoint_manifest(
        [tmp_path],
        expected_train_seeds=(1,),
        expected_jammer_modes=("sweeping",),
        require_complete=True,
    )
    assert {(row["model"], row["canonical_train_seed"]) for row in manifest} == {
        ("hnp", 1),
        ("matched", 1),
    }

    drift = formal.with_overrides(gamma=0.0)
    drift_path = checkpoint_dir / "hnp_random_seed1.pt"
    torch.save(
        {
            "model_name": "hnp",
            "seed": 1,
            "training_jammer_mode": "random",
            "config": drift.to_dict(),
        },
        drift_path,
    )
    with pytest.raises(ValueError, match="not canonical formal"):
        profiler.canonical_checkpoint_manifest(
            [drift_path],
            expected_train_seeds=(1,),
            expected_jammer_modes=("random",),
            require_complete=False,
        )


def test_output_labels_storage_and_noninferential_seed_scope(tmp_path: Path) -> None:
    training = [
        {
            "record_type": "isolated_training_wall_time",
            "model": "hnp",
            "profiling_seed": 91001,
            "profiling_seed_role": profiler.PROFILING_SEED_ROLE,
            "training_loop_wall_seconds": 1.0,
        }
    ]
    inference = [
        {
            "record_type": "canonical_checkpoint_inference",
            "model": "hnp",
            "canonical_train_seed": 1,
            "parameter_bytes": 4,
            "registered_buffer_bytes": 2,
            "inference_persistent_tensor_bytes": 6,
            "inference_persistent_tensor_bytes_scope": profiler.MEMORY_SCOPE,
            "serialized_state_bytes": 10,
            "serialized_state_bytes_role": "storage size, not memory",
            "checkpoint_bytes": 20,
            "checkpoint_bytes_role": "storage size, not memory",
        }
    ]
    csv_path, json_path = profiler.write_isolated_outputs(
        tmp_path, training, inference
    )
    assert csv_path.name == "isolated_model_costs.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "not units for policy-performance inference" in (
        payload["metadata"]["profiling_seed_role"]
    )
    assert "allocator overhead" in payload["metadata"]["memory_scope"]
    assert payload["canonical_checkpoint_inference"][0][
        "serialized_state_bytes_role"
    ] == "storage size, not memory"
    with pytest.raises(FileExistsError):
        profiler.write_isolated_outputs(tmp_path, training, inference)
