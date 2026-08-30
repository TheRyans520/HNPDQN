"""Tests for training, leakage guards, deterministic evaluation and inference."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agent import DDQNAgent, RFNormalizer
from src.baselines import (
    ClairvoyantOraclePolicy,
    ScheduleAwareSweepPolicy,
    StayPolicy,
)
from src.config import ABLATION_MODELS, ExperimentConfig, PRIMARY_MODELS
from src.data import CHANNELS, ScanSplit, TrajectoryKey
from src.env import AntiJammingEnv
from src.runner import (
    _primary_comparisons,
    EvaluationCondition,
    build_balanced_scan_schedule,
    evaluate_policy,
    model_persistent_tensor_bytes,
    preview_jammer_sequence,
    train_agent,
    write_predeclared_evaluation_schedule,
)
from src.statistics import (
    exact_sign_flip_pvalue,
    holm_adjust,
    paired_comparison,
)


def _split(
    name: str = "train", *, scan_ids: tuple[int, ...] = (0,), length: int = 24
) -> ScanSplit:
    trajectories = {}
    starts = {}
    sources = {}
    for scan_id in scan_ids:
        for jammer_index, jammer_channel in enumerate(CHANNELS):
            key = TrajectoryKey(scan_id, jammer_channel)
            values = np.zeros((length, 40), dtype=np.float32)
            for channel_index in range(8):
                offset = 5 * channel_index
                values[:, offset] = (
                    channel_index
                    + 0.1 * jammer_index
                    + np.linspace(0.0, 0.5, length)
                    + 0.01 * scan_id
                )
                values[:, offset + 1 : offset + 5] = 1.0 + channel_index
            trajectories[key] = values
            starts[key] = np.arange(length, dtype=np.int64)
            sources[key] = Path(f"synthetic_{name}_{scan_id}_{jammer_channel}.csv")
    return ScanSplit(
        name=name,
        channels=CHANNELS,
        trajectories=MappingProxyType(trajectories),
        window_starts=MappingProxyType(starts),
        source_files=MappingProxyType(sources),
        window_size=3,
        stride=1,
        distance_cm=20,
        power_dbm=10,
    )


def _tiny_config(**overrides) -> ExperimentConfig:
    values = {
        "train_episodes": 2,
        "episode_length": 4,
        "batch_size": 4,
        "learning_starts": 4,
        "replay_capacity": 32,
        "eval_seeds": (1001, 1002),
    }
    values.update(overrides)
    return ExperimentConfig.preset("smoke").with_overrides(**values)


def test_formal_preset_matches_frozen_hyperparameters() -> None:
    config = ExperimentConfig.preset("formal")
    assert config.train_episodes == 200
    assert config.episode_length == 100
    assert config.gamma == pytest.approx(0.95)
    assert config.tau == pytest.approx(0.005)
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.batch_size == 64
    assert config.replay_capacity == 100_000
    assert config.learning_starts == 64
    assert config.gradient_clip_norm == pytest.approx(0.7)
    assert config.train_seeds == tuple(range(1, 11))
    assert config.eval_seeds == tuple(range(1001, 1021))
    assert config.models == PRIMARY_MODELS


def test_ablation_models_are_explicit_and_do_not_change_formal_default() -> None:
    assert ExperimentConfig.preset("formal").models == ("hnp", "matched")
    for model_name in ABLATION_MODELS:
        selected = ExperimentConfig.preset("formal").with_overrides(
            models=(model_name,)
        )
        assert selected.models == (model_name,)


def test_ablation_only_seed_summary_has_no_primary_comparisons() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        {
            "condition": ["within_condition_pilot"] * 3,
            "jammer_mode": ["sweeping"] * 3,
            "method": list(ABLATION_MODELS),
            "train_seed": [1.0, 1.0, 1.0],
            "return": [90.0, 91.0, 92.0],
        }
    )
    assert _primary_comparisons(rows) == []


def test_normalizer_is_train_only_and_preserves_action_block() -> None:
    train = _split("train")
    normalizer = RFNormalizer.fit_training_split(train)
    observation = np.concatenate(
        (train.observations[0], np.eye(8, dtype=np.float32)[3])
    )
    transformed = normalizer.transform(observation)
    np.testing.assert_array_equal(transformed[40:], observation[40:])
    batch = np.stack((observation, observation))
    np.testing.assert_array_equal(normalizer.transform(batch)[:, 40:], batch[:, 40:])
    with pytest.raises(ValueError, match="restricted to train"):
        RFNormalizer.fit_training_split(_split("val"))


def test_episode_epsilon_schedule() -> None:
    config = _tiny_config()
    agent = DDQNAgent("hnp", config, RFNormalizer.fit_training_split(_split()), seed=1)
    agent.set_episode(0)
    assert agent.epsilon == pytest.approx(1.0)
    agent.set_episode(10)
    assert agent.epsilon == pytest.approx(0.5)
    agent.set_episode(199)
    assert agent.epsilon == pytest.approx(1.0 / 20.9)


def test_batch1_action_does_not_update_batchnorm_statistics() -> None:
    train = _split()
    agent = DDQNAgent("hnp", _tiny_config(), RFNormalizer.fit_training_split(train), seed=2)
    agent.online.train()
    batchnorm = next(
        module for module in agent.online.modules() if isinstance(module, torch.nn.BatchNorm1d)
    )
    before_mean = batchnorm.running_mean.detach().clone()
    before_batches = batchnorm.num_batches_tracked.detach().clone()
    action = agent.act(np.concatenate((train.observations[0], np.zeros(8))))
    assert action in range(8)
    assert agent.online.training  # previous mode is restored
    torch.testing.assert_close(batchnorm.running_mean, before_mean)
    torch.testing.assert_close(batchnorm.num_batches_tracked, before_batches)


def test_soft_update_includes_batchnorm_buffers() -> None:
    train = _split()
    config = _tiny_config(tau=0.5)
    agent = DDQNAgent("hnp", config, RFNormalizer.fit_training_split(train), seed=3)
    online_bn = next(
        module for module in agent.online.modules() if isinstance(module, torch.nn.BatchNorm1d)
    )
    target_bn = next(
        module for module in agent.target.modules() if isinstance(module, torch.nn.BatchNorm1d)
    )
    online_bn.running_mean.fill_(4.0)
    target_bn.running_mean.zero_()
    online_bn.num_batches_tracked.fill_(7)
    target_bn.num_batches_tracked.zero_()
    agent._soft_update()
    torch.testing.assert_close(target_bn.running_mean, torch.full_like(target_bn.running_mean, 2.0))
    assert int(target_bn.num_batches_tracked) == 7


def test_ddqn_next_action_selection_does_not_double_update_batchnorm() -> None:
    train = _split()
    config = _tiny_config(gradient_steps=1)
    agent = DDQNAgent("hnp", config, RFNormalizer.fit_training_split(train), seed=31)
    observation = np.concatenate((train.observations[0], np.zeros(8, dtype=np.float32)))
    for index in range(config.batch_size):
        agent.store_transition(
            observation,
            index % config.n_actions,
            1.0,
            observation,
            False,
        )
    batchnorms = [
        module
        for module in agent.online.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    before = [int(module.num_batches_tracked) for module in batchnorms]
    assert agent.learn() is not None
    after = [int(module.num_batches_tracked) for module in batchnorms]
    assert after == [value + 1 for value in before]


def test_inference_persistent_bytes_are_parameters_plus_registered_buffers() -> None:
    train = _split()
    agent = DDQNAgent("hnp", _tiny_config(), RFNormalizer.fit_training_split(train), seed=32)
    sizes = model_persistent_tensor_bytes(agent.online)
    expected_parameters = sum(
        tensor.numel() * tensor.element_size()
        for tensor in agent.online.parameters()
    )
    expected_buffers = sum(
        tensor.numel() * tensor.element_size() for tensor in agent.online.buffers()
    )
    assert sizes["parameter_bytes"] == expected_parameters
    assert sizes["registered_buffer_bytes"] == expected_buffers
    assert sizes["inference_persistent_tensor_bytes"] == (
        expected_parameters + expected_buffers
    )


def test_tiny_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    train = _split(length=24)
    config = _tiny_config()
    normalizer = RFNormalizer.fit_training_split(train)
    agent = DDQNAgent("hnp", config, normalizer, seed=4)
    rows = train_agent(
        agent,
        train,
        jammer_mode="sweeping",
        config=config,
        train_seed=4,
    )
    assert len(rows) == 2
    assert agent.update_steps > 0
    # The episode horizon is a truncation, not an absorbing terminal state;
    # every collected target therefore retains its bootstrap term.
    assert not agent.replay.dones[: len(agent.replay)].any()
    checkpoint = agent.save(tmp_path / "agent.pt")
    loaded = DDQNAgent.load(checkpoint, device="cpu")
    assert loaded.config.gamma == config.gamma
    assert loaded.normalizer.fit_split == "train"
    np.testing.assert_allclose(loaded.normalizer.mean, normalizer.mean)


def test_fixed_evaluation_seeds_reproduce_same_trajectories() -> None:
    split = _split("val", scan_ids=(7,), length=24)
    config = _tiny_config()
    first = evaluate_policy(
        StayPolicy(initial_action=0),
        split,
        method="stay",
        condition="validation",
        role="development_only",
        jammer_mode="random",
        eval_seeds=config.eval_seeds,
        config=config,
        train_seed=None,
    )
    second = evaluate_policy(
        StayPolicy(initial_action=0),
        split,
        method="stay",
        condition="validation",
        role="development_only",
        jammer_mode="random",
        eval_seeds=config.eval_seeds,
        config=config,
        train_seed=None,
    )
    assert first == second
    assert [row["eval_seed"] for row in first] == [1001, 1002]


def test_twenty_trajectories_cover_each_of_ten_scans_exactly_twice() -> None:
    split = _split("ood", scan_ids=tuple(range(10)), length=24)
    schedule = build_balanced_scan_schedule(
        split, tuple(range(1001, 1021)), episodes_per_seed=1
    )
    assert len(schedule) == 20
    counts = {
        scan_id: sum(item.scan_id == scan_id for item in schedule)
        for scan_id in split.scan_ids
    }
    assert counts == {scan_id: 2 for scan_id in range(10)}
    assert {
        item.scan_replicate for item in schedule if item.scan_id == 0
    } == {0, 1}


def test_pilot_schedule_is_balanced_to_twenty_total_trajectories() -> None:
    split = _split("test", scan_ids=(8, 9), length=24)
    schedule = build_balanced_scan_schedule(
        split, tuple(range(1001, 1021)), episodes_per_seed=1
    )
    assert len(schedule) == 20
    assert sum(item.scan_id == 8 for item in schedule) == 10
    assert sum(item.scan_id == 9 for item in schedule) == 10


def test_predeclared_schedule_artifact_is_written_before_evaluation(tmp_path: Path) -> None:
    split = _split("ood", scan_ids=tuple(range(10)), length=24)
    config = _tiny_config(eval_seeds=tuple(range(1001, 1021)))
    schedules, rows, path = write_predeclared_evaluation_schedule(
        [EvaluationCondition("synthetic_transfer", split, "cross_configuration")],
        jammer_modes=("sweeping", "random"),
        config=config,
        output_dir=tmp_path,
    )
    assert path.name == "PREDECLARED_EVALUATION_SCHEDULE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["entries"]) == 40
    assert len(rows) == 40
    assert len(schedules["synthetic_transfer"]) == 20
    required = {
        "condition",
        "jammer_mode",
        "eval_index",
        "eval_seed",
        "scan_id",
        "replicate",
        "trajectory_seed",
        "start_index",
        "forced_reset_options",
    }
    assert required <= set(payload["entries"][0])
    for jammer_mode in ("sweeping", "random"):
        subset = [row for row in rows if row["jammer_mode"] == jammer_mode]
        assert {
            scan_id: sum(row["scan_id"] == scan_id for row in subset)
            for scan_id in range(10)
        } == {scan_id: 2 for scan_id in range(10)}


def test_clairvoyant_preplay_reuses_identical_forced_trajectories() -> None:
    split = _split("ood", scan_ids=tuple(range(10)), length=24)
    config = _tiny_config(eval_seeds=(1001, 1002, 1003, 1004))
    schedule = build_balanced_scan_schedule(split, config.eval_seeds)
    stay_rows = evaluate_policy(
        StayPolicy(initial_action=0),
        split,
        method="stay",
        condition="synthetic_ood",
        role="cross_configuration",
        jammer_mode="random",
        eval_seeds=config.eval_seeds,
        config=config,
        train_seed=None,
        trajectory_schedule=schedule,
    )
    oracle_rows = evaluate_policy(
        ClairvoyantOraclePolicy(CHANNELS),
        split,
        method="clairvoyant_oracle",
        condition="synthetic_ood",
        role="cross_configuration",
        jammer_mode="random",
        eval_seeds=config.eval_seeds,
        config=config,
        train_seed=None,
        trajectory_schedule=schedule,
    )
    for stay, oracle in zip(stay_rows, oracle_rows):
        for field in ("scan_id", "scan_replicate", "start_index", "trajectory_seed"):
            assert stay[field] == oracle[field]
        assert oracle["return"] >= stay["return"]


def test_schedule_policy_is_rejected_for_random_jammer() -> None:
    split = _split("val", scan_ids=(7,), length=24)
    config = _tiny_config()
    with pytest.raises(ValueError, match="not defined"):
        evaluate_policy(
            ScheduleAwareSweepPolicy(),
            split,
            method="schedule_sweep",
            condition="validation",
            role="development_only",
            jammer_mode="random",
            eval_seeds=config.eval_seeds,
            config=config,
            train_seed=None,
        )


def test_random_jammer_preview_exactly_matches_actual_sequence() -> None:
    split = _split("ood", scan_ids=(0,), length=24)
    config = _tiny_config(episode_length=8)
    trajectory_seed = 735711
    preview = preview_jammer_sequence(
        split,
        jammer_mode="random",
        trajectory_seed=trajectory_seed,
        scan_id=0,
        config=config,
    )
    env = AntiJammingEnv(
        split,
        jammer_mode="random",
        episode_length=config.episode_length,
        switch_cost=config.switch_cost,
        seed=trajectory_seed,
    )
    _, info = env.reset(seed=trajectory_seed, options={"scan_id": 0})
    actual = [info["jammer_channel"]]
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(7)
        done = terminated or truncated
        if not done:
            actual.append(info["jammer_channel"])
    assert tuple(actual) == preview


def test_primary_heuristic_is_predeclared_by_jammer_mode() -> None:
    import pandas as pd

    rows = []
    for jammer_mode in ("sweeping", "random"):
        for seed, value in ((1.0, 90.0), (2.0, 91.0)):
            rows.append(
                {
                    "condition": "transfer",
                    "jammer_mode": jammer_mode,
                    "method": "hnp",
                    "train_seed": seed,
                    "return": value,
                }
            )
            rows.append(
                {
                    "condition": "transfer",
                    "jammer_mode": jammer_mode,
                    "method": "matched",
                    "train_seed": seed,
                    "return": value - 1.0,
                }
            )
        for method, value in (
            ("schedule_sweep", 98.6),
            ("threshold", 95.0),
            ("jammer_greedy", 97.0),
            ("clairvoyant_oracle", 99.0),
        ):
            rows.append(
                {
                    "condition": "transfer",
                    "jammer_mode": jammer_mode,
                    "method": method,
                    "train_seed": np.nan,
                    "return": value,
                }
            )
    comparisons = _primary_comparisons(pd.DataFrame(rows))
    pairs = {
        (row["jammer_mode"], row["method_b"]) for row in comparisons
    }
    assert pairs == {
        ("sweeping", "matched"),
        ("sweeping", "schedule_sweep"),
        ("random", "matched"),
        ("random", "threshold"),
    }


def test_exact_sign_flip_and_holm_are_seed_level() -> None:
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == pytest.approx(0.25)
    comparison = paired_comparison(
        [3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0],
        method_a="hnp",
        method_b="matched",
        metric="return",
    )
    assert comparison.n_pairs == 3
    assert comparison.mean_difference == pytest.approx(1.0)
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
