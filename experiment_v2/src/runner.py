"""End-to-end training, frozen evaluation, cost reporting and export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from .agent import DDQNAgent, RFNormalizer, require_torch, seed_everything
from .baselines import build_baselines
from .config import ExperimentConfig, RunPaths
from .data import DatasetBundle, ScanSplit, build_dataset, build_ood_dataset
from .env import AntiJammingEnv
from .statistics import apply_holm, paired_comparison


class ObservablePolicy(Protocol):
    def reset(self, *, seed: int | None = None) -> None: ...

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any] | None = None
    ) -> int: ...


@dataclass(frozen=True)
class EvaluationCondition:
    name: str
    split: ScanSplit
    role: str


@dataclass(frozen=True)
class EvaluationTrajectory:
    """One predeclared scan-stratified evaluation slot."""

    eval_seed: int
    scan_id: int
    scan_replicate: int


def build_balanced_scan_schedule(
    split: ScanSplit,
    eval_seeds: tuple[int, ...],
    *,
    episodes_per_seed: int = 1,
) -> tuple[EvaluationTrajectory, ...]:
    """Round-robin fixed seeds across scans with per-scan replicate indices.

    With the formal 20 seeds and a ten-scan transfer split, every scan appears
    exactly twice.  The two-scan transparent pilot remains a 20-trajectory
    balanced diagnostic, so each pilot scan appears ten times.
    """

    if not eval_seeds:
        raise ValueError("eval_seeds must not be empty")
    if int(episodes_per_seed) <= 0:
        raise ValueError("episodes_per_seed must be positive")
    scan_ids = tuple(sorted(int(scan_id) for scan_id in split.scan_ids))
    if not scan_ids:
        raise ValueError("evaluation split has no scans")
    replicate_counts = {scan_id: 0 for scan_id in scan_ids}
    schedule: list[EvaluationTrajectory] = []
    slot_index = 0
    for eval_seed in eval_seeds:
        for _ in range(int(episodes_per_seed)):
            scan_id = scan_ids[slot_index % len(scan_ids)]
            schedule.append(
                EvaluationTrajectory(
                    eval_seed=int(eval_seed),
                    scan_id=scan_id,
                    scan_replicate=replicate_counts[scan_id],
                )
            )
            replicate_counts[scan_id] += 1
            slot_index += 1
    return tuple(schedule)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_seed(seed: int, *labels: str | int) -> int:
    payload = "|".join([str(int(seed)), *(str(label) for label in labels)])
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")
    return value & 0x7FFFFFFF


def _software_manifest() -> dict[str, Any]:
    import scipy

    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        payload["torch"] = None
    return payload


def train_agent(
    agent: DDQNAgent,
    train_split: ScanSplit,
    *,
    jammer_mode: str,
    config: ExperimentConfig,
    train_seed: int,
) -> list[dict[str, Any]]:
    """Train for the fixed episode budget; no evaluation split is consulted."""

    environment_seed = _derived_seed(train_seed, "train_env", jammer_mode)
    env = AntiJammingEnv(
        train_split,
        jammer_mode=jammer_mode,
        episode_length=config.episode_length,
        switch_cost=config.switch_cost,
        reward_safe=config.reward_safe,
        reward_collision=config.reward_collision,
        seed=environment_seed,
        random_start=True,
    )
    rows: list[dict[str, Any]] = []
    for episode in range(config.train_episodes):
        agent.set_episode(episode)
        epsilon_at_start = agent.epsilon
        episode_seed = _derived_seed(environment_seed, "episode", episode)
        observation, _ = env.reset(seed=episode_seed)
        episode_return = 0.0
        losses: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(observation, explore=True)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            # Stop collection on either flag, but mask bootstrapping only for a
            # genuine absorbing terminal state.  Time/data limits are truncations.
            bootstrap_terminal = bool(terminated)
            agent.store_transition(
                observation,
                action,
                reward,
                next_observation,
                bootstrap_terminal,
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            observation = next_observation
            episode_return += reward
        rows.append(
            {
                "model": agent.model_name,
                "jammer_mode": jammer_mode,
                "train_seed": int(train_seed),
                "episode": int(episode),
                "epsilon": float(epsilon_at_start),
                "return": float(episode_return),
                "collisions": int(env.collision_count),
                "collision_rate": float(env.collision_count / env.step_count),
                "switches": int(env.switch_count),
                "switch_rate": float(env.switch_count / env.step_count),
                "steps": int(env.step_count),
                "mean_td_mse": float(np.mean(losses)) if losses else np.nan,
                "updates": int(agent.update_steps),
            }
        )
    return rows


def _policy_action(
    policy: DDQNAgent | ObservablePolicy,
    observation: np.ndarray,
    info: Mapping[str, Any],
) -> int:
    if isinstance(policy, DDQNAgent):
        return policy.act(observation, explore=False)
    return int(policy.act(observation, info))


def preview_jammer_sequence(
    split: ScanSplit,
    *,
    jammer_mode: str,
    trajectory_seed: int,
    scan_id: int,
    config: ExperimentConfig,
) -> tuple[int, ...]:
    """Replay the action-independent jammer process without scoring a policy."""

    preview_env = AntiJammingEnv(
        split,
        jammer_mode=jammer_mode,
        episode_length=config.episode_length,
        switch_cost=config.switch_cost,
        reward_safe=config.reward_safe,
        reward_collision=config.reward_collision,
        seed=trajectory_seed,
        random_start=True,
    )
    _, preview_info = preview_env.reset(
        seed=trajectory_seed, options={"scan_id": int(scan_id)}
    )
    jammer_sequence = [int(preview_info["jammer_channel"])]
    preview_done = False
    while not preview_done:
        _, _, preview_terminated, preview_truncated, preview_info = preview_env.step(0)
        preview_done = bool(preview_terminated or preview_truncated)
        if not preview_done:
            jammer_sequence.append(int(preview_info["jammer_channel"]))
    return tuple(jammer_sequence)


def evaluate_policy(
    policy: DDQNAgent | ObservablePolicy,
    split: ScanSplit,
    *,
    method: str,
    condition: str,
    role: str,
    jammer_mode: str,
    eval_seeds: tuple[int, ...],
    config: ExperimentConfig,
    train_seed: int | None,
    trajectory_schedule: tuple[EvaluationTrajectory, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one frozen policy on exactly the declared seed trajectories."""

    supported_modes = getattr(policy, "supported_jammer_modes", None)
    if supported_modes is not None and jammer_mode not in supported_modes:
        raise ValueError(
            f"policy {method!r} is not defined for jammer_mode={jammer_mode!r}"
        )
    if isinstance(policy, DDQNAgent):
        policy.online.eval()
    rows: list[dict[str, Any]] = []
    if trajectory_schedule is None:
        trajectory_schedule = build_balanced_scan_schedule(
            split,
            eval_seeds,
            episodes_per_seed=config.evaluation_episodes_per_seed,
        )
    for trajectory in trajectory_schedule:
            eval_seed = trajectory.eval_seed
            trajectory_seed = _derived_seed(
                eval_seed,
                condition,
                jammer_mode,
                "trajectory",
                trajectory.scan_id,
                trajectory.scan_replicate,
            )
            policy_seed = _derived_seed(
                eval_seed,
                method,
                jammer_mode,
                "policy",
                trajectory.scan_id,
                trajectory.scan_replicate,
            )
            reset_options = {"scan_id": trajectory.scan_id}

            # Preplay only the action-independent jammer process, using the
            # identical forced scan/start seed.  This provides a complete
            # sequence solely to an explicitly clairvoyant reference.
            jammer_sequence = preview_jammer_sequence(
                split,
                jammer_mode=jammer_mode,
                trajectory_seed=trajectory_seed,
                scan_id=trajectory.scan_id,
                config=config,
            )

            prepare_episode = getattr(policy, "prepare_episode", None)
            if callable(prepare_episode):
                prepare_episode(jammer_sequence)
            env = AntiJammingEnv(
                split,
                jammer_mode=jammer_mode,
                episode_length=config.episode_length,
                switch_cost=config.switch_cost,
                reward_safe=config.reward_safe,
                reward_collision=config.reward_collision,
                seed=trajectory_seed,
                random_start=True,
            )
            if not isinstance(policy, DDQNAgent):
                policy.reset(seed=policy_seed)
            observation, info = env.reset(
                seed=trajectory_seed, options=reset_options
            )
            start_index = int(info["window_index"])
            raw_window_start = int(info["window_start"])
            total_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = _policy_action(policy, observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += reward
            rows.append(
                {
                    "condition": condition,
                    "condition_role": role,
                    "distance_cm": int(split.distance_cm),
                    "power_dbm": int(split.power_dbm),
                    "jammer_mode": jammer_mode,
                    "method": method,
                    "train_seed": train_seed,
                    "eval_seed": int(eval_seed),
                    "scan_id": int(trajectory.scan_id),
                    "scan_replicate": int(trajectory.scan_replicate),
                    "start_index": start_index,
                    "window_start": raw_window_start,
                    "trajectory_seed": int(trajectory_seed),
                    "return": float(total_return),
                    "collisions": int(env.collision_count),
                    "collision_rate": float(env.collision_count / env.step_count),
                    "switches": int(env.switch_count),
                    "switch_rate": float(env.switch_count / env.step_count),
                    "steps": int(env.step_count),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    return rows


def _serialised_state_size(model: Any) -> int:
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def model_persistent_tensor_bytes(model: Any) -> dict[str, int]:
    """Exact resident bytes for parameters and registered model buffers.

    This intentionally excludes activations, framework/runtime objects, CUDA
    allocator reservations, optimizer state, and temporary workspaces.
    """

    parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )
    buffer_bytes = int(
        sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    )
    return {
        "parameter_bytes": parameter_bytes,
        "registered_buffer_bytes": buffer_bytes,
        "inference_persistent_tensor_bytes": parameter_bytes + buffer_bytes,
    }


def _latency_ms(model: Any, device: Any, input_dim: int, *, smoke: bool) -> float:
    import torch

    model.eval()
    sample = torch.zeros((1, input_dim), dtype=torch.float32, device=device)
    warmup = 5 if smoke else 50
    repeats = 20 if smoke else 200
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(timings))


def benchmark_agent(
    agent: DDQNAgent,
    *,
    checkpoint: Path,
    training_seconds: float,
) -> dict[str, Any]:
    """Report trainable capacity, storage and batch-1 CPU/GPU latency."""

    require_torch()
    import copy
    import torch

    try:
        from .models import count_trainable_parameters

        parameters = int(count_trainable_parameters(agent.online))
    except (ImportError, TypeError, AttributeError):
        parameters = int(
            sum(parameter.numel() for parameter in agent.online.parameters() if parameter.requires_grad)
        )

    cpu_model = copy.deepcopy(agent.online).to("cpu").eval()
    cpu_latency = _latency_ms(
        cpu_model,
        torch.device("cpu"),
        agent.config.input_dim,
        smoke=agent.config.mode == "smoke",
    )
    gpu_latency: float | None = None
    if torch.cuda.is_available():
        gpu_model = copy.deepcopy(agent.online).to("cuda").eval()
        gpu_latency = _latency_ms(
            gpu_model,
            torch.device("cuda"),
            agent.config.input_dim,
            smoke=agent.config.mode == "smoke",
        )
        del gpu_model
    tensor_bytes = model_persistent_tensor_bytes(cpu_model)
    return {
        "model": agent.model_name,
        "train_seed": agent.seed,
        "trainable_parameters": parameters,
        **tensor_bytes,
        "inference_tensor_bytes_scope": (
            "parameters + registered buffers only; excludes activations, "
            "framework/runtime objects, and allocator overhead"
        ),
        "serialized_state_bytes": _serialised_state_size(cpu_model),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        "training_seconds": float(training_seconds),
        "batch1_cpu_latency_ms_median": cpu_latency,
        "batch1_gpu_latency_ms_median": gpu_latency,
        "latency_repeats": 20 if agent.config.mode == "smoke" else 200,
        "device_used_for_training": str(agent.device),
    }


def _summarise_seeds(evaluation: pd.DataFrame) -> pd.DataFrame:
    metrics = ["return", "collisions", "collision_rate", "switches", "switch_rate"]
    learned = evaluation[evaluation["train_seed"].notna()].copy()
    baseline = evaluation[evaluation["train_seed"].isna()].copy()
    learned_summary = (
        learned.groupby(
            ["condition", "condition_role", "distance_cm", "power_dbm", "jammer_mode", "method", "train_seed"],
            as_index=False,
            dropna=False,
        )[metrics]
        .mean()
    )
    baseline_summary = (
        baseline.groupby(
            ["condition", "condition_role", "distance_cm", "power_dbm", "jammer_mode", "method"],
            as_index=False,
            dropna=False,
        )[metrics]
        .mean()
    )
    baseline_summary["train_seed"] = np.nan
    return pd.concat([learned_summary, baseline_summary], ignore_index=True)


def _primary_comparisons(seed_summary: pd.DataFrame) -> list[dict[str, Any]]:
    """Declared return comparisons; one Holm family across reported settings."""

    comparisons: list[tuple[dict[str, Any], Any]] = []
    settings = seed_summary[["condition", "jammer_mode"]].drop_duplicates()
    for setting in settings.itertuples(index=False):
        subset = seed_summary[
            (seed_summary["condition"] == setting.condition)
            & (seed_summary["jammer_mode"] == setting.jammer_mode)
        ]
        hnp = subset[(subset["method"] == "hnp") & subset["train_seed"].notna()]
        if hnp.empty:
            continue
        hnp = hnp.sort_values("train_seed")
        heuristic = (
            "schedule_sweep"
            if setting.jammer_mode == "sweeping"
            else "threshold"
        )
        for comparator in ("matched", heuristic):
            other = subset[subset["method"] == comparator]
            if other.empty:
                continue
            if comparator == "matched":
                joined = hnp[["train_seed", "return"]].merge(
                    other[["train_seed", "return"]],
                    on="train_seed",
                    suffixes=("_hnp", "_other"),
                    validate="one_to_one",
                )
                if joined.empty:
                    continue
                first = joined["return_hnp"].to_numpy()
                second = joined["return_other"].to_numpy()
            else:
                baseline_mean = float(other["return"].iloc[0])
                first = hnp["return"].to_numpy()
                second = np.full(first.shape, baseline_mean, dtype=np.float64)
            comparison = paired_comparison(
                first,
                second,
                method_a="hnp",
                method_b=comparator,
                metric="return",
            )
            comparisons.append(
                (
                    {
                        "condition": setting.condition,
                        "jammer_mode": setting.jammer_mode,
                    },
                    comparison,
                )
            )
    adjusted = apply_holm([item[1] for item in comparisons])
    rows: list[dict[str, Any]] = []
    for (setting, _), comparison in zip(comparisons, adjusted):
        rows.append({**setting, **comparison.to_dict()})
    return rows


def _condition_manifest(condition: EvaluationCondition) -> dict[str, Any]:
    split = condition.split
    return {
        "name": condition.name,
        "role": condition.role,
        "split_name": split.name,
        "distance_cm": split.distance_cm,
        "power_dbm": split.power_dbm,
        "scan_ids": list(split.scan_ids),
        "source_file_count": len(split.source_files),
        "observation_count": int(len(split.observations)),
    }


def write_predeclared_evaluation_schedule(
    conditions: list[EvaluationCondition],
    *,
    jammer_modes: tuple[str, ...],
    config: ExperimentConfig,
    output_dir: Path,
) -> tuple[
    dict[str, tuple[EvaluationTrajectory, ...]], list[dict[str, Any]], Path
]:
    """Write the forced scan/start schedule before any evaluation policy step."""

    schedules_by_condition: dict[str, tuple[EvaluationTrajectory, ...]] = {}
    schedule_rows: list[dict[str, Any]] = []
    for condition in conditions:
        schedule = build_balanced_scan_schedule(
            condition.split,
            config.eval_seeds,
            episodes_per_seed=config.evaluation_episodes_per_seed,
        )
        schedules_by_condition[condition.name] = schedule
        for jammer_mode in jammer_modes:
            for eval_index, trajectory in enumerate(schedule):
                trajectory_seed = _derived_seed(
                    trajectory.eval_seed,
                    condition.name,
                    jammer_mode,
                    "trajectory",
                    trajectory.scan_id,
                    trajectory.scan_replicate,
                )
                audit_env = AntiJammingEnv(
                    condition.split,
                    jammer_mode=jammer_mode,
                    episode_length=config.episode_length,
                    switch_cost=config.switch_cost,
                    reward_safe=config.reward_safe,
                    reward_collision=config.reward_collision,
                    seed=trajectory_seed,
                    random_start=True,
                )
                _, audit_info = audit_env.reset(
                    seed=trajectory_seed,
                    options={"scan_id": trajectory.scan_id},
                )
                schedule_rows.append(
                    {
                        "condition": condition.name,
                        "condition_role": condition.role,
                        "jammer_mode": jammer_mode,
                        "eval_index": int(eval_index),
                        "eval_seed": trajectory.eval_seed,
                        "scan_id": trajectory.scan_id,
                        "replicate": trajectory.scan_replicate,
                        "scan_replicate": trajectory.scan_replicate,
                        "forced_reset_options": {
                            "scan_id": trajectory.scan_id,
                        },
                        "start_index": int(audit_info["window_index"]),
                        "window_start": int(audit_info["window_start"]),
                        "trajectory_seed": trajectory_seed,
                    }
                )

    schedule_path = output_dir / "PREDECLARED_EVALUATION_SCHEDULE.json"
    _write_json(
        schedule_path,
        {
            "schema_version": 1,
            "statement": (
                "This forced scan/start schedule was materialized before any "
                "evaluation policy step or policy-performance outcome."
            ),
            "entries": schedule_rows,
        },
    )
    flat_rows = [
        {
            **row,
            "forced_reset_options": json.dumps(
                row["forced_reset_options"], sort_keys=True
            ),
        }
        for row in schedule_rows
    ]
    pd.DataFrame(flat_rows).to_csv(
        output_dir / "evaluation_schedule.csv", index=False
    )
    return schedules_by_condition, schedule_rows, schedule_path


def _core_code_hashes(project_root: Path) -> dict[str, str]:
    relative_paths = (
        "run_experiment.py",
        "src/agent.py",
        "src/baselines.py",
        "src/config.py",
        "src/data.py",
        "src/env.py",
        "src/models.py",
        "src/runner.py",
        "src/statistics.py",
    )
    return {
        relative_path: _sha256(project_root / relative_path)
        for relative_path in relative_paths
    }


def _load_evaluation_conditions(
    config: ExperimentConfig,
    data_dir: Path,
    development: DatasetBundle,
) -> list[EvaluationCondition]:
    if config.mode == "smoke":
        # Engineering checks cannot accidentally inspect pilot or OOD outcomes.
        return [
            EvaluationCondition(
                "development_validation", development.val, "development_only"
            )
        ]
    distance = build_ood_dataset(
        data_dir,
        distance_cm=config.distance_shift_cm,
        power_dbm=config.development_power_dbm,
        window_size=config.window_size,
        stride=config.stride,
    )
    power = build_ood_dataset(
        data_dir,
        distance_cm=config.development_distance_cm,
        power_dbm=config.power_shift_dbm,
        window_size=config.window_size,
        stride=config.stride,
    )
    return [
        EvaluationCondition(
            "within_condition_pilot", development.test, "transparent_pilot"
        ),
        EvaluationCondition(
            "distance_shift_40cm_10dBm", distance["ood"], "cross_configuration"
        ),
        EvaluationCondition(
            "power_shift_20cm_5dBm", power["ood"], "cross_configuration"
        ),
    ]


def run_experiment(config: ExperimentConfig, paths: RunPaths) -> dict[str, Any]:
    """Execute a complete run and return a path/count summary.

    All learned agents and the train/validation-fitted threshold are frozen and
    hashed before any policy-performance outcome is computed on either declared
    cross-configuration set. Prior format/finiteness auditing is compatible
    with this boundary and is not represented as outcome inspection.
    """

    require_torch()
    if not paths.data_dir.is_dir():
        raise FileNotFoundError(paths.data_dir)
    if paths.output_dir.exists() and any(paths.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {paths.output_dir}"
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = paths.output_dir / "checkpoints"
    checkpoint_dir.mkdir()

    _write_json(paths.output_dir / "config.json", config.to_dict())
    _write_json(paths.output_dir / "software.json", _software_manifest())

    development_kwargs: dict[str, Any] = {}
    if config.mode == "smoke":
        # Do not even construct pilot windows during engineering checks.
        development_kwargs["split_scan_ids"] = {
            "train": tuple(range(0, 7)),
            "val": (7,),
        }
    development = build_dataset(
        paths.data_dir,
        window_size=config.window_size,
        stride=config.stride,
        distance_cm=config.development_distance_cm,
        power_dbm=config.development_power_dbm,
        **development_kwargs,
    )
    normalizer = RFNormalizer.fit_training_split(
        development.train, rf_dim=config.rf_dim
    )
    _write_json(paths.output_dir / "normalizer.json", normalizer.to_dict())
    baselines = build_baselines(
        development.train,
        development.val,
        seed=0,
        threshold_quantile=config.threshold_quantile,
        switch_cost=config.switch_cost,
        reward_safe=config.reward_safe,
        reward_collision=config.reward_collision,
    )

    agents: dict[tuple[str, str, int], DDQNAgent] = {}
    checkpoints: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for jammer_mode in config.jammer_modes:
        for model_name in config.models:
            for train_seed in config.train_seeds:
                seed_everything(
                    train_seed,
                    deterministic_torch=config.deterministic_torch,
                )
                agent = DDQNAgent(
                    model_name,
                    config,
                    normalizer,
                    seed=train_seed,
                )
                agent.training_jammer_mode = jammer_mode
                started = time.perf_counter()
                rows = train_agent(
                    agent,
                    development.train,
                    jammer_mode=jammer_mode,
                    config=config,
                    train_seed=train_seed,
                )
                training_seconds = time.perf_counter() - started
                training_rows.extend(rows)
                checkpoint = checkpoint_dir / (
                    f"{model_name}_{jammer_mode}_seed{train_seed}.pt"
                )
                agent.save(checkpoint)
                checkpoints.append(
                    {
                        "model": model_name,
                        "jammer_mode": jammer_mode,
                        "train_seed": train_seed,
                        "path": str(checkpoint.relative_to(paths.output_dir)),
                        "sha256": _sha256(checkpoint),
                    }
                )
                cost = benchmark_agent(
                    agent,
                    checkpoint=checkpoint,
                    training_seconds=training_seconds,
                )
                cost["jammer_mode"] = jammer_mode
                cost_rows.append(cost)
                agents[(model_name, jammer_mode, train_seed)] = agent

    training_frame = pd.DataFrame(training_rows)
    training_frame.to_csv(paths.output_dir / "training_history.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(paths.output_dir / "model_costs.csv", index=False)

    freeze_payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "statement": (
            "All models, hyperparameters, seeds, normalizer statistics, and "
            "threshold parameters were frozen before any policy-performance "
            "outcome was computed on the cross-configuration sets. Prior "
            "format/finiteness auditing did not inspect policy outcomes."
        ),
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "normalizer": normalizer.to_dict(),
        "threshold": {
            "value": float(baselines["threshold"].threshold),
            "quantile": float(baselines["threshold"].quantile),
            "fitted_splits": list(baselines["threshold"].fitted_splits),
        },
        "checkpoints": checkpoints,
    }
    # Loading declared files for a format/finiteness check is allowed here; no
    # evaluation policy is stepped and no policy outcome is computed.
    conditions = _load_evaluation_conditions(config, paths.data_dir, development)
    _write_json(
        paths.output_dir / "evaluation_conditions.json",
        [_condition_manifest(condition) for condition in conditions],
    )

    schedules_by_condition, schedule_rows, schedule_path = (
        write_predeclared_evaluation_schedule(
            conditions,
            jammer_modes=config.jammer_modes,
            config=config,
            output_dir=paths.output_dir,
        )
    )
    freeze_payload["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_payload["predeclared_evaluation_schedule"] = {
        "path": schedule_path.name,
        "sha256": _sha256(schedule_path),
        "entry_count": len(schedule_rows),
    }
    freeze_payload["core_code_sha256"] = _core_code_hashes(
        Path(__file__).resolve().parents[1]
    )
    _write_json(paths.output_dir / "FROZEN_BEFORE_EVALUATION.json", freeze_payload)

    # Every evaluation policy step occurs below both immutable artefacts.

    evaluation_rows: list[dict[str, Any]] = []
    for condition in conditions:
        trajectory_schedule = schedules_by_condition[condition.name]
        for jammer_mode in config.jammer_modes:
            for baseline_name, policy in baselines.items():
                supported_modes = getattr(policy, "supported_jammer_modes", None)
                if supported_modes is not None and jammer_mode not in supported_modes:
                    continue
                evaluation_rows.extend(
                    evaluate_policy(
                        policy,
                        condition.split,
                        method=baseline_name,
                        condition=condition.name,
                        role=condition.role,
                        jammer_mode=jammer_mode,
                        eval_seeds=config.eval_seeds,
                        config=config,
                        train_seed=None,
                        trajectory_schedule=trajectory_schedule,
                    )
                )
            for model_name in config.models:
                for train_seed in config.train_seeds:
                    evaluation_rows.extend(
                        evaluate_policy(
                            agents[(model_name, jammer_mode, train_seed)],
                            condition.split,
                            method=model_name,
                            condition=condition.name,
                            role=condition.role,
                            jammer_mode=jammer_mode,
                            eval_seeds=config.eval_seeds,
                            config=config,
                            train_seed=train_seed,
                            trajectory_schedule=trajectory_schedule,
                        )
                    )

    evaluation_frame = pd.DataFrame(evaluation_rows)
    evaluation_frame.to_csv(
        paths.output_dir / "evaluation_episodes.csv", index=False
    )
    seed_summary = _summarise_seeds(evaluation_frame)
    seed_summary.to_csv(paths.output_dir / "seed_summary.csv", index=False)
    comparison_rows = _primary_comparisons(seed_summary)
    comparison_columns = [
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
    ]
    pd.DataFrame(comparison_rows, columns=comparison_columns).to_csv(
        paths.output_dir / "primary_comparisons.csv", index=False
    )
    _write_json(paths.output_dir / "primary_comparisons.json", comparison_rows)

    summary = {
        "mode": config.mode,
        "output_dir": str(paths.output_dir),
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "seed_summary_rows": int(len(seed_summary)),
        "primary_comparisons": len(comparison_rows),
        "checkpoint_count": len(checkpoints),
        "conditions": [_condition_manifest(condition) for condition in conditions],
        "files": {
            "freeze": "FROZEN_BEFORE_EVALUATION.json",
            "training": "training_history.csv",
            "evaluation": "evaluation_episodes.csv",
            "evaluation_schedule": "evaluation_schedule.csv",
            "predeclared_evaluation_schedule": (
                "PREDECLARED_EVALUATION_SCHEDULE.json"
            ),
            "seed_summary": "seed_summary.csv",
            "comparisons": "primary_comparisons.csv",
            "costs": "model_costs.csv",
        },
    }
    _write_json(paths.output_dir / "run_summary.json", summary)
    return summary


__all__ = [
    "EvaluationCondition",
    "EvaluationTrajectory",
    "build_balanced_scan_schedule",
    "preview_jammer_sequence",
    "write_predeclared_evaluation_schedule",
    "model_persistent_tensor_bytes",
    "train_agent",
    "evaluate_policy",
    "benchmark_agent",
    "run_experiment",
]
