"""Isolated development-only cost profiling for the round-2 revision.

This script is intentionally separate from performance experiments.  It never
constructs the pilot split or either cross-configuration split.  Run it only
after concurrent formal jobs have ended and the system is otherwise idle.
"""

from __future__ import annotations

# These must be set before importing torch through ``src.agent``.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.agent import DDQNAgent, RFNormalizer
from src.config import ExperimentConfig
from src.data import DatasetBundle, build_dataset
from src.models import count_trainable_parameters
from src.runner import model_persistent_tensor_bytes, train_agent


PROJECT_ROOT = Path(__file__).resolve().parent
PROFILING_SEEDS: tuple[int, ...] = (91001, 91002, 91003)
PROFILE_MODELS: tuple[str, ...] = ("hnp", "matched")
PROFILE_JAMMER_MODE = "sweeping"
LATENCY_WARMUP = 200
LATENCY_REPETITIONS = 1000
GPU_LATENCY_PACING_SECONDS = 0.005
GPU_LATENCY_PACING_SCOPE = (
    "fixed sleep after every GPU warm-up forward+sync and after every recorded "
    "GPU forward+sync; sleep occurs outside the recorded latency interval"
)
GPU_UTILIZATION_INTENT = (
    "5 ms fixed pacing is designed to keep average utilization substantially "
    "below 85% on an otherwise idle, exclusively profiled GPU; external GPU "
    "load invalidates that operating condition"
)
MEMORY_SCOPE = (
    "parameters + registered buffers only; excludes activations, optimizer "
    "state, framework/runtime objects, temporary workspaces, and allocator overhead"
)
PROFILING_SEED_ROLE = (
    "engineering cost-profiling seeds only; not units for policy-performance inference"
)


def configure_single_thread_cpu() -> None:
    """Force PyTorch CPU kernels to one intra-op and one inter-op thread."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work begins.  A
        # fresh CLI process takes the first branch; tests may import after work.
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "start profile_model_costs.py in a fresh Python process so "
                "inter-op threads can be fixed to one"
            )


def isolated_training_config() -> ExperimentConfig:
    """Exact formal update budget on CPU, with dedicated non-inferential seeds."""

    return ExperimentConfig.preset("formal").with_overrides(
        models=PROFILE_MODELS,
        train_seeds=PROFILING_SEEDS,
        jammer_modes=(PROFILE_JAMMER_MODE,),
        device="cpu",
    )


def load_development_only_dataset(
    data_dir: str | Path, config: ExperimentConfig
) -> DatasetBundle:
    """Load scans 0--7 only; pilot scans 8--9 and OOD files are not constructed."""

    return build_dataset(
        data_dir,
        split_scan_ids={"train": tuple(range(0, 7)), "val": (7,)},
        window_size=config.window_size,
        stride=config.stride,
        distance_cm=config.development_distance_cm,
        power_dbm=config.development_power_dbm,
    )


def counterbalanced_training_jobs() -> tuple[tuple[int, str], ...]:
    """Predetermined blocked order to reduce systematic first-model bias."""

    jobs: list[tuple[int, str]] = []
    for block_index, seed in enumerate(PROFILING_SEEDS):
        order = PROFILE_MODELS if block_index % 2 == 0 else tuple(reversed(PROFILE_MODELS))
        jobs.extend((seed, model_name) for model_name in order)
    return tuple(jobs)


def profile_isolated_training(
    development: DatasetBundle,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Sequentially time fixed-budget CPU training; never evaluate a policy."""

    normalizer = RFNormalizer.fit_training_split(
        development.train, rf_dim=config.rf_dim
    )
    rows: list[dict[str, Any]] = []
    for order_index, (profiling_seed, model_name) in enumerate(
        counterbalanced_training_jobs()
    ):
        initialization_started = time.perf_counter()
        agent = DDQNAgent(
            model_name,
            config,
            normalizer,
            seed=profiling_seed,
            device="cpu",
        )
        initialization_seconds = time.perf_counter() - initialization_started
        training_started = time.perf_counter()
        train_agent(
            agent,
            development.train,
            jammer_mode=PROFILE_JAMMER_MODE,
            config=config,
            train_seed=profiling_seed,
        )
        training_seconds = time.perf_counter() - training_started
        tensor_sizes = model_persistent_tensor_bytes(agent.online)
        rows.append(
            {
                "record_type": "isolated_training_wall_time",
                "profile_order": order_index,
                "model": model_name,
                "jammer_mode": PROFILE_JAMMER_MODE,
                "profiling_seed": profiling_seed,
                "profiling_seed_role": PROFILING_SEED_ROLE,
                "training_data_scope": "20cm/10dBm train scans 0-6 only",
                "validation_loaded_not_used_for_training": True,
                "train_episodes": config.train_episodes,
                "episode_length": config.episode_length,
                "environment_steps": agent.environment_steps,
                "optimizer_updates": agent.update_steps,
                "cpu_intraop_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(),
                "agent_initialization_wall_seconds": initialization_seconds,
                "training_loop_wall_seconds": training_seconds,
                "agent_init_plus_training_wall_seconds": (
                    initialization_seconds + training_seconds
                ),
                "trainable_parameters": count_trainable_parameters(agent.online),
                **tensor_sizes,
                "inference_persistent_tensor_bytes_scope": MEMORY_SCOPE,
            }
        )
        del agent
        gc.collect()
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_candidates(roots: Iterable[str | Path]) -> list[Path]:
    candidates: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            candidates.add(root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        checkpoint_dir = root / "checkpoints"
        search_root = checkpoint_dir if checkpoint_dir.is_dir() else root
        candidates.update(path.resolve() for path in search_root.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError("no .pt checkpoint found in the supplied roots")
    return sorted(candidates)


def _jammer_mode_from_filename(path: Path) -> str | None:
    match = re.search(r"_(sweeping|random)_seed\d+\.pt$", path.name)
    return None if match is None else match.group(1)


def canonical_checkpoint_manifest(
    roots: Iterable[str | Path],
    *,
    expected_train_seeds: Sequence[int] = tuple(range(1, 11)),
    expected_jammer_modes: Sequence[str] = ("sweeping", "random"),
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Discover primary checkpoints and reject duplicates or protocol drift."""

    expected_seeds = tuple(int(seed) for seed in expected_train_seeds)
    expected_modes = tuple(str(mode) for mode in expected_jammer_modes)
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    formal = ExperimentConfig.preset("formal")
    protocol_fields = (
        "train_episodes",
        "episode_length",
        "gamma",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "tau",
        "gradient_clip_norm",
        "epsilon_end",
        "epsilon_denominator_episodes",
    )
    for path in _checkpoint_candidates(roots):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(payload.get("model_name", "")).lower()
        if model_name not in PROFILE_MODELS:
            continue
        seed = int(payload["seed"])
        jammer_mode = payload.get("training_jammer_mode")
        if jammer_mode is None:
            jammer_mode = _jammer_mode_from_filename(path)
        if jammer_mode not in expected_modes:
            continue
        saved_config = ExperimentConfig(**payload["config"])
        drift = {
            field: (getattr(saved_config, field), getattr(formal, field))
            for field in protocol_fields
            if getattr(saved_config, field) != getattr(formal, field)
        }
        if saved_config.mode != "formal":
            drift["mode"] = (saved_config.mode, "formal")
        if drift:
            raise ValueError(f"checkpoint {path} is not canonical formal: {drift}")
        key = (model_name, str(jammer_mode), seed)
        if key in records:
            raise ValueError(
                f"duplicate canonical checkpoint for {key}: "
                f"{records[key]['checkpoint_path']} and {path}"
            )
        records[key] = {
            "model": model_name,
            "jammer_mode": str(jammer_mode),
            "canonical_train_seed": seed,
            "checkpoint_path": str(path),
            "checkpoint_sha256": _sha256(path),
            "checkpoint_bytes": path.stat().st_size,
        }

    if require_complete:
        expected_keys = {
            (model, jammer_mode, seed)
            for model in PROFILE_MODELS
            for jammer_mode in expected_modes
            for seed in expected_seeds
        }
        missing = sorted(expected_keys - set(records))
        extra = sorted(set(records) - expected_keys)
        if missing or extra:
            raise ValueError(
                f"canonical checkpoint set mismatch; missing={missing}, extra={extra}"
            )
    return [records[key] for key in sorted(records)]


def _serialized_state_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def fixed_batch1_latency_ms(
    model: torch.nn.Module,
    *,
    device: torch.device,
    input_dim: int = 48,
    warmup: int = LATENCY_WARMUP,
    repetitions: int = LATENCY_REPETITIONS,
    pacing_sleep_seconds: float = 0.0,
) -> dict[str, float]:
    """Measure synchronized end-to-end batch-1 forward latency sequentially."""

    if int(warmup) < 0 or int(repetitions) <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    if not np.isfinite(float(pacing_sleep_seconds)) or float(
        pacing_sleep_seconds
    ) < 0:
        raise ValueError("pacing_sleep_seconds must be finite and non-negative")
    pacing_sleep_seconds = float(pacing_sleep_seconds)
    model = model.to(device).eval()
    sample = torch.zeros((1, input_dim), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(int(warmup)):
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if pacing_sleep_seconds > 0:
                time.sleep(pacing_sleep_seconds)
        samples = np.empty(int(repetitions), dtype=np.float64)
        for index in range(int(repetitions)):
            started = time.perf_counter_ns()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples[index] = (time.perf_counter_ns() - started) / 1e6
            # Deliberately outside the timed interval above. This reduces GPU
            # duty cycle without inflating the reported forward latency.
            if pacing_sleep_seconds > 0:
                time.sleep(pacing_sleep_seconds)
    return {
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "p05_ms": float(np.quantile(samples, 0.05)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


def profile_canonical_checkpoint_inference(
    manifest: Sequence[Mapping[str, Any]],
    *,
    warmup: int = LATENCY_WARMUP,
    repetitions: int = LATENCY_REPETITIONS,
) -> list[dict[str, Any]]:
    """Sequentially remeasure every canonical seed checkpoint on CPU/GPU."""

    rows: list[dict[str, Any]] = []
    cuda_available = torch.cuda.is_available()
    for order_index, record in enumerate(manifest):
        checkpoint_path = Path(str(record["checkpoint_path"]))
        agent = DDQNAgent.load(checkpoint_path, device="cpu")
        cpu_model = agent.online.eval()
        tensor_sizes = model_persistent_tensor_bytes(cpu_model)
        cpu = fixed_batch1_latency_ms(
            cpu_model,
            device=torch.device("cpu"),
            input_dim=agent.config.input_dim,
            warmup=warmup,
            repetitions=repetitions,
            pacing_sleep_seconds=0.0,
        )
        gpu: dict[str, float] | None = None
        if cuda_available:
            gpu_model = copy.deepcopy(cpu_model).to("cuda").eval()
            gpu = fixed_batch1_latency_ms(
                gpu_model,
                device=torch.device("cuda"),
                input_dim=agent.config.input_dim,
                warmup=warmup,
                repetitions=repetitions,
                pacing_sleep_seconds=GPU_LATENCY_PACING_SECONDS,
            )
            del gpu_model
            torch.cuda.empty_cache()
        rows.append(
            {
                "record_type": "canonical_checkpoint_inference",
                "profile_order": order_index,
                **dict(record),
                "profiling_seed_role": PROFILING_SEED_ROLE,
                "latency_input": "batch-1 all-zero normalized 48D observation",
                "latency_warmup": int(warmup),
                "latency_repetitions": int(repetitions),
                "latency_synchronization": (
                    "per-forward torch.cuda.synchronize" if cuda_available else "CPU call return"
                ),
                "gpu_latency_pacing_seconds": (
                    GPU_LATENCY_PACING_SECONDS if cuda_available else None
                ),
                "gpu_latency_pacing_scope": (
                    GPU_LATENCY_PACING_SCOPE
                    if cuda_available
                    else "not applied because CUDA was unavailable"
                ),
                "gpu_utilization_intent": GPU_UTILIZATION_INTENT,
                "batch1_cpu_latency_ms_median": cpu["median_ms"],
                "batch1_cpu_latency_ms_mean": cpu["mean_ms"],
                "batch1_cpu_latency_ms_p05": cpu["p05_ms"],
                "batch1_cpu_latency_ms_p95": cpu["p95_ms"],
                "batch1_gpu_latency_ms_median": None if gpu is None else gpu["median_ms"],
                "batch1_gpu_latency_ms_mean": None if gpu is None else gpu["mean_ms"],
                "batch1_gpu_latency_ms_p05": None if gpu is None else gpu["p05_ms"],
                "batch1_gpu_latency_ms_p95": None if gpu is None else gpu["p95_ms"],
                "trainable_parameters": count_trainable_parameters(cpu_model),
                **tensor_sizes,
                "inference_persistent_tensor_bytes_scope": MEMORY_SCOPE,
                "serialized_state_bytes": _serialized_state_bytes(cpu_model),
                "serialized_state_bytes_role": "storage size, not memory",
                "checkpoint_bytes_role": "storage size, not memory",
            }
        )
        del agent, cpu_model
        gc.collect()
    return rows


def _metadata() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "isolated engineering cost comparison; no policy-performance inference",
        "data_scope": "development 20cm/10dBm train scans 0-6; validation scan 7 loaded only for provenance",
        "excluded_data": "pilot scans 8-9 and all cross-configuration files are never constructed",
        "profiling_seeds": list(PROFILING_SEEDS),
        "profiling_seed_role": PROFILING_SEED_ROLE,
        "training_order": [list(job) for job in counterbalanced_training_jobs()],
        "latency_warmup": LATENCY_WARMUP,
        "latency_repetitions": LATENCY_REPETITIONS,
        "gpu_latency_pacing_seconds": GPU_LATENCY_PACING_SECONDS,
        "gpu_latency_pacing_scope": GPU_LATENCY_PACING_SCOPE,
        "gpu_utilization_intent": GPU_UTILIZATION_INTENT,
        "memory_scope": MEMORY_SCOPE,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_intraop_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
    }


def write_isolated_outputs(
    output_dir: str | Path,
    training_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "isolated_model_costs.csv"
    json_path = output_dir / "isolated_model_costs.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing isolated cost profile; choose a new output directory"
        )
    combined = [dict(row) for row in training_rows] + [
        dict(row) for row in inference_rows
    ]
    pd.DataFrame(combined).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "metadata": _metadata(),
                "isolated_training": [dict(row) for row in training_rows],
                "canonical_checkpoint_inference": [
                    dict(row) for row in inference_rows
                ],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential, development-only HNP/matched cost profiling. Run only "
            "after all concurrent formal jobs have stopped."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        required=True,
        help="Canonical formal run directory or checkpoints directory; repeat as needed",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT.parent / "analysis"
    )
    parser.add_argument(
        "--acknowledge-idle-system",
        action="store_true",
        help="Required confirmation that concurrent jobs have ended",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.acknowledge_idle_system:
        raise SystemExit(
            "Refusing to profile while system-idle status is unconfirmed. "
            "Re-run after all formal jobs end with --acknowledge-idle-system."
        )
    configure_single_thread_cpu()
    config = isolated_training_config()
    development = load_development_only_dataset(args.data_dir, config)
    training_rows = profile_isolated_training(development, config)
    manifest = canonical_checkpoint_manifest(args.checkpoint_root)
    inference_rows = profile_canonical_checkpoint_inference(manifest)
    csv_path, json_path = write_isolated_outputs(
        args.output_dir, training_rows, inference_rows
    )
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
