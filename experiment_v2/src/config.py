"""Frozen configuration objects for the round-2 experiment.

The presets deliberately separate engineering smoke checks from the declared
formal run.  ``smoke`` evaluates only the development validation scan; it does
not load either cross-configuration condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable


FORMAL_TRAIN_SEEDS: tuple[int, ...] = tuple(range(1, 11))
FORMAL_EVAL_SEEDS: tuple[int, ...] = tuple(range(1001, 1021))
FORMAL_JAMMER_MODES: tuple[str, ...] = ("sweeping", "random")
PRIMARY_MODELS: tuple[str, ...] = ("hnp", "matched")
ABLATION_MODELS: tuple[str, ...] = (
    "no_polynomial",
    "no_layernorm",
    "no_dueling",
)
SUPPORTED_MODELS: tuple[str, ...] = PRIMARY_MODELS + ABLATION_MODELS


def _int_tuple(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError("a seed list must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"seed list contains duplicates: {result}")
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, JSON-serialisable training and evaluation configuration."""

    mode: str = "smoke"
    models: tuple[str, ...] = ("hnp",)
    train_seeds: tuple[int, ...] = (1,)
    eval_seeds: tuple[int, ...] = (1001, 1002)
    jammer_modes: tuple[str, ...] = ("sweeping",)

    input_dim: int = 48
    rf_dim: int = 40
    n_actions: int = 8
    episode_length: int = 100
    train_episodes: int = 4
    evaluation_episodes_per_seed: int = 1

    gamma: float = 0.95
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 64
    train_frequency: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    gradient_clip_norm: float = 0.7
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_denominator_episodes: float = 10.0
    epsilon_decay_steps: int = 20_000  # retained for manifest compatibility
    switch_cost: float = 0.1
    reward_safe: float = 1.0
    reward_collision: float = -1.0

    window_size: int = 32
    stride: int = 1
    development_distance_cm: int = 20
    development_power_dbm: int = 10
    distance_shift_cm: int = 40
    power_shift_dbm: int = 5
    threshold_quantile: float = 0.5

    deterministic_torch: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("mode must be 'smoke' or 'formal'")
        object.__setattr__(self, "models", tuple(str(v).lower() for v in self.models))
        object.__setattr__(self, "train_seeds", _int_tuple(self.train_seeds))
        object.__setattr__(self, "eval_seeds", _int_tuple(self.eval_seeds))
        object.__setattr__(
            self, "jammer_modes", tuple(str(v).lower() for v in self.jammer_modes)
        )
        unknown = sorted(set(self.models) - set(SUPPORTED_MODELS))
        if unknown:
            raise ValueError(f"unsupported models: {unknown}")
        if not self.models:
            raise ValueError("models must not be empty")
        bad_jammers = sorted(
            set(self.jammer_modes) - {"sweeping", "random", "constant"}
        )
        if bad_jammers or not self.jammer_modes:
            raise ValueError(f"invalid jammer modes: {bad_jammers}")
        if set(self.train_seeds) & set(self.eval_seeds):
            raise ValueError("training and evaluation seeds must be disjoint")
        for name in (
            "input_dim",
            "rf_dim",
            "n_actions",
            "episode_length",
            "train_episodes",
            "evaluation_episodes_per_seed",
            "batch_size",
            "replay_capacity",
            "train_frequency",
            "gradient_steps",
            "epsilon_decay_steps",
            "window_size",
            "stride",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.input_dim != self.rf_dim + self.n_actions:
            raise ValueError("input_dim must equal rf_dim + n_actions")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if self.epsilon_denominator_episodes <= 0:
            raise ValueError("epsilon_denominator_episodes must be positive")
        if not 0.0 <= self.threshold_quantile <= 1.0:
            raise ValueError("threshold_quantile must be in [0, 1]")

    @classmethod
    def preset(cls, mode: str) -> "ExperimentConfig":
        """Return a frozen smoke or formal preset.

        Formal training uses a fixed update budget and never performs test-set
        early stopping.  Any command-line override is recorded in the manifest.
        """

        mode = str(mode).lower()
        if mode == "smoke":
            return cls()
        if mode == "formal":
            return cls(
                mode="formal",
                # Ablations are explicit secondary runs and never enter the
                # default primary protocol or its Holm comparison family.
                models=PRIMARY_MODELS,
                train_seeds=FORMAL_TRAIN_SEEDS,
                eval_seeds=FORMAL_EVAL_SEEDS,
                jammer_modes=FORMAL_JAMMER_MODES,
                train_episodes=200,
                replay_capacity=100_000,
                learning_starts=64,
            )
        raise ValueError("mode must be 'smoke' or 'formal'")

    def with_overrides(self, **changes: Any) -> "ExperimentConfig":
        cleaned = {key: value for key, value in changes.items() if value is not None}
        return replace(self, **cleaned)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunPaths:
    """Resolved input/output locations kept outside the numeric config."""

    data_dir: Path
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )


__all__ = [
    "FORMAL_TRAIN_SEEDS",
    "FORMAL_EVAL_SEEDS",
    "FORMAL_JAMMER_MODES",
    "PRIMARY_MODELS",
    "ABLATION_MODELS",
    "SUPPORTED_MODELS",
    "ExperimentConfig",
    "RunPaths",
]
