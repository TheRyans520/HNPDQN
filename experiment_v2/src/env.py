"""Gym-free anti-jamming environment with observable switching state.

The 40 measured RF features are augmented by an eight-dimensional one-hot
encoding of the previous action.  This makes the switching-dependent reward
observable to the policy; it does not by itself establish that the measured RF
process is fully Markov.  Before the first action the one-hot block is all zero.

The API follows Gymnasium's tuple convention without importing Gym:
``reset() -> (observation, info)`` and
``step(action) -> (observation, reward, terminated, truncated, info)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import STATE_FEATURES, ScanSplit


DEFAULT_TRAIN_SEED = 20260810
DEFAULT_EVAL_SEED = 120260810


@dataclass
class DiscreteSpace:
    """Small subset of ``gym.spaces.Discrete`` used by existing runners."""

    n: int
    _rng: np.random.Generator

    def sample(self) -> int:
        return int(self._rng.integers(self.n))

    def seed(self, seed: int | None = None) -> list[int]:
        actual = int(seed if seed is not None else np.random.SeedSequence().entropy)
        self._rng = np.random.default_rng(actual)
        return [actual]


@dataclass(frozen=True)
class ArraySpace:
    """Shape/dtype descriptor compatible with common observation checks."""

    shape: tuple[int, ...]
    dtype: np.dtype


class AntiJammingEnv:
    """Replay measured RF states under a sweeping or random jammer.

    Parameters
    ----------
    split:
        A scan-level :class:`~src.data.ScanSplit`.  At every time position the
        environment selects the measured file matching the current jammer.
    jammer_mode:
        ``"sweeping"`` cycles through channels; ``"random"`` samples a new
        channel uniformly at each transition.  ``"constant"`` is provided for
        diagnostics but is not a primary reported setting.
    seed:
        Local RNG seed.  If omitted, train and non-train splits receive
        different defaults.  No global NumPy RNG is touched.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        split: ScanSplit,
        *,
        jammer_mode: str = "sweeping",
        episode_length: int = 100,
        switch_cost: float = 0.1,
        reward_safe: float = 1.0,
        reward_collision: float = -1.0,
        seed: int | None = None,
        random_start: bool = True,
    ) -> None:
        if jammer_mode not in {"sweeping", "random", "constant"}:
            raise ValueError(
                "jammer_mode must be 'sweeping', 'random', or 'constant'"
            )
        if int(episode_length) <= 0:
            raise ValueError("episode_length must be positive")
        if float(switch_cost) < 0:
            raise ValueError("switch_cost must be non-negative")
        if tuple(split.jammer_channels) != tuple(split.channels):
            raise ValueError(
                "split must contain one trajectory for every configured channel"
            )
        if not split.scan_ids:
            raise ValueError("split contains no scans")

        self.split = split
        self.channels = tuple(split.channels)
        self.n_actions = len(self.channels)
        self.jammer_mode = jammer_mode
        self.episode_length = int(episode_length)
        self.switch_cost = float(switch_cost)
        self.reward_safe = float(reward_safe)
        self.reward_collision = float(reward_collision)
        self.random_start = bool(random_start)

        default_seed = (
            DEFAULT_TRAIN_SEED if split.name.lower() == "train" else DEFAULT_EVAL_SEED
        )
        self._seed = int(default_seed if seed is None else seed)
        self._rng = np.random.default_rng(self._seed)
        self.action_space = DiscreteSpace(
            self.n_actions, np.random.default_rng(self._seed + 1)
        )
        self.observation_size = split.observation_size + self.n_actions
        self.observation_space = ArraySpace(
            (self.observation_size,), np.dtype(np.float32)
        )

        self.previous_action: int | None = None
        self.freq: int | None = None
        self.jammer_channel: int | None = None
        self.scan_id: int | None = None
        self.window_index = 0
        self.window_start = 0
        self.step_count = 0
        self.collision_count = 0
        self.switch_count = 0
        self._jammer_index = 0
        self._common_length = 0
        self._episode_done = True

    @property
    def seed_value(self) -> int:
        return self._seed

    def seed(self, seed: int | None = None) -> list[int]:
        """Reset only this environment's random stream."""

        actual = int(seed if seed is not None else np.random.SeedSequence().entropy)
        self._seed = actual
        self._rng = np.random.default_rng(actual)
        self.action_space.seed(actual + 1)
        return [actual]

    def _action_one_hot(self) -> np.ndarray:
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        if self.previous_action is not None:
            one_hot[self.previous_action] = 1.0
        return one_hot

    def _observation(self) -> np.ndarray:
        if self.scan_id is None or self.jammer_channel is None:
            raise RuntimeError("reset() must be called before requesting state")
        features = self.split.trajectory(
            self.scan_id, self.jammer_channel
        )[self.window_index]
        observation = np.concatenate((features, self._action_one_hot()))
        return observation.astype(np.float32, copy=False)

    def _base_info(
        self,
        *,
        collision: bool,
        switched: bool,
        acted_jammer_channel: int | None,
    ) -> dict[str, Any]:
        """Build transition metadata.

        ``jammer_channel`` always describes the observation returned alongside
        this info dictionary.  On ``step``, ``acted_jammer_channel`` records
        the jammer against which the just-completed action was scored.
        """

        assert self.jammer_channel is not None
        assert self.scan_id is not None
        return {
            "collision": bool(collision),
            "switched": bool(switched),
            "jammer_channel": int(self.jammer_channel),
            "jammer_index": int(self._jammer_index),
            "acted_jammer_channel": (
                None
                if acted_jammer_channel is None
                else int(acted_jammer_channel)
            ),
            "scan_id": int(self.scan_id),
            "window_index": int(self.window_index),
            "window_start": int(
                self.split.window_starts[
                    next(
                        key
                        for key in self.split.window_starts
                        if key.scan_id == self.scan_id
                        and key.jammer_channel == self.jammer_channel
                    )
                ][self.window_index]
            ),
            "step_count": int(self.step_count),
            "collision_count": int(self.collision_count),
            "switch_count": int(self.switch_count),
            "previous_action": self.previous_action,
            "selected_channel": self.freq,
            "distance_cm": self.split.distance_cm,
            "power_dbm": self.split.power_dbm,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an independent episode and clear all cross-episode state."""

        if seed is not None:
            self.seed(seed)
        options = {} if options is None else dict(options)

        # These fields were not reset in the legacy environment.  Clearing all
        # of them is required for a valid episode boundary.
        self.previous_action = None
        self.freq = None
        self.step_count = 0
        self.collision_count = 0
        self.switch_count = 0

        if "scan_id" in options:
            scan_id = int(options["scan_id"])
            if scan_id not in self.split.scan_ids:
                raise ValueError(f"scan_id {scan_id} is not in split {self.split.name}")
            self.scan_id = scan_id
        else:
            self.scan_id = int(self._rng.choice(self.split.scan_ids))

        self._common_length = self.split.common_length(self.scan_id)
        if self._common_length < 2:
            raise ValueError("a trajectory needs at least two windows")

        max_start = max(0, self._common_length - self.episode_length - 1)
        if "start_index" in options:
            start_index = int(options["start_index"])
            if not 0 <= start_index < self._common_length - 1:
                raise ValueError(
                    f"start_index must be in [0, {self._common_length - 2}]"
                )
            self.window_start = start_index
        elif self.random_start and max_start > 0:
            self.window_start = int(self._rng.integers(max_start + 1))
        else:
            self.window_start = 0
        self.window_index = self.window_start

        if "jammer_channel" in options:
            requested_jammer = int(options["jammer_channel"])
            if requested_jammer not in self.channels:
                raise ValueError(f"unknown jammer channel {requested_jammer}")
            self._jammer_index = self.channels.index(requested_jammer)
        elif self.jammer_mode == "sweeping":
            # The declared sweeping protocol begins at the first candidate
            # channel; callers can override this explicitly for phase studies.
            self._jammer_index = 0
        else:
            self._jammer_index = int(self._rng.integers(self.n_actions))
        self.jammer_channel = self.channels[self._jammer_index]
        self._episode_done = False

        observation = self._observation()
        info = self._base_info(
            collision=False, switched=False, acted_jammer_channel=None
        )
        return observation, info

    def _advance_jammer(self) -> None:
        if self.jammer_mode == "sweeping":
            self._jammer_index = (self._jammer_index + 1) % self.n_actions
        elif self.jammer_mode == "random":
            self._jammer_index = int(self._rng.integers(self.n_actions))
        self.jammer_channel = self.channels[self._jammer_index]

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is done; call reset() before step()")
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise TypeError("action must be an integer channel index")
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]")

        assert self.jammer_channel is not None
        acted_jammer = self.jammer_channel
        collision = self.channels[action] == acted_jammer
        switched = (
            self.previous_action is not None and action != self.previous_action
        )

        reward = self.reward_collision if collision else self.reward_safe
        # Match the manuscript's declared three cases: collision=-1,
        # successful retention=+1, and successful switch=1-C_switch.  Collision
        # is the dominant event, so it does not receive an additional cost.
        if switched and not collision:
            reward -= self.switch_cost

        self.collision_count += int(collision)
        self.switch_count += int(switched)
        self.previous_action = action
        self.freq = self.channels[action]
        self.step_count += 1
        self.window_index += 1
        self._advance_jammer()

        reached_horizon = self.step_count >= self.episode_length
        reached_data_end = self.window_index >= self._common_length - 1
        # Neither condition is an absorbing task state.  Both are external
        # time/data limits, so value targets are allowed to bootstrap across
        # the final collected transition (Gymnasium time-limit semantics).
        terminated = False
        truncated = bool(reached_horizon or reached_data_end)
        self._episode_done = terminated or truncated

        observation = self._observation()
        info = self._base_info(
            collision=collision,
            switched=switched,
            acted_jammer_channel=acted_jammer,
        )
        return observation, float(reward), terminated, truncated, info


def make_train_eval_envs(
    train_split: ScanSplit,
    eval_split: ScanSplit,
    *,
    train_seed: int = DEFAULT_TRAIN_SEED,
    eval_seed: int = DEFAULT_EVAL_SEED,
    **env_kwargs: Any,
) -> tuple[AntiJammingEnv, AntiJammingEnv]:
    """Construct explicitly independent train/evaluation random streams."""

    if int(train_seed) == int(eval_seed):
        raise ValueError("train_seed and eval_seed must be different")
    return (
        AntiJammingEnv(train_split, seed=int(train_seed), **env_kwargs),
        AntiJammingEnv(eval_split, seed=int(eval_seed), **env_kwargs),
    )


# Compatibility alias for older scripts, while keeping the repaired behavior.
RfEnvironment = AntiJammingEnv
SpectrumSelectionEnv = AntiJammingEnv


__all__ = [
    "DEFAULT_TRAIN_SEED",
    "DEFAULT_EVAL_SEED",
    "DiscreteSpace",
    "ArraySpace",
    "AntiJammingEnv",
    "RfEnvironment",
    "SpectrumSelectionEnv",
    "make_train_eval_envs",
]
