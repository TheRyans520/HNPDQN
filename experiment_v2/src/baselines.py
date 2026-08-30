"""Transparent non-learning references for the anti-jamming experiment.

Ordinary observable policies never inspect ``info['jammer_channel']``.  The
schedule-aware sweeping rule uses only the protocol's public deterministic
phase, the jammer-aware greedy reference is explicitly privileged with the
current jammer label, and the clairvoyant oracle is an explicitly labelled
full-horizon reward upper bound.  The raw CSV
column named ``snr`` rises on the directly jammed channel in this corpus, so it
is treated as an *interference indicator*: lower raw SNR is better.  The
reported max-quality score is therefore ``quality = -snr_mean``.  Naming this
direction explicitly avoids the misleading legacy "maximum SNR" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .data import CHANNELS, STATE_FEATURES, ScanSplit


SNR_SCORE_DIRECTION = "lower raw snr_mean is better; quality = -snr_mean"


def _channel_matrix(
    observation: np.ndarray | Sequence[float], n_channels: int
) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    feature_count = n_channels * len(STATE_FEATURES)
    if values.size < feature_count:
        raise ValueError(
            f"observation has {values.size} values; at least {feature_count} required"
        )
    return values[:feature_count].reshape(n_channels, len(STATE_FEATURES))


def quality_scores(
    observation: np.ndarray | Sequence[float],
    *,
    n_channels: int = len(CHANNELS),
) -> np.ndarray:
    """Return a score to maximize using only measured features.

    In these measurements, the jammer increases ``snr_mean``; hence negation
    turns the quantity into a max-quality/min-interference score.
    """

    matrix = _channel_matrix(observation, n_channels)
    scores = -matrix[:, 0]
    return np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def previous_action_from_observation(
    observation: np.ndarray | Sequence[float],
    *,
    n_channels: int = len(CHANNELS),
) -> int | None:
    """Decode the previous-action block; all zeros means episode start."""

    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    feature_count = n_channels * len(STATE_FEATURES)
    if values.size < feature_count + n_channels:
        return None
    one_hot = values[feature_count : feature_count + n_channels]
    if not np.isfinite(one_hot).all() or np.isclose(one_hot.sum(), 0.0):
        return None
    return int(np.argmax(one_hot))


class RandomPolicy:
    """Uniform random channel selection using a policy-local RNG."""

    name = "Random"

    def __init__(self, n_actions: int = len(CHANNELS), *, seed: int = 0) -> None:
        if int(n_actions) <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = int(n_actions)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del observation, info
        return int(self._rng.integers(self.n_actions))


class StayPolicy:
    """Select one predeclared channel and never switch."""

    name = "Stay"

    def __init__(
        self, n_actions: int = len(CHANNELS), *, initial_action: int = 0
    ) -> None:
        if not 0 <= int(initial_action) < int(n_actions):
            raise ValueError("initial_action is outside the action space")
        self.n_actions = int(n_actions)
        self.initial_action = int(initial_action)

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del observation, info
        return self.initial_action


class MinInterferencePolicy:
    """Choose the observed channel with minimum raw ``snr_mean``."""

    name = "Max-Quality (Min-Interference)"
    score_direction = SNR_SCORE_DIRECTION

    def __init__(self, n_actions: int = len(CHANNELS)) -> None:
        self.n_actions = int(n_actions)

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        # `info` is intentionally discarded: this is an observable baseline,
        # not an oracle with access to the experimental jammer label.
        del info
        return int(np.argmax(quality_scores(observation, n_channels=self.n_actions)))


# A reader-facing alias: it maximizes the explicitly defined quality score.
MaxQualityPolicy = MinInterferencePolicy


@dataclass
class ThresholdPolicy:
    """Stay when the current channel clears a pre-fitted quality threshold.

    The threshold is a fixed quantile of observed channel qualities in train
    (and optionally validation) scans.  It never uses collision/jammer labels.
    Test or OOD splits are rejected by :meth:`fit`, preventing accidental
    post-hoc tuning on reported evaluation conditions.
    """

    threshold: float
    n_actions: int = len(CHANNELS)
    quantile: float = 0.5
    fitted_splits: tuple[str, ...] = ("train",)
    name: str = "Threshold/Hysteresis"

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not 0.0 <= float(self.quantile) <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        self._previous_action: int | None = None

    @classmethod
    def fit(
        cls,
        train_split: ScanSplit,
        validation_split: ScanSplit | None = None,
        *,
        quantile: float = 0.5,
    ) -> "ThresholdPolicy":
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        requested = [train_split]
        if validation_split is not None:
            requested.append(validation_split)

        allowed_names = {"train", "val", "validation"}
        forbidden = [split.name for split in requested if split.name.lower() not in allowed_names]
        if forbidden:
            raise ValueError(
                "threshold fitting is restricted to train/validation splits; "
                f"received {forbidden}"
            )
        if train_split.name.lower() != "train":
            raise ValueError("the first split passed to fit() must be the train split")
        if any(split.channels != train_split.channels for split in requested[1:]):
            raise ValueError("train and validation channels do not match")

        score_chunks: list[np.ndarray] = []
        for split in requested:
            for _, trajectory, _, _ in split.iter_trajectories():
                matrices = trajectory.reshape(
                    len(trajectory), len(split.channels), len(STATE_FEATURES)
                )
                score_chunks.append((-matrices[:, :, 0]).reshape(-1))
        if not score_chunks:
            raise ValueError("cannot fit a threshold from empty splits")
        observed_scores = np.concatenate(score_chunks)
        observed_scores = observed_scores[np.isfinite(observed_scores)]
        if not len(observed_scores):
            raise ValueError("no finite quality scores in fitting splits")

        threshold = float(np.quantile(observed_scores, float(quantile)))
        return cls(
            threshold=threshold,
            n_actions=len(train_split.channels),
            quantile=float(quantile),
            fitted_splits=tuple(split.name for split in requested),
        )

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._previous_action = None

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del info
        scores = quality_scores(observation, n_channels=self.n_actions)
        flat_observation = np.asarray(observation).reshape(-1)
        has_action_block = flat_observation.size >= (
            self.n_actions * len(STATE_FEATURES) + self.n_actions
        )
        encoded_previous = previous_action_from_observation(
            observation, n_channels=self.n_actions
        )
        previous = (
            encoded_previous
            if encoded_previous is not None
            else (None if has_action_block else self._previous_action)
        )
        if previous is not None and scores[previous] >= self.threshold:
            action = int(previous)
        else:
            action = int(np.argmax(scores))
        self._previous_action = action
        return action


ThresholdHysteresisPolicy = ThresholdPolicy


class ScheduleAwareSweepPolicy:
    """Collision-free rule for the protocol's public fixed sweeping schedule.

    It reads neither the RF observation nor ``info``.  The public schedule
    starts at ``phase_index`` and advances one channel per step.  The policy
    holds a safe channel until that channel is the current sweep target, then
    moves to the channel swept on the immediately preceding step.
    """

    name = "Schedule-Aware Sweep"
    supported_jammer_modes = frozenset({"sweeping"})

    def __init__(
        self, n_actions: int = len(CHANNELS), *, phase_index: int = 0
    ) -> None:
        if int(n_actions) < 2:
            raise ValueError("ScheduleAwareSweepPolicy needs at least two actions")
        if not 0 <= int(phase_index) < int(n_actions):
            raise ValueError("phase_index is outside the action space")
        self.n_actions = int(n_actions)
        self.phase_index = int(phase_index)
        self._step_index = 0
        self._previous_action: int | None = None

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._step_index = 0
        self._previous_action = None

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del observation, info
        current_jammer = (self.phase_index + self._step_index) % self.n_actions
        if self._previous_action is None or self._previous_action == current_jammer:
            action = (current_jammer - 1) % self.n_actions
        else:
            action = self._previous_action
        self._previous_action = int(action)
        self._step_index += 1
        return int(action)


class JammerAwareGreedyPolicy:
    """Greedy reference privileged with only the current jammer label.

    This is the policy called ``Oracle`` in the legacy code.  It is not a
    reward upper bound: it has no access to future jammer states.
    """

    name = "Jammer-Aware Greedy Reference"

    def __init__(self, channels: Sequence[int] = CHANNELS) -> None:
        self.channels = tuple(int(channel) for channel in channels)
        if len(self.channels) < 2:
            raise ValueError("JammerAwareGreedyPolicy needs at least two channels")
        self.n_actions = len(self.channels)
        self._previous_action: int | None = None

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._previous_action = None

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        if info is None or "jammer_channel" not in info:
            raise ValueError(
                "JammerAwareGreedyPolicy requires info['jammer_channel']"
            )
        jammer_channel = int(info["jammer_channel"])
        if jammer_channel not in self.channels:
            raise ValueError(f"unknown jammer channel {jammer_channel}")
        jammed_action = self.channels.index(jammer_channel)

        flat_observation = np.asarray(observation).reshape(-1)
        has_action_block = flat_observation.size >= (
            self.n_actions * len(STATE_FEATURES) + self.n_actions
        )
        encoded_previous = previous_action_from_observation(
            observation, n_channels=self.n_actions
        )
        previous = (
            encoded_previous
            if encoded_previous is not None
            else (None if has_action_block else self._previous_action)
        )
        if previous is not None and previous != jammed_action:
            action = int(previous)
        else:
            scores = quality_scores(observation, n_channels=self.n_actions)
            scores[jammed_action] = -np.inf
            action = int(np.argmax(scores))
        self._previous_action = action
        return action


class ClairvoyantOraclePolicy:
    """Full-horizon dynamic-programming upper bound for one jammer sequence.

    ``prepare_episode`` must be called with the complete action-independent
    jammer sequence before evaluation.  The resulting action plan maximises
    exactly the environment reward, including the no-cost first action and the
    rule that a collision receives ``reward_collision`` without an additional
    switching penalty.  ``act`` then reads neither observations nor ``info``.
    """

    name = "Clairvoyant DP Oracle"

    def __init__(
        self,
        channels: Sequence[int] = CHANNELS,
        *,
        switch_cost: float = 0.1,
        reward_safe: float = 1.0,
        reward_collision: float = -1.0,
    ) -> None:
        self.channels = tuple(int(channel) for channel in channels)
        if len(self.channels) < 2 or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must contain at least two unique values")
        if float(switch_cost) < 0:
            raise ValueError("switch_cost must be non-negative")
        self.n_actions = len(self.channels)
        self.switch_cost = float(switch_cost)
        self.reward_safe = float(reward_safe)
        self.reward_collision = float(reward_collision)
        self._planned_actions: tuple[int, ...] = ()
        self._planned_return: float | None = None
        self._cursor = 0

    def _jammer_index(self, channel: int) -> int:
        channel = int(channel)
        if channel in self.channels:
            return self.channels.index(channel)
        if 0 <= channel < self.n_actions:
            return channel
        raise ValueError(f"unknown jammer channel/index {channel}")

    def prepare_episode(self, jammer_sequence: Sequence[int]) -> None:
        jammer = np.asarray(
            [self._jammer_index(channel) for channel in jammer_sequence],
            dtype=np.int64,
        )
        if jammer.size == 0:
            raise ValueError("jammer_sequence must not be empty")
        horizon = int(jammer.size)
        scores = np.full((horizon, self.n_actions), -np.inf, dtype=np.float64)
        backpointers = np.full(
            (horizon, self.n_actions), -1, dtype=np.int64
        )

        for action in range(self.n_actions):
            scores[0, action] = (
                self.reward_collision
                if action == jammer[0]
                else self.reward_safe
            )

        previous_actions = np.arange(self.n_actions)
        for step in range(1, horizon):
            for action in range(self.n_actions):
                if action == jammer[step]:
                    transition_reward = np.full(
                        self.n_actions, self.reward_collision, dtype=np.float64
                    )
                else:
                    transition_reward = np.full(
                        self.n_actions, self.reward_safe, dtype=np.float64
                    )
                    transition_reward[previous_actions != action] -= self.switch_cost
                candidates = scores[step - 1] + transition_reward
                best_previous = int(np.argmax(candidates))
                scores[step, action] = candidates[best_previous]
                backpointers[step, action] = best_previous

        final_action = int(np.argmax(scores[-1]))
        actions = np.empty(horizon, dtype=np.int64)
        actions[-1] = final_action
        for step in range(horizon - 1, 0, -1):
            actions[step - 1] = backpointers[step, actions[step]]
        self._planned_actions = tuple(int(action) for action in actions)
        self._planned_return = float(scores[-1, final_action])
        self._cursor = 0

    @property
    def planned_actions(self) -> tuple[int, ...]:
        return self._planned_actions

    @property
    def planned_return(self) -> float | None:
        return self._planned_return

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._cursor = 0

    def act(
        self,
        observation: np.ndarray | Sequence[float],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del observation, info
        if not self._planned_actions:
            raise RuntimeError("prepare_episode() must be called before act()")
        if self._cursor >= len(self._planned_actions):
            raise RuntimeError("the prepared episode plan is exhausted")
        action = self._planned_actions[self._cursor]
        self._cursor += 1
        return int(action)


# Import compatibility only.  Reports and runner outputs use
# ``jammer_greedy`` / ``Jammer-Aware Greedy Reference`` and never call this an
# oracle.  New code should use :class:`JammerAwareGreedyPolicy` explicitly.
OraclePolicy = JammerAwareGreedyPolicy


def build_baselines(
    train_split: ScanSplit,
    validation_split: ScanSplit | None = None,
    *,
    seed: int = 0,
    threshold_quantile: float = 0.5,
    switch_cost: float = 0.1,
    reward_safe: float = 1.0,
    reward_collision: float = -1.0,
) -> dict[str, object]:
    """Create the complete declared baseline suite.

    Passing train/validation splits here fits the threshold exactly once.  The
    returned object can then be reused unchanged on within-distribution and OOD
    test environments.
    """

    n_actions = len(train_split.channels)
    return {
        "random": RandomPolicy(n_actions, seed=seed),
        "stay": StayPolicy(n_actions),
        "max_quality": MinInterferencePolicy(n_actions),
        "threshold": ThresholdPolicy.fit(
            train_split,
            validation_split,
            quantile=threshold_quantile,
        ),
        "schedule_sweep": ScheduleAwareSweepPolicy(n_actions),
        "jammer_greedy": JammerAwareGreedyPolicy(train_split.channels),
        "clairvoyant_oracle": ClairvoyantOraclePolicy(
            train_split.channels,
            switch_cost=switch_cost,
            reward_safe=reward_safe,
            reward_collision=reward_collision,
        ),
    }


get_baselines = build_baselines


__all__ = [
    "SNR_SCORE_DIRECTION",
    "quality_scores",
    "previous_action_from_observation",
    "RandomPolicy",
    "StayPolicy",
    "MinInterferencePolicy",
    "MaxQualityPolicy",
    "ThresholdPolicy",
    "ThresholdHysteresisPolicy",
    "ScheduleAwareSweepPolicy",
    "JammerAwareGreedyPolicy",
    "ClairvoyantOraclePolicy",
    "OraclePolicy",
    "build_baselines",
    "get_baselines",
]
