"""Reproducible Double-DQN agent and train-only RF normalisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

try:  # Provide a clear error when someone runs data-only checks without torch.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - torch is a declared dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

from .config import ExperimentConfig

if TYPE_CHECKING:
    from .data import ScanSplit


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for learned-model experiments; install the "
            "wheel listed in requirements.txt"
        ) from _TORCH_IMPORT_ERROR


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without sharing train/eval streams."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class RFNormalizer:
    """Z-score the 40 RF entries while preserving the action one-hot block."""

    mean: np.ndarray
    scale: np.ndarray
    rf_dim: int = 40
    fit_split: str = "train"
    fit_scan_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        if mean.shape != (int(self.rf_dim),) or scale.shape != (int(self.rf_dim),):
            raise ValueError("normalizer mean/scale have the wrong dimension")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("normalizer statistics must be finite")
        if np.any(scale <= 0):
            raise ValueError("normalizer scale must be positive")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "fit_scan_ids", tuple(int(x) for x in self.fit_scan_ids))

    @classmethod
    def fit_array(
        cls,
        observations: np.ndarray,
        *,
        rf_dim: int = 40,
        fit_split: str = "train",
        fit_scan_ids: tuple[int, ...] = (),
        minimum_scale: float = 1e-6,
    ) -> "RFNormalizer":
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < rf_dim:
            raise ValueError("observations must be a non-empty [n, rf_dim+] array")
        rf = values[:, :rf_dim]
        if not np.isfinite(rf).all():
            raise ValueError("training observations contain non-finite values")
        mean = rf.mean(axis=0)
        scale = rf.std(axis=0, ddof=0)
        scale = np.where(scale < float(minimum_scale), 1.0, scale)
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            rf_dim=int(rf_dim),
            fit_split=str(fit_split),
            fit_scan_ids=fit_scan_ids,
        )

    @classmethod
    def fit_training_split(
        cls, split: "ScanSplit", *, rf_dim: int = 40
    ) -> "RFNormalizer":
        """Fit only a split explicitly named ``train`` (leakage guard)."""

        if split.name.lower() != "train":
            raise ValueError(
                f"normalizer fitting is restricted to train; received {split.name!r}"
            )
        return cls.fit_array(
            split.observations,
            rf_dim=rf_dim,
            fit_split=split.name,
            fit_scan_ids=split.scan_ids,
        )

    def transform(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape[-1] < self.rf_dim:
            raise ValueError("observation is shorter than rf_dim")
        result = values.copy()
        result[..., : self.rf_dim] = (
            result[..., : self.rf_dim] - self.mean
        ) / self.scale
        # Deliberately do not touch result[..., rf_dim:] (the previous-action
        # one-hot vector), including exact all-zero episode-start states.
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "rf_dim": self.rf_dim,
            "fit_split": self.fit_split,
            "fit_scan_ids": list(self.fit_scan_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RFNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            rf_dim=int(payload["rf_dim"]),
            fit_split=str(payload.get("fit_split", "train")),
            fit_scan_ids=tuple(int(x) for x in payload.get("fit_scan_ids", ())),
        )


class ReplayBuffer:
    """Fixed-capacity, agent-local replay memory with a seeded sampler."""

    def __init__(self, capacity: int, state_dim: int, *, seed: int) -> None:
        if int(capacity) <= 0 or int(state_dim) <= 0:
            raise ValueError("capacity and state_dim must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self._rng = np.random.default_rng(int(seed))
        self._size = 0
        self._position = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self._position
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = float(bool(done))
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: "torch.device") -> tuple[Any, ...]:
        require_torch()
        if int(batch_size) > self._size:
            raise ValueError("cannot sample more transitions than are stored")
        indices = self._rng.choice(self._size, size=int(batch_size), replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


def resolve_device(requested: str = "auto") -> "torch.device":
    require_torch()
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


class DDQNAgent:
    """Double-DQN with replay, episode-wise exploration and soft targets."""

    def __init__(
        self,
        model_name: str,
        config: ExperimentConfig,
        normalizer: RFNormalizer,
        *,
        seed: int,
        device: str | None = None,
    ) -> None:
        require_torch()
        from .models import build_model

        self.model_name = str(model_name).lower()
        self.config = config
        self.normalizer = normalizer
        self.seed = int(seed)
        seed_everything(self.seed, deterministic_torch=config.deterministic_torch)
        self.device = resolve_device(config.device if device is None else device)
        def make_model():
            # ``matched`` is the public protocol name; accept the more explicit
            # factory alias used by early model-module drafts.
            candidates = (
                (self.model_name, "matched_mlp")
                if self.model_name == "matched"
                else (self.model_name,)
            )
            last_error: Exception | None = None
            for candidate in candidates:
                try:
                    try:
                        return build_model(
                            candidate,
                            input_dim=config.input_dim,
                            n_actions=config.n_actions,
                        )
                    except TypeError as signature_error:
                        if not all(
                            token in str(signature_error)
                            for token in ("input_dim",)
                        ):
                            raise
                        return build_model(
                            candidate,
                            observation_dim=config.input_dim,
                            action_dim=config.n_actions,
                        )
                except (KeyError, ValueError) as exc:
                    last_error = exc
            assert last_error is not None
            raise last_error

        self.online = make_model().to(self.device)
        self.target = make_model().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay = ReplayBuffer(
            config.replay_capacity, config.input_dim, seed=self.seed + 17
        )
        self._action_rng = np.random.default_rng(self.seed + 31)
        self.environment_steps = 0
        self.update_steps = 0
        self.current_episode = 0
        self.training_jammer_mode: str | None = None
        self.last_loss: float | None = None

    @property
    def epsilon(self) -> float:
        # Freeze the schedule used by the submitted implementation:
        # epsilon_e = max(0.01, 1 / (1 + e / 10)).
        reciprocal = 1.0 / (
            1.0
            + float(self.current_episode)
            / float(self.config.epsilon_denominator_episodes)
        )
        return float(max(self.config.epsilon_end, reciprocal))

    def set_episode(self, episode_index: int) -> None:
        if int(episode_index) < 0:
            raise ValueError("episode_index must be non-negative")
        self.current_episode = int(episode_index)

    def normalise(self, observation: np.ndarray) -> np.ndarray:
        return self.normalizer.transform(observation)

    def act(
        self,
        observation: np.ndarray,
        *,
        explore: bool = False,
        already_normalized: bool = False,
    ) -> int:
        if explore and self._action_rng.random() < self.epsilon:
            return int(self._action_rng.integers(self.config.n_actions))
        state = (
            np.asarray(observation, dtype=np.float32)
            if already_normalized
            else self.normalise(observation)
        )
        was_training = self.online.training
        self.online.eval()
        try:
            with torch.inference_mode():
                tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
                q_values = self.online(tensor)
                return int(q_values.argmax(dim=1).item())
        finally:
            if was_training:
                self.online.train()

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(
            self.normalise(observation),
            action,
            reward,
            self.normalise(next_observation),
            done,
        )
        self.environment_steps += 1

    def _soft_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)
            target_buffers = dict(self.target.named_buffers())
            for name, online_buffer in self.online.named_buffers():
                target_buffer = target_buffers[name]
                if torch.is_floating_point(target_buffer):
                    target_buffer.mul_(1.0 - tau).add_(online_buffer, alpha=tau)
                else:
                    target_buffer.copy_(online_buffer)

    def learn(self) -> float | None:
        cfg = self.config
        if self.environment_steps < cfg.learning_starts:
            return None
        if self.environment_steps % cfg.train_frequency != 0:
            return None
        if len(self.replay) < cfg.batch_size:
            return None

        losses: list[float] = []
        self.online.train()
        for _ in range(cfg.gradient_steps):
            states, actions, rewards, next_states, dones = self.replay.sample(
                cfg.batch_size, self.device
            )
            # Action selection is inference, not a second BatchNorm training
            # pass.  Use frozen running statistics so one optimizer update
            # increments each BN counter exactly once (on ``states`` below).
            self.online.eval()
            with torch.no_grad():
                # Double-DQN: online network selects; target network evaluates.
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
                targets = rewards + cfg.gamma * (1.0 - dones) * next_q
            self.online.train()
            predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
            loss = F.mse_loss(predicted, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.online.parameters(), cfg.gradient_clip_norm
            )
            self.optimizer.step()
            self._soft_update()
            self.update_steps += 1
            losses.append(float(loss.detach().cpu()))
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "current_episode": self.current_episode,
            "training_jammer_mode": self.training_jammer_mode,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)
        return path

    @classmethod
    def load(
        cls, path: str | Path, *, device: str = "auto"
    ) -> "DDQNAgent":
        require_torch()
        load_device = resolve_device(device)
        payload = torch.load(Path(path), map_location=load_device, weights_only=False)
        config = ExperimentConfig(**payload["config"])
        normalizer = RFNormalizer.from_dict(payload["normalizer"])
        agent = cls(
            payload["model_name"],
            config,
            normalizer,
            seed=int(payload["seed"]),
            device=str(load_device),
        )
        agent.online.load_state_dict(payload["online_state_dict"])
        agent.target.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.environment_steps = int(payload.get("environment_steps", 0))
        agent.update_steps = int(payload.get("update_steps", 0))
        agent.current_episode = int(payload.get("current_episode", 0))
        agent.training_jammer_mode = payload.get("training_jammer_mode")
        return agent

    def metadata_json(self) -> str:
        """Small audit payload; model weights remain in the checkpoint."""

        return json.dumps(
            {
                "model_name": self.model_name,
                "seed": self.seed,
                "device": str(self.device),
                "environment_steps": self.environment_steps,
                "update_steps": self.update_steps,
                "normalizer": self.normalizer.to_dict(),
                "config": self.config.to_dict(),
            },
            sort_keys=True,
        )


__all__ = [
    "seed_everything",
    "require_torch",
    "RFNormalizer",
    "ReplayBuffer",
    "resolve_device",
    "DDQNAgent",
]
