"""Q-network definitions for the leakage-safe anti-jamming experiments.

The module intentionally remains importable when PyTorch is unavailable so
that data preparation, heuristic baselines, and statistical analysis do not
acquire a hard deep-learning dependency.  Constructing a neural network (or
calling a model utility) then raises a concise installation error.

Architectures are frozen before confirmatory evaluation:

* HNP-DQN: ``48 -> 128(BN) -> 64(BN) -> [h, h^2] -> LN -> 128`` followed
  by a dueling value/advantage head.
* Capacity-matched MLP-DDQN: ``48 -> 174(BN) -> 128(BN)`` followed by the
  same dueling head.  With eight actions it has 32,691 trainable parameters,
  versus 32,841 for HNP-DQN (a 0.46% difference).
* Vanilla DDQN: a conventional ``48 -> 128 -> 64 -> 8`` MLP.  Double-DQN is
  a target-construction rule implemented by the agent, not by the network.
"""

from __future__ import annotations

from typing import Any, Final


_TORCH_IMPORT_ERROR: BaseException | None = None
try:  # Keep heuristic-only workflows usable without PyTorch.
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE: Final[bool] = True
except (ImportError, OSError) as exc:  # OSError also covers broken native DLLs.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


DEFAULT_OBSERVATION_DIM: Final[int] = 48
DEFAULT_ACTION_DIM: Final[int] = 8


def require_torch() -> None:
    """Raise an actionable error if the optional PyTorch dependency is absent."""

    if not TORCH_AVAILABLE:
        detail = (
            f" Original import error: {_TORCH_IMPORT_ERROR}"
            if _TORCH_IMPORT_ERROR is not None
            else ""
        )
        raise ImportError(
            "PyTorch is required for learned-agent training and inference. "
            "Install a compatible build from https://pytorch.org/get-started/."
            + detail
        ) from _TORCH_IMPORT_ERROR


if TORCH_AVAILABLE:

    class PolynomialExpansion(nn.Module):
        """Concatenate element-wise first- and second-order hidden features.

        This is deliberately ``[h, h^2]`` rather than an unconstrained or
        cross-feature polynomial basis.  LayerNorm is applied after the fixed
        expansion, exactly matching the preregistered HNP component.
        """

        def __init__(self, hidden_dim: int = 64) -> None:
            super().__init__()
            if int(hidden_dim) <= 0:
                raise ValueError("hidden_dim must be positive")
            self.hidden_dim = int(hidden_dim)
            self.output_dim = 2 * self.hidden_dim
            self.layer_norm = nn.LayerNorm(self.output_dim)

        def forward(self, hidden: Tensor) -> Tensor:
            if hidden.ndim < 2 or hidden.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "PolynomialExpansion expected the last dimension to be "
                    f"{self.hidden_dim}, received shape {tuple(hidden.shape)}"
                )
            expanded = torch.cat((hidden, hidden.square()), dim=-1)
            return self.layer_norm(expanded)


    class DuelingHead(nn.Module):
        """Canonical dueling aggregation ``Q = V + A - mean(A)``."""

        def __init__(self, feature_dim: int, action_dim: int) -> None:
            super().__init__()
            if int(feature_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("feature_dim and action_dim must be positive")
            self.feature_dim = int(feature_dim)
            self.action_dim = int(action_dim)
            self.value = nn.Linear(self.feature_dim, 1)
            self.advantage = nn.Linear(self.feature_dim, self.action_dim)

        def forward(self, features: Tensor) -> Tensor:
            value = self.value(features)
            advantage = self.advantage(features)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)


    class HNPQNetwork(nn.Module):
        """Frozen HNP-DQN Q-network used with Double-DQN target updates."""

        architecture_name = "hnp_dqn"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)

            self.feature_extractor = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.polynomial = PolynomialExpansion(64)
            self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            self.dueling = DuelingHead(128, self.action_dim)

        def forward(self, observation: Tensor) -> Tensor:
            hidden = self.feature_extractor(observation)
            expanded = self.polynomial(hidden)
            return self.dueling(self.projection(expanded))


    class CapacityMatchedMLPQNetwork(nn.Module):
        """Conventional MLP dueling network capacity-matched to HNP-DQN.

        The width 174 was selected from parameter counts before evaluation.
        It preserves BatchNorm and the dueling head, so the primary comparison
        controls parameter count and head type.  Its hidden widths and lack of
        post-expansion LayerNorm still differ from HNP-DQN; therefore it is a
        capacity control, not a single-factor causal isolation of squaring.
        """

        architecture_name = "capacity_matched_mlp_ddqn"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)

            self.feature_extractor = nn.Sequential(
                nn.Linear(self.observation_dim, 174),
                nn.BatchNorm1d(174),
                nn.ReLU(),
                nn.Linear(174, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.dueling = DuelingHead(128, self.action_dim)

        def forward(self, observation: Tensor) -> Tensor:
            return self.dueling(self.feature_extractor(observation))


    class HNPNoPolynomialQNetwork(nn.Module):
        """HNP ablation removing only the explicit ``[h, h^2]`` expansion.

        LayerNorm, the 128-wide projection, and the dueling head are retained.
        Their input dimensions necessarily follow the unexpanded 64-vector;
        no hidden width is enlarged to disguise the resulting capacity change.
        """

        architecture_name = "hnp_no_polynomial"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.layer_norm = nn.LayerNorm(64)
            self.projection = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
            self.dueling = DuelingHead(128, self.action_dim)

        def forward(self, observation: Tensor) -> Tensor:
            hidden = self.feature_extractor(observation)
            return self.dueling(self.projection(self.layer_norm(hidden)))


    class HNPNoLayerNormQNetwork(nn.Module):
        """HNP ablation retaining ``[h, h^2]`` but removing LayerNorm only."""

        architecture_name = "hnp_no_layernorm"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.layer_norm = nn.Identity()
            self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            self.dueling = DuelingHead(128, self.action_dim)

        def forward(self, observation: Tensor) -> Tensor:
            hidden = self.feature_extractor(observation)
            expanded = torch.cat((hidden, hidden.square()), dim=-1)
            return self.dueling(self.projection(self.layer_norm(expanded)))


    class HNPNoDuelingQNetwork(nn.Module):
        """HNP ablation replacing only the dueling head by ``Linear(128, A)``."""

        architecture_name = "hnp_no_dueling"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.polynomial = PolynomialExpansion(64)
            self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            self.q_head = nn.Linear(128, self.action_dim)

        def forward(self, observation: Tensor) -> Tensor:
            hidden = self.feature_extractor(observation)
            expanded = self.polynomial(hidden)
            return self.q_head(self.projection(expanded))


    class VanillaQNetwork(nn.Module):
        """Conventional non-dueling MLP used by the secondary DDQN reference."""

        architecture_name = "vanilla_ddqn"

        def __init__(
            self,
            observation_dim: int = DEFAULT_OBSERVATION_DIM,
            action_dim: int = DEFAULT_ACTION_DIM,
        ) -> None:
            super().__init__()
            if int(observation_dim) <= 0 or int(action_dim) <= 0:
                raise ValueError("observation_dim and action_dim must be positive")
            self.observation_dim = int(observation_dim)
            self.action_dim = int(action_dim)
            self.network = nn.Sequential(
                nn.Linear(self.observation_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_dim),
            )

        def forward(self, observation: Tensor) -> Tensor:
            return self.network(observation)


else:

    class _TorchRequired:
        """Import-safe placeholder used when torch cannot be imported."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_torch()


    PolynomialExpansion = _TorchRequired
    DuelingHead = _TorchRequired
    HNPQNetwork = _TorchRequired
    CapacityMatchedMLPQNetwork = _TorchRequired
    HNPNoPolynomialQNetwork = _TorchRequired
    HNPNoLayerNormQNetwork = _TorchRequired
    HNPNoDuelingQNetwork = _TorchRequired
    VanillaQNetwork = _TorchRequired


# Reader- and legacy-friendly aliases.  The algorithm remains Double-DQN in
# the agent update; these classes define only Q-function parameterizations.
HNPDQNNetwork = HNPQNetwork
HNP_DQN_Network = HNPQNetwork
CapacityMatchedMLPDDQN = CapacityMatchedMLPQNetwork
MatchedMLPQNetwork = CapacityMatchedMLPQNetwork
VanillaDDQNNetwork = VanillaQNetwork
HNP_DQN_No_Polynomial = HNPNoPolynomialQNetwork
HNP_DQN_No_LayerNorm = HNPNoLayerNormQNetwork
HNP_DQN_No_Dueling = HNPNoDuelingQNetwork


def build_q_network(
    name: str,
    observation_dim: int = DEFAULT_OBSERVATION_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
):
    """Construct a registered network from a stable command-line name."""

    require_torch()
    normalised = str(name).strip().lower().replace("-", "_")
    factories = {
        "hnp": HNPQNetwork,
        "hnp_dqn": HNPQNetwork,
        "capacity_matched": CapacityMatchedMLPQNetwork,
        "capacity_matched_mlp": CapacityMatchedMLPQNetwork,
        "capacity_matched_mlp_ddqn": CapacityMatchedMLPQNetwork,
        "matched": CapacityMatchedMLPQNetwork,
        "matched_mlp": CapacityMatchedMLPQNetwork,
        "matched_mlp_ddqn": CapacityMatchedMLPQNetwork,
        "no_polynomial": HNPNoPolynomialQNetwork,
        "hnp_no_polynomial": HNPNoPolynomialQNetwork,
        "no_layernorm": HNPNoLayerNormQNetwork,
        "no_layer_norm": HNPNoLayerNormQNetwork,
        "hnp_no_layernorm": HNPNoLayerNormQNetwork,
        "hnp_no_layer_norm": HNPNoLayerNormQNetwork,
        "no_dueling": HNPNoDuelingQNetwork,
        "hnp_no_dueling": HNPNoDuelingQNetwork,
        "ddqn": VanillaQNetwork,
        "vanilla": VanillaQNetwork,
        "vanilla_ddqn": VanillaQNetwork,
    }
    try:
        factory = factories[normalised]
    except KeyError as exc:
        raise ValueError(
            f"unknown network {name!r}; choose one of {sorted(factories)}"
        ) from exc
    return factory(observation_dim=observation_dim, action_dim=action_dim)


def count_trainable_parameters(model: Any) -> int:
    """Return the exact number of scalar parameters optimized for ``model``."""

    require_torch()
    if not hasattr(model, "parameters"):
        raise TypeError("model must provide a parameters() iterator")
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# Compatibility name used by the unified training agent/runner.
build_model = build_q_network


def relative_parameter_difference(model_a: Any, model_b: Any) -> float:
    """Symmetric trainable-parameter difference as a fraction of the larger model."""

    count_a = count_trainable_parameters(model_a)
    count_b = count_trainable_parameters(model_b)
    denominator = max(count_a, count_b)
    return 0.0 if denominator == 0 else abs(count_a - count_b) / denominator


__all__ = [
    "TORCH_AVAILABLE",
    "DEFAULT_OBSERVATION_DIM",
    "DEFAULT_ACTION_DIM",
    "require_torch",
    "PolynomialExpansion",
    "DuelingHead",
    "HNPQNetwork",
    "HNPDQNNetwork",
    "HNP_DQN_Network",
    "CapacityMatchedMLPQNetwork",
    "CapacityMatchedMLPDDQN",
    "MatchedMLPQNetwork",
    "HNPNoPolynomialQNetwork",
    "HNPNoLayerNormQNetwork",
    "HNPNoDuelingQNetwork",
    "HNP_DQN_No_Polynomial",
    "HNP_DQN_No_LayerNorm",
    "HNP_DQN_No_Dueling",
    "VanillaQNetwork",
    "VanillaDDQNNetwork",
    "build_q_network",
    "build_model",
    "count_trainable_parameters",
    "relative_parameter_difference",
]
