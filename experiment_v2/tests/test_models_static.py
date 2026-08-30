"""Static architecture checks for the frozen confirmatory Q-networks."""

from __future__ import annotations

import pytest

from src import models


pytestmark = pytest.mark.skipif(
    not models.TORCH_AVAILABLE, reason="optional PyTorch dependency is unavailable"
)


def test_hnp_forward_shape_and_finite_values() -> None:
    torch = pytest.importorskip("torch")
    network = models.HNPQNetwork().eval()
    observations = torch.randn(7, 48)
    with torch.no_grad():
        q_values = network(observations)
    assert q_values.shape == (7, 8)
    assert torch.isfinite(q_values).all()


def test_polynomial_expansion_is_h_and_h_squared_before_layer_norm() -> None:
    torch = pytest.importorskip("torch")
    layer = models.PolynomialExpansion(hidden_dim=64).eval()
    hidden = torch.randn(5, 64)
    expected = layer.layer_norm(torch.cat((hidden, hidden.square()), dim=-1))
    with torch.no_grad():
        actual = layer(hidden)
    assert actual.shape == (5, 128)
    assert torch.allclose(actual, expected)


def test_hnp_has_frozen_layer_dimensions_and_dueling_head() -> None:
    torch_nn = pytest.importorskip("torch.nn")
    network = models.HNPQNetwork()
    linear_layers = [
        module for module in network.modules() if isinstance(module, torch_nn.Linear)
    ]
    batch_norms = [
        module
        for module in network.modules()
        if isinstance(module, torch_nn.BatchNorm1d)
    ]
    layer_norms = [
        module for module in network.modules() if isinstance(module, torch_nn.LayerNorm)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (48, 128),
        (128, 64),
        (128, 128),
        (128, 1),
        (128, 8),
    ]
    assert [layer.num_features for layer in batch_norms] == [128, 64]
    assert [tuple(layer.normalized_shape) for layer in layer_norms] == [(128,)]


def test_capacity_matched_mlp_forward_and_declared_dimensions() -> None:
    torch = pytest.importorskip("torch")
    torch_nn = pytest.importorskip("torch.nn")
    network = models.CapacityMatchedMLPQNetwork().eval()
    with torch.no_grad():
        assert network(torch.randn(3, 48)).shape == (3, 8)
    linear_layers = [
        module for module in network.modules() if isinstance(module, torch_nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (48, 174),
        (174, 128),
        (128, 1),
        (128, 8),
    ]


def test_capacity_match_is_below_five_percent() -> None:
    hnp = models.HNPQNetwork()
    matched = models.CapacityMatchedMLPQNetwork()
    assert models.count_trainable_parameters(hnp) == 32_841
    assert models.count_trainable_parameters(matched) == 32_691
    assert models.relative_parameter_difference(hnp, matched) < 0.05
    assert models.relative_parameter_difference(hnp, matched) == pytest.approx(
        150 / 32_841
    )


def test_vanilla_ddqn_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    network = models.VanillaQNetwork().eval()
    with torch.no_grad():
        assert network(torch.randn(2, 48)).shape == (2, 8)


def test_registry_rejects_unknown_network() -> None:
    with pytest.raises(ValueError, match="unknown network"):
        models.build_q_network("not-a-real-model")


def test_runner_registry_compatibility_aliases() -> None:
    assert isinstance(models.build_model("hnp"), models.HNPQNetwork)
    assert isinstance(models.build_model("matched"), models.CapacityMatchedMLPQNetwork)


@pytest.mark.parametrize(
    ("name", "model_type", "expected_parameters"),
    [
        ("no_polynomial", models.HNPNoPolynomialQNetwork, 24_521),
        ("no_layernorm", models.HNPNoLayerNormQNetwork, 32_585),
        ("no_dueling", models.HNPNoDuelingQNetwork, 32_712),
    ],
)
def test_single_component_ablation_forward_and_parameter_count(
    name: str, model_type: type, expected_parameters: int
) -> None:
    torch = pytest.importorskip("torch")
    network = models.build_model(name).eval()
    assert isinstance(network, model_type)
    with torch.no_grad():
        output = network(torch.randn(4, 48))
    assert output.shape == (4, 8)
    assert torch.isfinite(output).all()
    assert models.count_trainable_parameters(network) == expected_parameters


def test_no_polynomial_retains_layernorm_without_squared_features() -> None:
    torch_nn = pytest.importorskip("torch.nn")
    network = models.HNPNoPolynomialQNetwork()
    assert isinstance(network.layer_norm, torch_nn.LayerNorm)
    assert tuple(network.layer_norm.normalized_shape) == (64,)
    assert network.projection[0].in_features == 64


def test_no_layernorm_retains_expansion_and_uses_identity() -> None:
    torch_nn = pytest.importorskip("torch.nn")
    network = models.HNPNoLayerNormQNetwork()
    assert isinstance(network.layer_norm, torch_nn.Identity)
    assert network.projection[0].in_features == 128


def test_no_dueling_uses_one_direct_action_head() -> None:
    torch_nn = pytest.importorskip("torch.nn")
    network = models.HNPNoDuelingQNetwork()
    assert isinstance(network.q_head, torch_nn.Linear)
    assert (network.q_head.in_features, network.q_head.out_features) == (128, 8)
    assert not any(isinstance(module, models.DuelingHead) for module in network.modules())
