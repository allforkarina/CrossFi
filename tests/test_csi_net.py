import pytest
import torch

from models.csi_net import CSINet, prepare_csi_net_input


def test_prepare_csi_net_input_shape_and_nonfinite_fill() -> None:
    csi_amplitude = torch.arange(2 * 3 * 114 * 10, dtype=torch.float32).reshape(2, 3, 114, 10)
    csi_phase_cos = torch.ones_like(csi_amplitude)
    csi_amplitude[0, 0, 1, 0] = float("nan")
    csi_phase_cos[1, 2, 10, 3] = float("inf")

    prepared = prepare_csi_net_input(csi_amplitude, csi_phase_cos)

    assert prepared.shape == (2, 2, 10, 342)
    assert torch.isfinite(prepared).all()


def test_csi_net_forward_shape_and_range() -> None:
    pytest.importorskip("torchvision")
    model = CSINet(feature_dim=32, projection_dim=16, num_heads=4, pretrained=False)
    query = torch.randn(2, 2, 10, 342)
    key = torch.randn(3, 2, 10, 342)

    with torch.no_grad():
        scores = model(query, key)

    assert scores.shape == (2, 3)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)


def test_csi_net_rejects_invalid_head_layout() -> None:
    pytest.importorskip("torchvision")
    with pytest.raises(ValueError, match="projection_dim must be divisible"):
        CSINet(projection_dim=10, num_heads=4, pretrained=False)
