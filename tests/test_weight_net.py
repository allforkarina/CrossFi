import pytest
import torch

from models.weight_net import WeightNet


def test_weight_net_forward_shape_and_range() -> None:
    pytest.importorskip("torchvision")
    model = WeightNet(num_samples=5, pretrained=False)
    similarity_matrix = torch.randn(2, 5, 5)

    with torch.no_grad():
        weights = model(similarity_matrix)

    assert weights.shape == (2, 5)
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1)


def test_weight_net_rejects_wrong_matrix_shape() -> None:
    pytest.importorskip("torchvision")
    model = WeightNet(num_samples=5, pretrained=False)

    with pytest.raises(ValueError, match="b x k x k"):
        model(torch.randn(5, 5))

    with pytest.raises(ValueError, match="b x 5 x 5"):
        model(torch.randn(2, 5, 4))


def test_weight_net_rejects_invalid_num_samples() -> None:
    pytest.importorskip("torchvision")

    with pytest.raises(ValueError, match="num_samples must be positive"):
        WeightNet(num_samples=0, pretrained=False)
