from __future__ import annotations

import torch
from torch import Tensor, nn


class WeightNet(nn.Module):
    """Adaptive template sample quality estimator."""

    def __init__(self, num_samples: int, pretrained: bool = True) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.num_samples = num_samples
        self.backbone = _build_resnet18_feature_extractor(pretrained=pretrained)
        self.classifier = nn.Linear(512, num_samples)

    def forward(self, similarity_matrix: Tensor) -> Tensor:
        """Return b x k quality confidence scores from b x k x k similarities."""

        _validate_similarity_matrix(similarity_matrix, self.num_samples)
        features = self.backbone(similarity_matrix.unsqueeze(1))
        return torch.sigmoid(self.classifier(features))


def _build_resnet18_feature_extractor(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Install torchvision to use WeightNet: python -m pip install torchvision") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    with torch.no_grad():
        averaged_weight = original_conv.weight.mean(dim=1, keepdim=True)
        model.conv1.weight.copy_(averaged_weight)
        if original_conv.bias is not None and model.conv1.bias is not None:
            model.conv1.bias.copy_(original_conv.bias)

    model.fc = nn.Identity()
    return model


def _validate_similarity_matrix(similarity_matrix: Tensor, num_samples: int) -> None:
    if not torch.is_floating_point(similarity_matrix):
        raise TypeError("similarity_matrix must be a floating point tensor")
    if similarity_matrix.ndim != 3:
        raise ValueError(f"similarity_matrix must have shape b x k x k, got {tuple(similarity_matrix.shape)}")
    if tuple(similarity_matrix.shape[1:]) != (num_samples, num_samples):
        raise ValueError(
            "similarity_matrix must have shape b x "
            f"{num_samples} x {num_samples}, got {tuple(similarity_matrix.shape)}"
        )
