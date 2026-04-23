from __future__ import annotations

import math

import torch
from torch import Tensor, nn


CSI_NET_INPUT_SHAPE = (2, 10, 342)
RAW_CSI_SHAPE = (3, 114, 10)


def prepare_csi_net_input(csi_amplitude: Tensor, csi_phase_cos: Tensor) -> Tensor:
    """Stack CSI amplitude and phase cosine into b x 2 x 10 x 342 tensors."""

    _validate_raw_csi_tensor(csi_amplitude, "csi_amplitude")
    _validate_raw_csi_tensor(csi_phase_cos, "csi_phase_cos")
    if csi_amplitude.shape != csi_phase_cos.shape:
        raise ValueError(
            "csi_amplitude and csi_phase_cos must have the same shape, "
            f"got {tuple(csi_amplitude.shape)} and {tuple(csi_phase_cos.shape)}"
        )

    stacked = torch.stack((csi_amplitude, csi_phase_cos), dim=1)
    stacked = stacked.permute(0, 1, 4, 2, 3).reshape(-1, *CSI_NET_INPUT_SHAPE)
    return _interpolate_nonfinite(stacked)


class CSINet(nn.Module):
    """Cross-domain Siamese CSI similarity network."""

    def __init__(
        self,
        feature_dim: int = 256,
        projection_dim: int = 128,
        num_heads: int = 4,
        temperature: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if projection_dim % num_heads != 0:
            raise ValueError("projection_dim must be divisible by num_heads")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.head_dim = projection_dim // num_heads

        self.backbone = _build_resnet18_feature_extractor(pretrained=pretrained)
        self.reducer = nn.Linear(512, feature_dim)
        self.query_mapper = nn.Linear(feature_dim, projection_dim)
        self.key_mapper = nn.Linear(feature_dim, projection_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))

    def encode(self, csi_input: Tensor) -> Tensor:
        """Encode prepared CSI tensors into shared low-dimensional features."""

        _validate_prepared_csi_tensor(csi_input, "csi_input")
        return self.reducer(self.backbone(csi_input))

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Return a b1 x b2 matrix of query-key similarity probabilities."""

        query_features = self.query_mapper(self.encode(query))
        key_features = self.key_mapper(self.encode(key))

        query_heads = query_features.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        key_heads = key_features.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.mean(dim=0)

        temperature = self.log_temperature.exp().clamp_min(1e-6)
        return torch.sigmoid(scores / temperature)


def _build_resnet18_feature_extractor(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Install torchvision to use CSINet: python -m pip install torchvision") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=2,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    with torch.no_grad():
        averaged_weight = original_conv.weight.mean(dim=1, keepdim=True)
        model.conv1.weight.copy_(averaged_weight.repeat(1, 2, 1, 1))
        if original_conv.bias is not None and model.conv1.bias is not None:
            model.conv1.bias.copy_(original_conv.bias)

    model.fc = nn.Identity()
    return model


def _validate_raw_csi_tensor(csi_tensor: Tensor, name: str) -> None:
    if not torch.is_floating_point(csi_tensor):
        raise TypeError(f"{name} must be a floating point tensor")
    if csi_tensor.ndim != 4 or tuple(csi_tensor.shape[1:]) != RAW_CSI_SHAPE:
        raise ValueError(f"{name} must have shape b x 3 x 114 x 10, got {tuple(csi_tensor.shape)}")


def _validate_prepared_csi_tensor(csi_tensor: Tensor, name: str) -> None:
    if not torch.is_floating_point(csi_tensor):
        raise TypeError(f"{name} must be a floating point tensor")
    if csi_tensor.ndim != 4 or tuple(csi_tensor.shape[1:]) != CSI_NET_INPUT_SHAPE:
        raise ValueError(f"{name} must have shape b x 2 x 10 x 342, got {tuple(csi_tensor.shape)}")


def _interpolate_nonfinite(values: Tensor) -> Tensor:
    """Linearly fill non-finite values along the flattened spatial-frequency axis."""

    if torch.isfinite(values).all():
        return values

    filled = values.clone().contiguous()
    flat = filled.view(-1, filled.shape[-1])

    row_indices = torch.nonzero(~torch.isfinite(flat).all(dim=1), as_tuple=False).flatten().tolist()
    for row_index in row_indices:
        row = flat[row_index]
        finite_mask = torch.isfinite(row)
        finite_positions = torch.nonzero(finite_mask, as_tuple=False).flatten()
        missing_positions = torch.nonzero(~finite_mask, as_tuple=False).flatten()

        if finite_positions.numel() == 0:
            row[missing_positions] = 0
            continue
        if finite_positions.numel() == 1:
            row[missing_positions] = row[finite_positions[0]]
            continue

        insert_positions = torch.searchsorted(finite_positions, missing_positions)
        right_indices = insert_positions.clamp(max=finite_positions.numel() - 1)
        left_indices = (right_indices - 1).clamp(min=0)

        left_positions = finite_positions[left_indices]
        right_positions = finite_positions[right_indices]
        left_values = row[left_positions]
        right_values = row[right_positions]

        span = (right_positions - left_positions).clamp(min=1).to(row.dtype)
        weight = (missing_positions - left_positions).to(row.dtype) / span
        row[missing_positions] = left_values + weight * (right_values - left_values)

    return filled
