from __future__ import annotations

import torch
from torch import nn


class FutureProbabilityMLP(nn.Module):
    """Small MLP classifier returning logits for wildfire probability."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        act = _activation_from_name(activation)
        layers: list[nn.Module] = []
        prev = int(input_dim)

        for width in hidden_dims:
            width = int(width)
            if width <= 0:
                raise ValueError(f"hidden layer width must be > 0, got {width}")
            layers.append(nn.Linear(prev, width))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MLP input must be [N, input_dim], got shape={tuple(x.shape)}")
        x = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.net(x).squeeze(-1)
        return logits


def _activation_from_name(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation='{name}'. Use 'gelu' or 'relu'.")
