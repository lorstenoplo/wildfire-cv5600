from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss on top of BCE-with-logits."""

    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


def build_loss(
    pos_weight: float = 1.0,
    loss_name: str = "bce",
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build training criterion.

    Default and recommended path is BCEWithLogitsLoss.
    """
    key = str(loss_name).strip().lower()
    pw = float(pos_weight)
    if pw <= 0:
        pw = 1.0

    if key in {"bce", "bcewithlogits", "bcewithlogitsloss", "default"}:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))
    if key in {"focal", "focal_bce", "focal_bce_with_logits"}:
        return FocalBCEWithLogitsLoss(pos_weight=pw, gamma=float(focal_gamma))

    raise ValueError(f"Unsupported loss_name={loss_name}. Use 'bce' or 'focal'.")
