from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ForecastConfig:
    """Runtime defaults for recursive Monte Carlo future forecasting."""

    horizon_days: int = 7
    history_days: int = 64
    window_size: int = 16
    num_mc_samples: int = 100000
    mc_batch_size: int = 1024
    sigma_floor: float = 1e-6
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
