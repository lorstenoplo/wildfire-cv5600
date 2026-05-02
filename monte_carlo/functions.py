from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from future_mc_forecast.forecast_runner import run_recursive_future_forecast


def run_future_mc_forecast(
    history_features: torch.Tensor | np.ndarray,
    current_features: torch.Tensor | np.ndarray,
    current_probability: torch.Tensor | np.ndarray | float,
    future_day_of_year: int,
    model_checkpoint: str | Path | None = None,
    output_dir: str | Path = "outputs/future_forecast",
    horizon_days: int = 7,
    num_mc_samples: int = 100000,
    mc_batch_size: int = 1024,
    seed: int = 42,
    allow_random_model: bool = False,
    use_data_parallel: bool = True,
    gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the recursive Monte Carlo forecast module."""
    return run_recursive_future_forecast(
        history_features=history_features,
        current_features=current_features,
        current_probability=current_probability,
        future_day_of_year=future_day_of_year,
        model_checkpoint=model_checkpoint,
        horizon_days=horizon_days,
        num_mc_samples=num_mc_samples,
        mc_batch_size=mc_batch_size,
        output_dir=output_dir,
        seed=seed,
        allow_random_model=allow_random_model,
        use_data_parallel=use_data_parallel,
        gpu_ids=gpu_ids,
    )
