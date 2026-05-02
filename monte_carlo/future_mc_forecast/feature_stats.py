from __future__ import annotations

import torch


def compute_featurewise_homoscedastic_noise_stats(
    history_features: torch.Tensor,
    sigma_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate daily Gaussian noise stats from 64-day feature history.

    Args:
        history_features: Tensor with shape [F, 64, H, W].
        sigma_floor: Minimum feature std to avoid degenerate noise.

    Returns:
        mu_delta: Cell-wise mean daily change, shape [F, H, W].
        sigma_delta: Feature-wise homoscedastic std, shape [F].
    """
    history = torch.as_tensor(history_features, dtype=torch.float32)
    if history.ndim == 2:
        history = history.unsqueeze(-1).unsqueeze(-1)
    if history.ndim != 4:
        raise ValueError(
            "history_features must be [F,64,H,W] or cell-wise [F,64], "
            f"got shape={tuple(history.shape)}"
        )
    _, history_days, _, _ = history.shape
    if history_days != 64:
        raise ValueError(
            f"history_features second dimension must be 64 days, got {history_days}"
        )

    history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
    deltas = history[:, 1:, :, :] - history[:, :-1, :, :]
    if deltas.shape[1] != 63:
        raise AssertionError(
            f"Expected 63 daily deltas from 64 days, got {deltas.shape[1]}"
        )

    mu_delta = deltas.mean(dim=1)
    sigma_delta = deltas.std(dim=(1, 2, 3), unbiased=False)
    sigma_delta = torch.clamp(sigma_delta, min=float(sigma_floor))

    expected_mu = (history.shape[0], history.shape[2], history.shape[3])
    expected_sigma = (history.shape[0],)
    if tuple(mu_delta.shape) != expected_mu:
        raise AssertionError(
            f"mu_delta shape mismatch. expected={expected_mu}, got={tuple(mu_delta.shape)}"
        )
    if tuple(sigma_delta.shape) != expected_sigma:
        raise AssertionError(
            "sigma_delta shape mismatch. "
            f"expected={expected_sigma}, got={tuple(sigma_delta.shape)}"
        )

    mu_delta = torch.nan_to_num(mu_delta, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_delta = torch.nan_to_num(sigma_delta, nan=float(sigma_floor), posinf=1.0, neginf=1.0)
    sigma_delta = torch.clamp(sigma_delta, min=float(sigma_floor))
    return mu_delta, sigma_delta


def compute_history_window_means(
    history_features: torch.Tensor,
    window_size: int = 16,
) -> torch.Tensor:
    """Compute optional 16-day rolling-window means for diagnostics/logging.

    Args:
        history_features: Tensor with shape [F, 64, H, W].
        window_size: Number of days per diagnostic window.

    Returns:
        Tensor with shape [F, num_windows, H, W].
    """
    history = torch.as_tensor(history_features, dtype=torch.float32)
    if history.ndim == 2:
        history = history.unsqueeze(-1).unsqueeze(-1)
    if history.ndim != 4:
        raise ValueError(
            "history_features must be [F,64,H,W] or [F,64], "
            f"got shape={tuple(history.shape)}"
        )
    _, days, _, _ = history.shape
    if days % window_size != 0:
        raise ValueError(
            f"history day count ({days}) must be divisible by window_size ({window_size})"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
    num_windows = days // window_size
    windows: list[torch.Tensor] = []
    for idx in range(num_windows):
        start = idx * window_size
        end = (idx + 1) * window_size
        windows.append(history[:, start:end, :, :].mean(dim=1))
    return torch.stack(windows, dim=1)
