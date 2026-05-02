from __future__ import annotations

import torch


def _validate_projector_shapes(
    current_features: torch.Tensor,
    mu_delta: torch.Tensor,
    sigma_delta: torch.Tensor,
) -> tuple[int, int, int]:
    if current_features.ndim != 3:
        raise ValueError(
            f"current_features must be [F, H, W], got {tuple(current_features.shape)}"
        )
    if mu_delta.ndim != 3:
        raise ValueError(f"mu_delta must be [F, H, W], got {tuple(mu_delta.shape)}")
    if sigma_delta.ndim != 1:
        raise ValueError(f"sigma_delta must be [F], got {tuple(sigma_delta.shape)}")

    f, h, w = current_features.shape
    if tuple(mu_delta.shape) != (f, h, w):
        raise ValueError(
            "mu_delta shape mismatch. "
            f"expected={(f, h, w)}, got={tuple(mu_delta.shape)}"
        )
    if tuple(sigma_delta.shape) != (f,):
        raise ValueError(
            "sigma_delta shape mismatch. "
            f"expected={(f,)}, got={tuple(sigma_delta.shape)}"
        )
    return f, h, w


def _coerce_current_features(current_features: torch.Tensor) -> torch.Tensor:
    current = torch.as_tensor(current_features, dtype=torch.float32)
    if current.ndim == 1:
        current = current.unsqueeze(-1).unsqueeze(-1)
    return current


def _coerce_mu_delta(mu_delta: torch.Tensor) -> torch.Tensor:
    mu = torch.as_tensor(mu_delta, dtype=torch.float32)
    if mu.ndim == 1:
        mu = mu.unsqueeze(-1).unsqueeze(-1)
    return mu


def _coerce_sigma_delta(sigma_delta: torch.Tensor) -> torch.Tensor:
    sigma = torch.as_tensor(sigma_delta, dtype=torch.float32)
    if sigma.ndim == 0:
        sigma = sigma.view(1)
    return sigma


def _prepare_bounds(
    bounds: torch.Tensor | None,
    f: int,
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor | None:
    if bounds is None:
        return None
    b = torch.as_tensor(bounds, dtype=dtype, device=device)
    if b.ndim == 1 and tuple(b.shape) == (f,):
        return b.view(1, f, 1, 1)
    if b.ndim == 3 and tuple(b.shape) == (f, h, w):
        return b.unsqueeze(0)
    raise ValueError(
        f"{name} must be [F] or [F,H,W]. expected {(f,)} or {(f,h,w)}, got {tuple(b.shape)}"
    )


def recursive_sample_future_features_featurewise_homoscedastic(
    current_features: torch.Tensor,
    mu_delta: torch.Tensor,
    sigma_delta: torch.Tensor,
    horizon_days: int = 7,
    batch_size: int = 1024,
    generator: torch.Generator | None = None,
    feature_min: torch.Tensor | None = None,
    feature_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample B recursive Monte Carlo paths to t+horizon_days.

    The process follows:
      x_{t+h} = x_{t+h-1} + mu_delta + sigma_delta * eps_h
      eps_h ~ N(0, I), independently for each future step h.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    current = _coerce_current_features(current_features)
    mu = _coerce_mu_delta(mu_delta).to(device=current.device)
    sigma = _coerce_sigma_delta(sigma_delta).to(device=current.device)
    current = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=1e-6, posinf=1.0, neginf=1.0)

    f, h, w = _validate_projector_shapes(current, mu, sigma)
    x = current.unsqueeze(0).expand(batch_size, f, h, w).clone()

    mu_b = mu.unsqueeze(0)
    sigma_b = sigma.view(1, f, 1, 1)
    lower = _prepare_bounds(feature_min, f, h, w, device=x.device, dtype=x.dtype, name="feature_min")
    upper = _prepare_bounds(feature_max, f, h, w, device=x.device, dtype=x.dtype, name="feature_max")

    for _ in range(horizon_days):
        eps = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        noise = mu_b + sigma_b * eps
        x = x + noise
        if lower is not None:
            x = torch.maximum(x, lower)
        if upper is not None:
            x = torch.minimum(x, upper)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if tuple(x.shape) != (batch_size, f, h, w):
        raise AssertionError(
            "Projected feature shape mismatch. "
            f"expected={(batch_size, f, h, w)}, got={tuple(x.shape)}"
        )
    return x


def deterministic_recursive_future_features(
    current_features: torch.Tensor,
    mu_delta: torch.Tensor,
    horizon_days: int,
) -> torch.Tensor:
    """Deterministic comparison projection using only mean drift."""
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    current = _coerce_current_features(current_features)
    mu = _coerce_mu_delta(mu_delta).to(device=current.device)
    _validate_projector_shapes(current, mu, torch.zeros((current.shape[0],), device=current.device))

    x = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(horizon_days):
        x = x + mu
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x
