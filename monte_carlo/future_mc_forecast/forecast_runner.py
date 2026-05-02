from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ForecastConfig
from .feature_stats import (
    compute_featurewise_homoscedastic_noise_stats,
    compute_history_window_means,
)
from .io_utils import save_metadata_json, save_numpy_map, validate_tensor_shapes
from .mlp_model import FutureProbabilityMLP
from .monte_carlo_projector import (
    deterministic_recursive_future_features,
    recursive_sample_future_features_featurewise_homoscedastic,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def run_recursive_future_forecast(
    history_features: torch.Tensor | np.ndarray,
    current_features: torch.Tensor | np.ndarray,
    current_probability: torch.Tensor | np.ndarray | float,
    future_day_of_year: int,
    model_checkpoint: str | Path | None = None,
    horizon_days: int = 7,
    num_mc_samples: int = 100000,
    mc_batch_size: int = 1024,
    output_dir: str | Path = "outputs/future_forecast",
    seed: int = 42,
    allow_random_model: bool = False,
    use_data_parallel: bool = True,
    gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run recursive t+7 wildfire probability forecast with Monte Carlo.

    Supports:
      - grid mode: history [F,64,H,W], current [F,H,W], probability [H,W]
      - cell mode: history [F,64], current [F], probability scalar
    """
    total_start = time.time()
    cfg = ForecastConfig(
        horizon_days=int(horizon_days),
        num_mc_samples=int(num_mc_samples),
        mc_batch_size=int(mc_batch_size),
        seed=int(seed),
    )

    if cfg.horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {cfg.horizon_days}")
    if cfg.num_mc_samples <= 0:
        raise ValueError(f"num_mc_samples must be positive, got {cfg.num_mc_samples}")
    if cfg.mc_batch_size <= 0:
        raise ValueError(f"mc_batch_size must be positive, got {cfg.mc_batch_size}")
    if not (1 <= int(future_day_of_year) <= 366):
        raise ValueError(f"future_day_of_year must be in [1, 366], got {future_day_of_year}")

    history_t, current_t, prob_t, shape_meta = validate_tensor_shapes(
        history_features=history_features,
        current_features=current_features,
        current_probability=current_probability,
        expected_history_days=cfg.history_days,
    )

    device = torch.device(cfg.device)
    history_t = history_t.to(device=device, dtype=torch.float32)
    current_t = current_t.to(device=device, dtype=torch.float32)
    prob_t = prob_t.to(device=device, dtype=torch.float32)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    mc_generator = _build_generator(device=device, seed=cfg.seed)

    stats_start = time.time()
    mu_delta, sigma_delta = compute_featurewise_homoscedastic_noise_stats(
        history_t, sigma_floor=cfg.sigma_floor
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    stats_time = time.time() - stats_start
    print(f"[Timing] Historical noise statistics computed in {stats_time:.3f} seconds")

    # Optional diagnostics from four 16-day windows.
    window_means = compute_history_window_means(history_t, window_size=cfg.window_size)
    window_means_global = window_means.mean(dim=(2, 3)).detach().cpu().numpy().tolist()

    num_features, _, h, w = history_t.shape
    model = FutureProbabilityMLP(input_dim=int(num_features + 3)).to(device)

    if model_checkpoint is not None:
        ckpt = torch.load(model_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=True)
    elif not allow_random_model:
        raise ValueError(
            "model_checkpoint is required for inference with a trained MLP. "
            "Set allow_random_model=True only for debug smoke tests."
        )

    if bool(use_data_parallel) and device.type == "cuda" and torch.cuda.device_count() >= 2:
        if gpu_ids is None:
            dp_ids = [0, 1]
        else:
            dp_ids = [int(g) for g in gpu_ids if 0 <= int(g) < torch.cuda.device_count()]
            dp_ids = dp_ids[:2]
        if len(dp_ids) >= 2:
            model = torch.nn.DataParallel(model, device_ids=dp_ids, output_device=dp_ids[0])
            print(f"[Inference] Using DataParallel on GPUs {dp_ids}")
    elif bool(use_data_parallel) and device.type == "cuda":
        print("[Inference] DataParallel requested but <2 CUDA GPUs found. Using single GPU.")

    model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_angle = 2.0 * math.pi * float(future_day_of_year) / 365.0
    sin_day = float(math.sin(day_angle))
    cos_day = float(math.cos(day_angle))

    with torch.no_grad():
        det_start = time.time()
        det_projected = deterministic_recursive_future_features(
            current_features=current_t,
            mu_delta=mu_delta,
            horizon_days=cfg.horizon_days,
        )
        det_input = _build_mlp_input(
            projected_features=det_projected.unsqueeze(0),
            current_probability=prob_t,
            sin_day=sin_day,
            cos_day=cos_day,
        )
        det_logits = model(det_input)
        det_prob = torch.sigmoid(det_logits).view(h, w)
        det_prob = torch.nan_to_num(det_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        det_time = time.time() - det_start
        print(
            f"[Timing] Deterministic t+{cfg.horizon_days} prediction completed in {det_time:.3f} seconds"
        )

        num_batches = math.ceil(cfg.num_mc_samples / cfg.mc_batch_size)
        batch_iter = range(num_batches)
        if tqdm is not None:
            batch_iter = tqdm(batch_iter, total=num_batches, desc="Monte Carlo")

        sum_prob = torch.zeros((h, w), dtype=torch.float32, device=device)
        sum_prob_sq = torch.zeros((h, w), dtype=torch.float32, device=device)
        sum_feat = torch.zeros((num_features, h, w), dtype=torch.float32, device=device)
        sum_feat_sq = torch.zeros((num_features, h, w), dtype=torch.float32, device=device)
        count = 0
        processed_samples = 0
        mc_start = time.time()

        for batch_idx in batch_iter:
            remaining = cfg.num_mc_samples - processed_samples
            batch_size = min(cfg.mc_batch_size, remaining)
            batch_start = time.time()

            projected_features = recursive_sample_future_features_featurewise_homoscedastic(
                current_features=current_t,
                mu_delta=mu_delta,
                sigma_delta=sigma_delta,
                horizon_days=cfg.horizon_days,
                batch_size=batch_size,
                generator=mc_generator,
            )

            mlp_input = _build_mlp_input(
                projected_features=projected_features,
                current_probability=prob_t,
                sin_day=sin_day,
                cos_day=cos_day,
            )
            logits = model(mlp_input)
            prob = torch.sigmoid(logits).view(batch_size, h, w)
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            sum_prob = sum_prob + prob.sum(dim=0)
            sum_prob_sq = sum_prob_sq + (prob * prob).sum(dim=0)
            sum_feat = sum_feat + projected_features.sum(dim=0)
            sum_feat_sq = sum_feat_sq + (projected_features * projected_features).sum(dim=0)
            count += int(batch_size)
            processed_samples += int(batch_size)

            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            elapsed_mc = time.time() - mc_start
            avg_time_per_sample = elapsed_mc / float(processed_samples)
            remaining_samples = cfg.num_mc_samples - processed_samples
            eta_seconds = avg_time_per_sample * float(remaining_samples)

            print(
                f"[Timing] MC batch {batch_idx + 1}/{num_batches} | "
                f"batch_size={batch_size} | "
                f"batch_time={batch_time:.3f}s | "
                f"processed={processed_samples}/{cfg.num_mc_samples} | "
                f"avg_time_per_sample={avg_time_per_sample:.6f}s | "
                f"eta={eta_seconds:.1f}s"
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        mc_total_time = time.time() - mc_start
        print(f"[Timing] Monte Carlo simulation completed in {mc_total_time:.3f} seconds")
        print(
            "[Timing] Average time per MC sample: "
            f"{mc_total_time / float(cfg.num_mc_samples):.6f} seconds"
        )

    if count <= 0:
        raise RuntimeError("No Monte Carlo samples were processed.")

    probability_mean = sum_prob / float(count)
    probability_var = sum_prob_sq / float(count) - probability_mean * probability_mean
    probability_var = torch.clamp(probability_var, min=0.0)
    probability_std = torch.sqrt(probability_var)
    probability_mean = probability_mean.clamp(0.0, 1.0)
    probability_std = torch.nan_to_num(probability_std, nan=0.0, posinf=0.0, neginf=0.0)

    projected_mean = sum_feat / float(count)
    projected_var = sum_feat_sq / float(count) - projected_mean * projected_mean
    projected_var = torch.clamp(projected_var, min=0.0)
    projected_std = torch.sqrt(projected_var)
    projected_mean = torch.nan_to_num(projected_mean, nan=0.0, posinf=0.0, neginf=0.0)
    projected_std = torch.nan_to_num(projected_std, nan=0.0, posinf=0.0, neginf=0.0)

    save_numpy_map(probability_mean, out_dir / "probability_t_plus_7_monte_carlo_mean.npy")
    save_numpy_map(probability_std, out_dir / "probability_t_plus_7_monte_carlo_std.npy")
    save_numpy_map(det_prob, out_dir / "probability_t_plus_7_deterministic.npy")
    save_numpy_map(projected_mean, out_dir / "projected_features_t_plus_7_mean.npy")
    save_numpy_map(projected_std, out_dir / "projected_features_t_plus_7_std.npy")

    total_time = time.time() - total_start
    timing_metadata = {
        "stats_time_seconds": float(stats_time),
        "deterministic_time_seconds": float(det_time),
        "monte_carlo_time_seconds": float(mc_total_time),
        "total_time_seconds": float(total_time),
        "num_mc_samples": int(cfg.num_mc_samples),
        "mc_batch_size": int(cfg.mc_batch_size),
        "average_time_per_mc_sample_seconds": float(mc_total_time / float(cfg.num_mc_samples)),
    }

    metadata = {
        "modeling_note": (
            "We model future feature evolution as a recursive Monte Carlo random walk "
            "initialized at the current-day feature state. The daily perturbation is "
            "Gaussian, with cell-wise mean drift and feature-wise homoscedastic variance "
            "estimated from the previous 64 days of daily feature differences. Future wildfire "
            "probability is obtained by passing each simulated t + 7 feature realization "
            "through an MLP and averaging the resulting sigmoid probabilities."
        ),
        "input_shapes": {
            "history_features": tuple(history_t.shape),
            "current_features": tuple(current_t.shape),
            "current_probability": tuple(prob_t.shape),
        },
        "output_shapes": {
            "probability_mean": tuple(probability_mean.shape),
            "probability_std": tuple(probability_std.shape),
            "deterministic_probability": tuple(det_prob.shape),
            "projected_features_mean": tuple(projected_mean.shape),
            "projected_features_std": tuple(projected_std.shape),
        },
        "settings": {
            "horizon_days": int(cfg.horizon_days),
            "history_days": int(cfg.history_days),
            "num_features": int(num_features),
            "num_mc_samples": int(cfg.num_mc_samples),
            "mc_batch_size": int(cfg.mc_batch_size),
            "sigma_floor": float(cfg.sigma_floor),
            "seed": int(cfg.seed),
            "device": str(device),
            "future_day_of_year": int(future_day_of_year),
            "temporal_encoding": {"sin": sin_day, "cos": cos_day},
        },
        "shape_metadata": shape_meta,
        "window_diagnostics_16day_feature_means": window_means_global,
        "timing": timing_metadata,
    }
    save_metadata_json(metadata, out_dir / "metadata.json")

    print("=" * 80)
    print("[Timing Summary]")
    print(f"Historical stats time       : {stats_time:.3f} seconds")
    print(f"Deterministic forecast time : {det_time:.3f} seconds")
    print(f"Monte Carlo forecast time   : {mc_total_time:.3f} seconds")
    print(f"Total forecast runtime      : {total_time:.3f} seconds")
    print(f"Total MC samples            : {cfg.num_mc_samples}")
    print(
        "Average time per MC sample  : "
        f"{mc_total_time / float(cfg.num_mc_samples):.6f} seconds"
    )
    print("=" * 80)

    return {
        "probability_mean": probability_mean.detach().cpu(),
        "probability_std": probability_std.detach().cpu(),
        "deterministic_probability": det_prob.detach().cpu(),
        "projected_features_mean": projected_mean.detach().cpu(),
        "projected_features_std": projected_std.detach().cpu(),
        "metadata": metadata,
    }


def _build_mlp_input(
    projected_features: torch.Tensor,
    current_probability: torch.Tensor,
    sin_day: float,
    cos_day: float,
) -> torch.Tensor:
    if projected_features.ndim != 4:
        raise ValueError(
            f"projected_features must be [B,F,H,W], got {tuple(projected_features.shape)}"
        )
    if current_probability.ndim != 2:
        raise ValueError(
            f"current_probability must be [H,W], got {tuple(current_probability.shape)}"
        )

    batch_size, num_features, h, w = projected_features.shape
    if tuple(current_probability.shape) != (h, w):
        raise ValueError(
            "current_probability shape mismatch with projected_features. "
            f"expected={(h, w)}, got={tuple(current_probability.shape)}"
        )

    feat = projected_features.permute(0, 2, 3, 1).contiguous()  # [B,H,W,F]
    prob = current_probability.view(1, h, w, 1).expand(batch_size, h, w, 1)
    sin_map = torch.full(
        (batch_size, h, w, 1),
        fill_value=float(sin_day),
        dtype=feat.dtype,
        device=feat.device,
    )
    cos_map = torch.full(
        (batch_size, h, w, 1),
        fill_value=float(cos_day),
        dtype=feat.dtype,
        device=feat.device,
    )
    mlp_input = torch.cat([feat, prob, sin_map, cos_map], dim=-1)
    mlp_input = mlp_input.view(batch_size * h * w, num_features + 3)
    mlp_input = torch.nan_to_num(mlp_input, nan=0.0, posinf=0.0, neginf=0.0)
    return mlp_input


def _build_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """Create a device-compatible random generator when available."""
    try:
        if device.type == "cuda":
            g = torch.Generator(device="cuda")
        else:
            g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        return g
    except Exception:
        # Fallback: rely on global torch seed.
        return None
