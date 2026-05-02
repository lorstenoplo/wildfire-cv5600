from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .feature_stats import compute_featurewise_homoscedastic_noise_stats
from .mlp_model import FutureProbabilityMLP
from .monte_carlo_projector import recursive_sample_future_features_featurewise_homoscedastic
from .training_data import build_training_records


def _build_streaming_mc_inputs_on_device(
    projected_t_plus_h_features: torch.Tensor,
    current_probability: float,
    future_day_of_year: int,
) -> torch.Tensor:
    """Build [N, F+3] MLP inputs directly on the projected feature device."""
    x = torch.as_tensor(projected_t_plus_h_features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(
            "projected_t_plus_h_features must be [F] or [N,F], "
            f"got shape={tuple(x.shape)}"
        )
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n, _ = x.shape
    p = torch.as_tensor(
        current_probability,
        dtype=x.dtype,
        device=x.device,
    ).view(1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).expand(n, 1)

    angle = 2.0 * math.pi * float(future_day_of_year) / 365.0
    sin_day = torch.full((n, 1), float(math.sin(angle)), dtype=x.dtype, device=x.device)
    cos_day = torch.full((n, 1), float(math.cos(angle)), dtype=x.dtype, device=x.device)

    out = torch.cat([x, p, sin_day, cos_day], dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class CellwiseMCDataset(Dataset):
    """Dataset over row-wise supervised records for cell-wise MLP training."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("records must be non-empty")
        self.records = records
        x_list: list[torch.Tensor] = []
        y_list: list[float] = []
        for idx, rec in enumerate(records):
            if "x" not in rec or "y" not in rec:
                raise KeyError(f"record at idx={idx} must include 'x' and 'y'")
            x = torch.as_tensor(rec["x"], dtype=torch.float32).flatten()
            if x.ndim != 1:
                raise ValueError(
                    f"record x must be 1D after flatten, idx={idx}, got shape={tuple(x.shape)}"
                )
            y = float(rec["y"])
            x_list.append(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
            y_list.append(y)

        feature_dim = int(x_list[0].numel())
        for idx, x in enumerate(x_list):
            if x.numel() != feature_dim:
                raise ValueError(
                    "All record x vectors must have equal length. "
                    f"idx={idx} has {x.numel()}, expected {feature_dim}"
                )

        self.x = torch.stack(x_list, dim=0).to(torch.float32)
        self.y = torch.as_tensor(y_list, dtype=torch.float32)
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
        }


def build_dataloader_from_records(
    records: list[dict[str, Any]],
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    dataset = CellwiseMCDataset(records=records)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
    )


@dataclass
class TrainMLPConfig:
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "gelu"
    dropout: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    use_data_parallel: bool = True
    gpu_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CellTrainingPoint:
    """One supervised anchor point for cell-wise t+h training."""

    anchor_date: str
    target_date: str
    row: int
    col: int
    history_features: torch.Tensor
    current_features: torch.Tensor
    current_probability_at_t: float
    observed_t_plus_h_label_or_probability: float
    future_day_of_year_t_plus_h: int | None = None


@dataclass
class StreamingMCTrainConfig:
    """Memory-safe training config for large MC expansion."""

    max_train_points: int = 1000
    mc_samples_per_point: int = 10000
    horizon_days: int = 7
    mc_batch_size: int = 256
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "gelu"
    dropout: float = 0.0
    sigma_floor: float = 1e-6
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    selection_mode: str = "random"  # random or first
    use_data_parallel: bool = True
    gpu_ids: tuple[int, ...] | None = None


def select_training_points(
    points: list[CellTrainingPoint],
    max_train_points: int = 1000,
    seed: int = 42,
    mode: str = "random",
) -> list[CellTrainingPoint]:
    """Select at most max_train_points from available points."""
    if max_train_points <= 0:
        raise ValueError(f"max_train_points must be positive, got {max_train_points}")
    if not points:
        raise ValueError("points must be non-empty")

    n = len(points)
    k = min(int(max_train_points), n)
    if k == n:
        return points

    mode_key = str(mode).strip().lower()
    if mode_key == "first":
        return points[:k]
    if mode_key == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:k].tolist()
        return [points[i] for i in idx]
    raise ValueError(f"Unsupported selection mode '{mode}'. Use 'random' or 'first'.")


def _resolve_data_parallel_ids(gpu_ids: tuple[int, ...] | None) -> list[int]:
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        return []
    if gpu_ids is None:
        return [0, 1]
    ids = [int(g) for g in gpu_ids]
    valid = [g for g in ids if 0 <= g < n_gpu]
    if len(valid) < 2:
        return []
    return valid[:2]


def build_training_records_from_cell_timeslice(
    anchor_date: str,
    target_date: str,
    row: int,
    col: int,
    history_features: torch.Tensor,
    current_features: torch.Tensor,
    current_probability_at_t: float | torch.Tensor,
    observed_t_plus_h_label_or_probability: float | torch.Tensor,
    *,
    future_day_of_year_t_plus_h: int | None = None,
    horizon_days: int = 7,
    num_mc_samples: int = 10000,
    mc_batch_size: int = 256,
    sigma_floor: float = 1e-6,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[dict[str, Any]]:
    """Create supervised MC-expanded rows for one (date,row,col) key.

    Only future features are simulated. Y is the real observed t+h label/probability.
    """
    if num_mc_samples <= 0:
        raise ValueError(f"num_mc_samples must be positive, got {num_mc_samples}")
    if mc_batch_size <= 0:
        raise ValueError(f"mc_batch_size must be positive, got {mc_batch_size}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    dev = torch.device(device)
    history = torch.as_tensor(history_features, dtype=torch.float32, device=dev)
    current = torch.as_tensor(current_features, dtype=torch.float32, device=dev)

    mu_delta, sigma_delta = compute_featurewise_homoscedastic_noise_stats(
        history_features=history,
        sigma_floor=float(sigma_floor),
    )
    generator = torch.Generator(device=dev.type if dev.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))

    records: list[dict[str, Any]] = []
    processed = 0
    while processed < int(num_mc_samples):
        batch_size = min(int(mc_batch_size), int(num_mc_samples) - processed)
        projected = recursive_sample_future_features_featurewise_homoscedastic(
            current_features=current,
            mu_delta=mu_delta,
            sigma_delta=sigma_delta,
            horizon_days=int(horizon_days),
            batch_size=int(batch_size),
            generator=generator,
        )
        # Cell-wise expected shape here is [B,F,1,1]. Flatten to [B,F] for training rows.
        projected_bf = projected[:, :, 0, 0]
        rows = build_training_records(
            anchor_date=anchor_date,
            target_date=target_date,
            row=int(row),
            col=int(col),
            projected_t_plus_h_features=projected_bf.detach().cpu(),
            current_probability_at_t=float(torch.as_tensor(current_probability_at_t).item()),
            future_day_of_year_t_plus_h=future_day_of_year_t_plus_h,
            observed_t_plus_h_label_or_probability=float(
                torch.as_tensor(observed_t_plus_h_label_or_probability).item()
            ),
            mc_id_offset=processed,
        )
        records.extend(rows)
        processed += batch_size
    return records


def train_future_probability_mlp_streaming_mc(
    points: list[CellTrainingPoint],
    config: StreamingMCTrainConfig | None = None,
) -> dict[str, Any]:
    """Train on selected points with MC simulation per point (streaming).

    This enforces the requested regime:
      - use only up to max_train_points anchors
      - run mc_samples_per_point simulations per selected anchor
    without materializing all MC records in memory.
    """
    cfg = config or StreamingMCTrainConfig()
    selected_points = select_training_points(
        points=points,
        max_train_points=int(cfg.max_train_points),
        seed=int(cfg.seed),
        mode=cfg.selection_mode,
    )
    device = torch.device(cfg.device)

    if not selected_points:
        raise ValueError("No training points selected.")

    # Infer input_dim from first point (F + 3).
    first_f = int(torch.as_tensor(selected_points[0].current_features).numel())
    input_dim = first_f + 3
    base_model = FutureProbabilityMLP(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout=float(cfg.dropout),
    ).to(device)
    if bool(cfg.use_data_parallel) and device.type == "cuda":
        dp_ids = _resolve_data_parallel_ids(cfg.gpu_ids)
        if dp_ids:
            model: nn.Module = nn.DataParallel(base_model, device_ids=dp_ids, output_device=dp_ids[0])
            print(f"[Train-Streaming] Using DataParallel on GPUs {dp_ids}")
        else:
            model = base_model
            print("[Train-Streaming] DataParallel requested but <2 valid CUDA GPUs found. Using single GPU.")
    else:
        model = base_model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()

    history_rows: list[dict[str, float]] = []
    total_simulations = int(len(selected_points) * int(cfg.mc_samples_per_point))
    print(
        "[Train-Streaming] "
        f"selected_points={len(selected_points)} "
        f"mc_samples_per_point={cfg.mc_samples_per_point} "
        f"total_simulations={total_simulations}"
    )

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for p_idx, p in enumerate(selected_points):
            history_t = torch.as_tensor(p.history_features, dtype=torch.float32, device=device)
            current_t = torch.as_tensor(p.current_features, dtype=torch.float32, device=device)
            future_doy = (
                int(p.future_day_of_year_t_plus_h)
                if p.future_day_of_year_t_plus_h is not None
                else int(date.fromisoformat(str(p.target_date)).timetuple().tm_yday)
            )
            mu_delta, sigma_delta = compute_featurewise_homoscedastic_noise_stats(
                history_features=history_t,
                sigma_floor=float(cfg.sigma_floor),
            )
            generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
            generator.manual_seed(int(cfg.seed + p_idx))

            processed = 0
            while processed < int(cfg.mc_samples_per_point):
                batch_size = min(int(cfg.mc_batch_size), int(cfg.mc_samples_per_point) - processed)
                projected = recursive_sample_future_features_featurewise_homoscedastic(
                    current_features=current_t,
                    mu_delta=mu_delta,
                    sigma_delta=sigma_delta,
                    horizon_days=int(cfg.horizon_days),
                    batch_size=batch_size,
                    generator=generator,
                )
                projected_bf = projected[:, :, 0, 0]

                x = _build_streaming_mc_inputs_on_device(
                    projected_t_plus_h_features=projected_bf,
                    current_probability=float(p.current_probability_at_t),
                    future_day_of_year=future_doy,
                )

                y = torch.full(
                    (batch_size,),
                    fill_value=float(p.observed_t_plus_h_label_or_probability),
                    dtype=torch.float32,
                    device=x.device,
                ).clamp(0.0, 1.0)
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().item()) * batch_size
                epoch_count += batch_size
                processed += batch_size

        train_loss = epoch_loss_sum / max(epoch_count, 1)
        history_rows.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "selected_points": float(len(selected_points)),
                "mc_samples_per_point": float(cfg.mc_samples_per_point),
                "total_simulations": float(total_simulations),
            }
        )
        print(
            f"[Train-Streaming] epoch={epoch + 1}/{cfg.epochs} "
            f"train_loss={train_loss:.6f}"
        )

    return {
        "model": model,
        "history": history_rows,
        "config": cfg,
        "selected_points_count": len(selected_points),
        "total_simulations": total_simulations,
    }


def train_future_probability_mlp(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]] | None = None,
    config: TrainMLPConfig | None = None,
) -> dict[str, Any]:
    """Train MLP on projected features (X) and real future labels (Y).

    Training loss uses BCEWithLogitsLoss on raw logits.
    """
    cfg = config or TrainMLPConfig()
    train_ds = CellwiseMCDataset(train_records)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_ds = CellwiseMCDataset(val_records) if val_records else None
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    device = torch.device(cfg.device)
    base_model = FutureProbabilityMLP(
        input_dim=int(train_ds.feature_dim),
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout=float(cfg.dropout),
    ).to(device)
    if bool(cfg.use_data_parallel) and device.type == "cuda":
        dp_ids = _resolve_data_parallel_ids(cfg.gpu_ids)
        if dp_ids:
            model: nn.Module = nn.DataParallel(base_model, device_ids=dp_ids, output_device=dp_ids[0])
            print(f"[Train] Using DataParallel on GPUs {dp_ids}")
        else:
            model = base_model
            print("[Train] DataParallel requested but <2 valid CUDA GPUs found. Using single GPU.")
    else:
        model = base_model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_n = int(x.shape[0])
            total_loss += float(loss.detach().item()) * batch_n
            total_count += batch_n

        train_loss = total_loss / max(total_count, 1)
        row: dict[str, float] = {"epoch": float(epoch + 1), "train_loss": float(train_loss)}

        if val_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device=device, dtype=torch.float32)
                    y = batch["y"].to(device=device, dtype=torch.float32)
                    logits = model(x)
                    loss = criterion(logits, y)
                    probs = torch.sigmoid(logits)  # sigmoid only outside model
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    _ = probs  # explicit; probabilities available for downstream metrics
                    batch_n = int(x.shape[0])
                    v_loss_sum += float(loss.detach().item()) * batch_n
                    v_n += batch_n
            row["val_loss"] = float(v_loss_sum / max(v_n, 1))

        history.append(row)
        if "val_loss" in row:
            print(
                f"[Train] epoch={int(row['epoch'])}/{cfg.epochs} "
                f"train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f}"
            )
        else:
            print(f"[Train] epoch={int(row['epoch'])}/{cfg.epochs} train_loss={row['train_loss']:.6f}")

    return {
        "model": model,
        "history": history,
        "config": cfg,
    }


def predict_probabilities_from_records(
    model: nn.Module,
    records: list[dict[str, Any]],
    batch_size: int = 4096,
    device: str | None = None,
) -> torch.Tensor:
    """Predict probabilities from training records.

    Model emits logits; sigmoid is applied here for inference.
    """
    ds = CellwiseMCDataset(records)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, drop_last=False)
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=dev, dtype=torch.float32)
            logits = model(x)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            out.append(probs.detach().cpu())
    if not out:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(out, dim=0)
