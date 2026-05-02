from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .configs import DataConfig, TrainConfig
from .losses import build_loss
from .metrics import compute_binary_metrics, find_best_threshold_by_f1


def make_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    cfg: DataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=bool(cfg.drop_last_train),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def _to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True).float()
    return x, y


def maybe_wrap_data_parallel(model: nn.Module, use_multi_gpu: bool = True) -> nn.Module:
    if use_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def _state_dict_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    base = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}


def _load_state_dict_any(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    base = model.module if isinstance(model, nn.DataParallel) else model
    base.load_state_dict(state, strict=True)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    name = str(getattr(cfg, "scheduler_name", "none")).lower().strip()
    if name in {"", "none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=float(getattr(cfg, "min_lr", 1e-5)),
        )
    raise ValueError(f"Unsupported scheduler_name={cfg.scheduler_name}")


def _monitor_direction(metric_name: str) -> tuple[str, bool]:
    key = str(metric_name).lower().strip()
    if key in {"val_loss", "loss"}:
        return "val_loss", True  # lower is better
    if key in {"pr_auc", "f1", "iou", "precision", "recall", "specificity"}:
        return key, False  # higher is better
    raise ValueError(f"Unsupported monitor_metric={metric_name}")


def _is_better(cur: float, best: float | None, lower_better: bool) -> bool:
    if np.isnan(cur):
        return False
    if best is None:
        return True
    if lower_better:
        return cur < best
    return cur > best


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip_norm: float = 0.0,
    use_amp: bool = True,
) -> float:
    model.train()
    running = 0.0
    n = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    for batch in loader:
        x, y = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

        bs = int(y.shape[0])
        running += float(loss.item()) * bs
        n += bs

    return float(running / max(n, 1))


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    ys = []
    ps = []

    for batch in loader:
        x, y = _to_device(batch, device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        if criterion is not None:
            loss = criterion(logits, y)
            bs = int(y.shape[0])
            running_loss += float(loss.item()) * bs
            total_samples += bs
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.float32)
    y_prob = np.concatenate(ps, axis=0) if ps else np.array([], dtype=np.float32)
    out = {
        "loss": float(running_loss / max(total_samples, 1)) if criterion is not None else float("nan"),
        "y_true": y_true.astype(np.float32),
        "y_prob": y_prob.astype(np.float32),
    }
    return out


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    use_multi_gpu: bool = True,
) -> dict[str, Any]:
    model = model.to(device)
    model = maybe_wrap_data_parallel(model, use_multi_gpu=use_multi_gpu)

    criterion = build_loss(
        pos_weight=cfg.pos_weight,
        loss_name=getattr(cfg, "loss_name", "bce"),
        focal_gamma=getattr(cfg, "focal_gamma", 2.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = _build_scheduler(optimizer, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp and device.type == "cuda"))

    monitor_key, lower_better = _monitor_direction(getattr(cfg, "monitor_metric", "pr_auc"))
    best_monitor: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int = 0
    bad_epochs = 0

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip_norm=cfg.grad_clip_norm,
            use_amp=cfg.use_amp,
        )

        val_out = evaluate_loader(model=model, loader=val_loader, criterion=criterion, device=device)
        val_loss = float(val_out["loss"])
        best_val_loss = min(best_val_loss, val_loss)

        val_metrics_05 = compute_binary_metrics(val_out["y_true"], val_out["y_prob"], threshold=0.5)

        row = {
            "epoch": float(epoch),
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_pr_auc": float(val_metrics_05["pr_auc"]),
            "val_precision": float(val_metrics_05["precision"]),
            "val_recall": float(val_metrics_05["recall"]),
            "val_f1": float(val_metrics_05["f1"]),
            "val_iou": float(val_metrics_05["iou"]),
            "val_specificity": float(val_metrics_05["specificity"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        cur_monitor = float(row[monitor_key if monitor_key == "val_loss" else f"val_{monitor_key}"])
        if _is_better(cur_monitor, best_monitor, lower_better=lower_better):
            best_monitor = cur_monitor
            best_epoch = int(epoch)
            best_state = _state_dict_for_save(model)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()

        if bad_epochs >= int(cfg.patience):
            break

    if best_state is not None:
        _load_state_dict_any(model, best_state)

    return {
        "model": model,
        "best_state": best_state,
        "best_val_loss": float(best_val_loss),
        "best_monitor_metric": str(getattr(cfg, "monitor_metric", "pr_auc")),
        "best_monitor_value": float(best_monitor) if best_monitor is not None else float("nan"),
        "best_epoch": int(best_epoch),
        "history": history,
        "train_config": asdict(cfg),
        "device": str(device),
        "n_gpu_used": int(torch.cuda.device_count()) if (use_multi_gpu and torch.cuda.is_available()) else int(
            1 if torch.cuda.is_available() else 0
        ),
    }


def evaluate_with_threshold_search(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, Any]:
    val_out = evaluate_loader(model=model, loader=val_loader, criterion=None, device=device)
    test_out = evaluate_loader(model=model, loader=test_loader, criterion=None, device=device)

    best_thr, val_best, _ = find_best_threshold_by_f1(
        y_true=val_out["y_true"],
        y_prob=val_out["y_prob"],
        t_min=cfg.threshold_grid_min,
        t_max=cfg.threshold_grid_max,
        t_steps=cfg.threshold_grid_steps,
    )
    test_best = compute_binary_metrics(test_out["y_true"], test_out["y_prob"], threshold=best_thr)
    test_05 = compute_binary_metrics(test_out["y_true"], test_out["y_prob"], threshold=0.5)

    return {
        "best_threshold_from_val": float(best_thr),
        "val_metrics_at_best_threshold": val_best,
        "test_metrics_at_best_threshold": test_best,
        "test_metrics_at_0_5": test_05,
        "val_y_true": val_out["y_true"],
        "val_y_prob": val_out["y_prob"],
        "test_y_true": test_out["y_true"],
        "test_y_prob": test_out["y_prob"],
    }
