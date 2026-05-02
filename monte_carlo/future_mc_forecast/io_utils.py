from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def validate_tensor_shapes(
    history_features: torch.Tensor | np.ndarray,
    current_features: torch.Tensor | np.ndarray,
    current_probability: torch.Tensor | np.ndarray | float,
    expected_history_days: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Validate and canonicalize inputs to [F,64,H,W], [F,H,W], [H,W].

    Supports cell-wise inference by accepting:
      - history_features: [F, 64]
      - current_features: [F]
      - current_probability: scalar
    and promoting to H=W=1.
    """
    history = torch.as_tensor(history_features, dtype=torch.float32)
    current = torch.as_tensor(current_features, dtype=torch.float32)
    prob = torch.as_tensor(current_probability, dtype=torch.float32)

    is_cellwise_input = False

    if history.ndim == 2:
        is_cellwise_input = True
        history = history.unsqueeze(-1).unsqueeze(-1)
    elif history.ndim != 4:
        raise ValueError(
            "history_features must be [F,64,H,W] or [F,64], "
            f"got shape={tuple(history.shape)}"
        )

    f, history_days, h, w = history.shape
    if history_days != expected_history_days:
        raise ValueError(
            f"history_features second dimension must be {expected_history_days}, got {history_days}"
        )

    if current.ndim == 1:
        current = current.unsqueeze(-1).unsqueeze(-1)
    elif current.ndim != 3:
        raise ValueError(
            f"current_features must be [F,H,W] or [F], got shape={tuple(current.shape)}"
        )

    if tuple(current.shape) != (f, h, w):
        raise ValueError(
            "current_features shape mismatch with history_features. "
            f"expected={(f, h, w)}, got={tuple(current.shape)}"
        )

    if prob.ndim == 0:
        prob = prob.view(1, 1)
    elif prob.ndim == 1 and prob.numel() == 1:
        prob = prob.view(1, 1)
    elif prob.ndim != 2:
        raise ValueError(
            f"current_probability must be [H,W] or scalar, got shape={tuple(prob.shape)}"
        )

    if tuple(prob.shape) != (h, w):
        raise ValueError(
            "current_probability shape mismatch with history/current features. "
            f"expected={(h, w)}, got={tuple(prob.shape)}"
        )

    history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
    current = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

    metadata = {
        "num_features": int(f),
        "history_days": int(history_days),
        "height": int(h),
        "width": int(w),
        "is_cellwise_input": bool(is_cellwise_input or (h == 1 and w == 1)),
    }
    return history, current, prob, metadata


def load_zarr_feature_stack(
    zarr_path: str | Path,
    dataset_key: str | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Load a feature stack from a Zarr array/group."""
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ImportError("zarr is required for load_zarr_feature_stack") from exc

    zpath = Path(zarr_path)
    if not zpath.exists():
        raise FileNotFoundError(zpath)

    root = zarr.open(str(zpath), mode="r")
    if dataset_key is None:
        arr = np.asarray(root, dtype=dtype)
    else:
        if dataset_key not in root:
            raise KeyError(f"dataset_key='{dataset_key}' not found in zarr store {zpath}")
        arr = np.asarray(root[dataset_key], dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def save_numpy_map(array: torch.Tensor | np.ndarray, output_path: str | Path) -> None:
    """Save tensor/array as .npy with parent directory creation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_arr = _to_numpy(array)
    np.save(path, np_arr)


def save_metadata_json(metadata: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON metadata with stable indentation and key order."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def save_zarr_map(
    array: torch.Tensor | np.ndarray,
    output_path: str | Path,
    dataset_key: str | None = None,
) -> None:
    """Optional Zarr saver for interoperability with existing pipelines."""
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ImportError("zarr is required for save_zarr_map") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_numpy(array).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if dataset_key is None:
        zarr.save(str(path), arr)
        return
    group = zarr.open(str(path), mode="a")
    group[dataset_key] = arr


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        arr = array.detach().cpu().numpy()
    else:
        arr = np.asarray(array)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr
