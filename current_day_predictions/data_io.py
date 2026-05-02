from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


KEY_COLS = ["target_date", "row", "col"]
X_META_COLS = {"target_date", "row", "col", "window_id", "split"}


@dataclass
class SplitArrays:
    x: np.ndarray  # [N, T, F], float32
    y: np.ndarray  # [N], float32
    keys: pd.DataFrame  # target_date,row,col,split,type_class,label
    feature_cols: list[str]
    split_name: str


class XYDataset(Dataset):
    def __init__(self, arrays: SplitArrays):
        self.x = torch.as_tensor(arrays.x, dtype=torch.float32)
        self.y = torch.as_tensor(arrays.y, dtype=torch.float32)
        self.keys = arrays.keys.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.keys.iloc[idx]
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "target_date": row["target_date"],
            "row": int(row["row"]),
            "col": int(row["col"]),
            "split": str(row["split"]),
            "type_class": str(row["type_class"]),
        }


class FeatureNormalizer:
    """Per-feature standardization over train split only."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "FeatureNormalizer":
        # x_train: [N,T,F]
        if x_train.ndim != 3:
            raise ValueError(f"x_train must be 3D [N,T,F], got {x_train.shape}")
        flat = x_train.reshape(-1, x_train.shape[-1]).astype(np.float64)
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fit before transform")
        out = (x - self.mean_[None, None, :]) / self.std_[None, None, :]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fit")
        return {
            "mean": self.mean_.astype(float).tolist(),
            "std": self.std_.astype(float).tolist(),
        }


def _assert_columns(df: pd.DataFrame, cols: list[str], file_path: str | Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} missing columns: {missing}")


def _filter_complete_window_keys(x_df: pd.DataFrame, expected_windows: int) -> pd.DataFrame:
    x = x_df.copy()
    x["window_id"] = pd.to_numeric(x["window_id"], errors="coerce").astype("Int64")
    x = x[x["window_id"].notna()].copy()
    x["window_id"] = x["window_id"].astype(np.int32)
    x = x[(x["window_id"] >= 1) & (x["window_id"] <= expected_windows)].copy()

    gb = x.groupby(KEY_COLS, sort=False)["window_id"]
    nunique = gb.nunique()
    wmin = gb.min()
    wmax = gb.max()
    complete = (nunique == expected_windows) & (wmin == 1) & (wmax == expected_windows)

    keep_keys = complete[complete].index.to_frame(index=False)
    out = x.merge(keep_keys, on=KEY_COLS, how="inner")
    return out


def _pivot_x_to_tensor(
    x_df: pd.DataFrame,
    feature_cols: list[str],
    expected_windows: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_df = x_df.copy()
    x_df = x_df.drop_duplicates(subset=KEY_COLS + ["window_id"], keep="first")
    pivot = x_df.pivot(index=KEY_COLS, columns="window_id", values=feature_cols)

    key_df = pivot.index.to_frame(index=False).reset_index(drop=True)
    n = len(key_df)
    f = len(feature_cols)
    t = int(expected_windows)
    out = np.full((n, t, f), np.nan, dtype=np.float32)

    for wi in range(1, t + 1):
        block = pivot.xs(wi, level=1, axis=1)
        block = block.reindex(columns=feature_cols)
        out[:, wi - 1, :] = block.to_numpy(dtype=np.float32)
    return out, key_df


def load_split_xy_csv(
    x_csv: str | Path,
    y_csv: str | Path,
    split_name: str,
    expected_windows: int = 4,
) -> SplitArrays:
    """Load one split X/Y CSV and return aligned arrays for training."""
    x_path = Path(x_csv)
    y_path = Path(y_csv)
    if not x_path.exists():
        raise FileNotFoundError(x_path)
    if not y_path.exists():
        raise FileNotFoundError(y_path)

    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    _assert_columns(x_df, KEY_COLS + ["window_id"], x_path)
    _assert_columns(y_df, KEY_COLS + ["label", "type_class", "split"], y_path)
    expected_split = str(split_name).strip().lower()
    found_splits = y_df["split"].astype(str).str.strip().str.lower()
    if not found_splits.eq(expected_split).all():
        found_unique = sorted(found_splits.dropna().unique().tolist())
        raise ValueError(
            f"Split mismatch in {y_path}: expected split='{expected_split}', "
            f"found split values={found_unique}"
        )

    feature_cols = [c for c in x_df.columns if c not in X_META_COLS]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {x_path}")

    # Ensure numeric feature matrix and deterministic key ordering.
    for c in feature_cols:
        x_df[c] = pd.to_numeric(x_df[c], errors="coerce")

    x_df = _filter_complete_window_keys(x_df, expected_windows=expected_windows)
    if x_df.empty:
        raise ValueError(f"No complete-window keys found in {x_path}")

    x_df = x_df.sort_values(KEY_COLS + ["window_id"]).reset_index(drop=True)
    x_tensor, key_df = _pivot_x_to_tensor(x_df, feature_cols=feature_cols, expected_windows=expected_windows)
    key_df["_x_idx"] = np.arange(len(key_df), dtype=np.int64)

    y_use = y_df[KEY_COLS + ["label", "type_class", "split"]].copy()
    y_use = y_use.drop_duplicates(KEY_COLS, keep="first")
    y_use = y_use.merge(key_df, on=KEY_COLS, how="inner")
    if y_use.empty:
        raise ValueError(f"No overlapping keys between X and Y for split={split_name}")

    y_use = y_use.sort_values("_x_idx").reset_index(drop=True)
    x_aligned = x_tensor[y_use["_x_idx"].to_numpy(dtype=np.int64)]
    y_arr = pd.to_numeric(y_use["label"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    keys = y_use[KEY_COLS + ["split", "type_class", "label"]].copy().reset_index(drop=True)
    arrays = SplitArrays(
        x=x_aligned.astype(np.float32),
        y=y_arr,
        keys=keys,
        feature_cols=feature_cols,
        split_name=split_name,
    )
    return arrays


def make_torch_datasets(
    train_arrays: SplitArrays,
    val_arrays: SplitArrays,
    test_arrays: SplitArrays,
) -> tuple[XYDataset, XYDataset, XYDataset]:
    return XYDataset(train_arrays), XYDataset(val_arrays), XYDataset(test_arrays)


def download_hf_split_files(
    x_repo_id: str,
    y_repo_id: str,
    x_files: dict[str, str],
    y_files: dict[str, str],
    token: str | None = None,
    repo_type: str = "dataset",
    cache_dir: str | Path | None = None,
) -> dict[str, str]:
    """Download split CSV files from Hugging Face Hub.

    Returns keys:
      x_train, x_val, x_test, y_train, y_val, y_test
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise ImportError("huggingface_hub is required for HF download") from e

    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        out[f"x_{split}"] = hf_hub_download(
            repo_id=x_repo_id,
            filename=x_files[split],
            repo_type=repo_type,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        out[f"y_{split}"] = hf_hub_download(
            repo_id=y_repo_id,
            filename=y_files[split],
            repo_type=repo_type,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    return out
