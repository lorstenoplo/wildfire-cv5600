"""Feature-to-grid utilities using the same projected 4 km grid as labels.

This module reuses label grid metadata (CRS, affine origin, cell size, rows/cols)
and rasterizes feature parquet files onto that exact fixed grid.

Outputs are designed for separate-per-feature Zarr stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
import zarr

DAY_SEC = 86_400


@dataclass(frozen=True)
class GridMeta:
    source_crs: str
    projected_crs: str
    cell_size_m: float
    rows: int
    cols: int
    x_min: float
    y_max: float

    @property
    def n_cells(self) -> int:
        return int(self.rows * self.cols)


def load_grid_meta(metadata_json_path: str | Path) -> GridMeta:
    """Load fixed label-grid metadata."""
    path = Path(metadata_json_path)
    meta = json.loads(path.read_text(encoding="utf-8"))

    shape = meta["grid_shape_rows_cols"]
    tfm = meta["affine_transform_projected"]

    return GridMeta(
        source_crs=str(meta["source_crs"]),
        projected_crs=str(meta["projected_crs"]),
        cell_size_m=float(meta["cell_size_m"]),
        rows=int(shape[0]),
        cols=int(shape[1]),
        x_min=float(tfm["c"]),
        y_max=float(tfm["f"]),
    )


def load_label_timestamps(labels_zarr_path: str | Path) -> np.ndarray:
    """Read label timestamps [N] from labels.zarr."""
    z = zarr.open(str(labels_zarr_path), mode="r")
    if "timestamp" not in z:
        raise KeyError("labels.zarr missing 'timestamp' array")
    ts = np.asarray(z["timestamp"][:], dtype=np.int64)
    if ts.ndim != 1:
        raise ValueError(f"timestamp must be 1D; got shape={ts.shape}")
    return ts


def to_epoch_day_seconds(values: pd.Series | pd.Index | np.ndarray) -> np.ndarray:
    """Convert datetime-like values to UTC day-start epoch seconds."""
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    # pd.to_datetime returns Series for Series input and DatetimeIndex otherwise.
    # Use the matching floor accessor, then convert with NaT-safe handling.
    if isinstance(dt, pd.Series):
        dt = dt.dt.floor("D")
        dt64 = dt.to_numpy(dtype="datetime64[ns]")
    else:
        dt = dt.floor("D")
        dt64 = dt.to_numpy(dtype="datetime64[ns]")

    ns = dt64.astype("int64")
    out = ns.astype(np.float64) / 1_000_000_000.0
    nat_sentinel = np.iinfo(np.int64).min
    out[ns == nat_sentinel] = np.nan
    return out


def _row_col_from_lat_lon(
    lat: np.ndarray,
    lon: np.ndarray,
    grid: GridMeta,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project lat/lon to grid row/col and return valid mask."""
    x, y = transformer.transform(lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    col = np.floor((x - grid.x_min) / grid.cell_size_m).astype(np.int32)
    row = np.floor((grid.y_max - y) / grid.cell_size_m).astype(np.int32)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (row >= 0)
        & (row < grid.rows)
        & (col >= 0)
        & (col < grid.cols)
    )
    return row, col, valid


def _iter_batches(
    parquet_path: str | Path, columns: list[str], batch_size: int
) -> Iterable[pd.DataFrame]:
    """Yield pandas DataFrame batches from parquet with selected columns only."""
    pqf = pq.ParquetFile(str(parquet_path))
    for record_batch in pqf.iter_batches(batch_size=batch_size, columns=columns):
        yield record_batch.to_pandas()


def rasterize_dynamic_feature(
    parquet_path: str | Path,
    feature_col: str,
    time_col: str,
    lat_col: str,
    lon_col: str,
    target_timestamps: np.ndarray,
    grid: GridMeta,
    batch_size: int = 500_000,
) -> np.ndarray:
    """Rasterize one time-varying feature to [T,H,W] using mean aggregation per cell-day."""
    target_ts = np.asarray(target_timestamps, dtype=np.int64)
    if target_ts.ndim != 1:
        raise ValueError("target_timestamps must be 1D")
    if target_ts.size == 0:
        raise ValueError("target_timestamps is empty")
    if not np.all(target_ts[:-1] <= target_ts[1:]):
        raise ValueError("target_timestamps must be sorted ascending")

    n_time = int(target_ts.size)
    n_cells = grid.n_cells

    sums = np.zeros(n_time * n_cells, dtype=np.float32)
    counts = np.zeros(n_time * n_cells, dtype=np.uint16)

    transformer = Transformer.from_crs(
        grid.source_crs, grid.projected_crs, always_xy=True
    )
    cols = [time_col, lat_col, lon_col, feature_col]

    for batch in _iter_batches(parquet_path, columns=cols, batch_size=batch_size):
        if batch.empty:
            continue

        ts = to_epoch_day_seconds(batch[time_col])
        lat = pd.to_numeric(batch[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(batch[lon_col], errors="coerce").to_numpy(dtype=np.float64)
        val = pd.to_numeric(batch[feature_col], errors="coerce").to_numpy(
            dtype=np.float32
        )

        ok = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
        if not np.any(ok):
            continue

        ts = ts[ok]
        lat = lat[ok]
        lon = lon[ok]
        val = val[ok]

        t_idx = np.searchsorted(target_ts, ts)
        ok_t = (t_idx < n_time) & (target_ts[t_idx] == ts)
        if not np.any(ok_t):
            continue

        t_idx = t_idx[ok_t]
        lat = lat[ok_t]
        lon = lon[ok_t]
        val = val[ok_t]

        row, col, ok_grid = _row_col_from_lat_lon(
            lat, lon, grid=grid, transformer=transformer
        )
        if not np.any(ok_grid):
            continue

        t_idx = t_idx[ok_grid].astype(np.int64)
        flat = row[ok_grid].astype(np.int64) * grid.cols + col[ok_grid].astype(np.int64)
        v = val[ok_grid].astype(np.float32)

        linear = t_idx * n_cells + flat
        np.add.at(sums, linear, v)
        np.add.at(counts, linear, 1)

    out = np.full(n_time * n_cells, np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out.reshape(n_time, grid.rows, grid.cols)


def rasterize_static_feature(
    parquet_path: str | Path,
    feature_col: str,
    lat_col: str,
    lon_col: str,
    grid: GridMeta,
    batch_size: int = 500_000,
) -> np.ndarray:
    """Rasterize one static feature to [H,W] using mean aggregation per cell."""
    n_cells = grid.n_cells
    sums = np.zeros(n_cells, dtype=np.float32)
    counts = np.zeros(n_cells, dtype=np.uint32)

    transformer = Transformer.from_crs(
        grid.source_crs, grid.projected_crs, always_xy=True
    )
    cols = [lat_col, lon_col, feature_col]

    for batch in _iter_batches(parquet_path, columns=cols, batch_size=batch_size):
        if batch.empty:
            continue

        lat = pd.to_numeric(batch[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(batch[lon_col], errors="coerce").to_numpy(dtype=np.float64)
        val = pd.to_numeric(batch[feature_col], errors="coerce").to_numpy(
            dtype=np.float32
        )

        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
        if not np.any(ok):
            continue

        lat = lat[ok]
        lon = lon[ok]
        val = val[ok]

        row, col, ok_grid = _row_col_from_lat_lon(
            lat, lon, grid=grid, transformer=transformer
        )
        if not np.any(ok_grid):
            continue

        flat = row[ok_grid].astype(np.int64) * grid.cols + col[ok_grid].astype(np.int64)
        v = val[ok_grid].astype(np.float32)

        np.add.at(sums, flat, v)
        np.add.at(counts, flat, 1)

    out = np.full(n_cells, np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out.reshape(grid.rows, grid.cols)


def rasterize_monthly_static_to_daily(
    parquet_path: str | Path,
    value_col: str,
    month_col: str,
    lat_col: str,
    lon_col: str,
    target_timestamps: np.ndarray,
    grid: GridMeta,
    batch_size: int = 500_000,
) -> np.ndarray:
    """Rasterize (lat,lon,month,value) climatology and expand to daily [T,H,W] by month."""
    target_ts = np.asarray(target_timestamps, dtype=np.int64)
    n_cells = grid.n_cells

    sums = np.zeros(12 * n_cells, dtype=np.float32)
    counts = np.zeros(12 * n_cells, dtype=np.uint32)

    transformer = Transformer.from_crs(
        grid.source_crs, grid.projected_crs, always_xy=True
    )
    cols = [lat_col, lon_col, month_col, value_col]

    for batch in _iter_batches(parquet_path, columns=cols, batch_size=batch_size):
        if batch.empty:
            continue

        lat = pd.to_numeric(batch[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(batch[lon_col], errors="coerce").to_numpy(dtype=np.float64)
        month = pd.to_numeric(batch[month_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        val = pd.to_numeric(batch[value_col], errors="coerce").to_numpy(
            dtype=np.float32
        )

        ok = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(month)
            & np.isfinite(val)
            & (month >= 1)
            & (month <= 12)
        )
        if not np.any(ok):
            continue

        lat = lat[ok]
        lon = lon[ok]
        month = month[ok].astype(np.int64)
        val = val[ok]

        row, col, ok_grid = _row_col_from_lat_lon(
            lat, lon, grid=grid, transformer=transformer
        )
        if not np.any(ok_grid):
            continue

        month = month[ok_grid]
        flat = row[ok_grid].astype(np.int64) * grid.cols + col[ok_grid].astype(np.int64)
        v = val[ok_grid].astype(np.float32)

        linear = (month - 1) * n_cells + flat
        np.add.at(sums, linear, v)
        np.add.at(counts, linear, 1)

    monthly = np.full(12 * n_cells, np.nan, dtype=np.float32)
    valid = counts > 0
    monthly[valid] = sums[valid] / counts[valid]
    monthly = monthly.reshape(12, grid.rows, grid.cols)

    months = pd.to_datetime(target_ts, unit="s", utc=True).month.to_numpy(
        dtype=np.int16
    )
    out = monthly[months - 1]
    return out.astype(np.float32)


def write_feature_zarr_dynamic(
    out_zarr_path: str | Path,
    feature_name: str,
    target_timestamps: np.ndarray,
    values_t_hw: np.ndarray,
    grid: GridMeta,
    source_name: str,
    chunks_t: int = 32,
) -> None:
    """Write a dynamic feature [T,H,W] to a standalone Zarr group."""
    out = Path(out_zarr_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(target_timestamps, dtype=np.int64)
    v = np.asarray(values_t_hw, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"dynamic values must be 3D [T,H,W], got shape={v.shape}")
    if v.shape[0] != t.shape[0] or v.shape[1] != grid.rows or v.shape[2] != grid.cols:
        raise ValueError(
            f"shape mismatch for feature={feature_name}: values={v.shape}, expected=({t.size},{grid.rows},{grid.cols})"
        )

    z = zarr.open_group(str(out), mode="w")
    z.create_array("timestamp", data=t, chunks=(min(1024, t.size),))
    z.create_array(
        "value",
        data=v,
        chunks=(min(max(1, chunks_t), t.size), grid.rows, grid.cols),
    )

    z.attrs.update(
        {
            "feature_name": feature_name,
            "kind": "dynamic",
            "source_name": source_name,
            "source_crs": grid.source_crs,
            "projected_crs": grid.projected_crs,
            "cell_size_m": float(grid.cell_size_m),
            "rows": int(grid.rows),
            "cols": int(grid.cols),
            "notes": "same fixed grid as labels",
        }
    )


def write_feature_zarr_static(
    out_zarr_path: str | Path,
    feature_name: str,
    values_hw: np.ndarray,
    grid: GridMeta,
    source_name: str,
) -> None:
    """Write a static feature [H,W] to a standalone Zarr group."""
    out = Path(out_zarr_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(values_hw, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"static values must be 2D [H,W], got shape={v.shape}")
    if v.shape[0] != grid.rows or v.shape[1] != grid.cols:
        raise ValueError(
            f"shape mismatch for feature={feature_name}: values={v.shape}, expected=({grid.rows},{grid.cols})"
        )

    z = zarr.open_group(str(out), mode="w")
    z.create_array("value", data=v, chunks=(grid.rows, grid.cols))

    z.attrs.update(
        {
            "feature_name": feature_name,
            "kind": "static",
            "source_name": source_name,
            "source_crs": grid.source_crs,
            "projected_crs": grid.projected_crs,
            "cell_size_m": float(grid.cell_size_m),
            "rows": int(grid.rows),
            "cols": int(grid.cols),
            "notes": "same fixed grid as labels",
        }
    )
