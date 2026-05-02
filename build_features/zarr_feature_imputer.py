"""Spatial imputation utilities for feature Zarr stores.

Imputation is applied independently to each 2D grid using the following
sequence:
1. Local 3x3 kernel-mean filling
2. `rasterio.fill.fillnodata` interpolation when available
3. Nearest-neighbor spatial propagation for any remaining gaps
4. Final global replacement to guarantee finite output

Dynamic feature arrays are processed day by day for `[time, row, col]`
inputs. Static feature arrays are processed once for `[row, col]` inputs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
import zarr
try:
    from rasterio.fill import fillnodata as _rio_fillnodata
    HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _rio_fillnodata = None
    HAS_RASTERIO = False

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class ImputeConfig:
    """Configuration for feature-grid imputation and diagnostic export."""

    source_dir: Path
    output_dir: Path
    stats_dir: Path

    include_features: tuple[str, ...] | None = None
    exclude_features: tuple[str, ...] = tuple()

    kernel_size: int = 3
    max_passes: int = 1
    use_rasterio_fillnodata: bool = True
    rasterio_max_search_distance: float = 100.0
    rasterio_smoothing_iterations: int = 0
    nearest_fill_fallback: bool = True
    ensure_no_nan: bool = True

    n_jobs: int = 1
    show_progress: bool = True
    copy_manifest_if_present: bool = True


def _progress(iterable, *, enabled: bool, **kwargs):
    """Return a tqdm-wrapped iterator when progress display is enabled."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _discover_features(
    source_dir: Path,
    include_features: Iterable[str] | None = None,
    exclude_features: Iterable[str] | None = None,
) -> list[Path]:
    """Enumerate feature Zarr directories selected for imputation."""
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    include = {x.strip() for x in include_features} if include_features else None
    exclude = {x.strip() for x in exclude_features} if exclude_features else set()

    feats: list[Path] = []
    for p in sorted(source_dir.glob("*.zarr")):
        name = p.stem
        if include is not None and name not in include:
            continue
        if name in exclude:
            continue
        feats.append(p)

    if not feats:
        raise ValueError("No feature zarr files matched include/exclude rules")
    return feats


def _kernel_fill_once_3x3(arr2d: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill missing values from valid entries in a 3x3 neighborhood."""
    if arr2d.ndim != 2:
        raise ValueError("_kernel_fill_once_3x3 expects 2D array")

    a = np.asarray(arr2d, dtype=np.float32)
    nan_mask = ~np.isfinite(a)
    if not nan_mask.any():
        return a, 0

    h, w = a.shape
    pad = np.pad(a, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)

    s = np.zeros((h, w), dtype=np.float64)
    c = np.zeros((h, w), dtype=np.int16)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nb = pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            valid = np.isfinite(nb)
            s += np.where(valid, nb, 0.0)
            c += valid.astype(np.int16)

    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.divide(s, c, out=np.full((h, w), np.nan, dtype=np.float64), where=c > 0)

    fill_mask = nan_mask & (c > 0)
    out = a.copy()
    out[fill_mask] = m[fill_mask].astype(np.float32)
    return out, int(fill_mask.sum())


def _kernel_fill_2d(arr2d: np.ndarray, max_passes: int) -> tuple[np.ndarray, int]:
    """Apply repeated 3x3 kernel-mean filling passes to a 2D grid."""
    out = np.asarray(arr2d, dtype=np.float32)
    total_filled = 0
    for _ in range(max(1, int(max_passes))):
        out, filled = _kernel_fill_once_3x3(out)
        total_filled += int(filled)
        if filled == 0:
            break
    return out, total_filled


def _rasterio_fill_2d(
    arr2d: np.ndarray,
    *,
    max_search_distance: float,
    smoothing_iterations: int,
) -> tuple[np.ndarray, int]:
    """Fill missing values using `rasterio.fill.fillnodata` interpolation."""
    if not HAS_RASTERIO or _rio_fillnodata is None:
        return np.asarray(arr2d, dtype=np.float32), 0

    a = np.asarray(arr2d, dtype=np.float32)
    nan_before = int((~np.isfinite(a)).sum())
    if nan_before == 0:
        return a, 0

    img = np.where(np.isfinite(a), a, 0.0).astype(np.float32, copy=False)
    # In GDAL/rasterio semantics, mask > 0 marks valid source pixels.
    mask = np.isfinite(a).astype(np.uint8)
    out = _rio_fillnodata(
        image=img,
        mask=mask,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=int(smoothing_iterations),
    )
    out = np.asarray(out, dtype=np.float32)
    nan_after = int((~np.isfinite(out)).sum())
    filled = max(nan_before - nan_after, 0)
    return out, int(filled)


def _nearest_propagation_fill(arr2d: np.ndarray, max_iters: int | None = None) -> tuple[np.ndarray, int, int]:
    """Fill remaining missing values by iterative nearest-neighbor propagation."""
    out = np.asarray(arr2d, dtype=np.float32).copy()
    h, w = out.shape
    if max_iters is None:
        max_iters = int(h + w)

    nan0 = int((~np.isfinite(out)).sum())
    if nan0 == 0:
        return out, 0, 0

    iters = 0
    for _ in range(max_iters):
        nan_mask = ~np.isfinite(out)
        if not nan_mask.any():
            break

        pad = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
        candidate = np.full_like(out, np.nan, dtype=np.float32)
        # The neighborhood order is fixed to keep the propagation deterministic.
        for dr, dc in ((0, -1), (-1, 0), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nb = pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            take = np.isnan(candidate) & np.isfinite(nb)
            candidate[take] = nb[take]

        fill_mask = nan_mask & np.isfinite(candidate)
        n_filled = int(fill_mask.sum())
        if n_filled == 0:
            break
        out[fill_mask] = candidate[fill_mask]
        iters += 1

    nan1 = int((~np.isfinite(out)).sum())
    return out, int(max(nan0 - nan1, 0)), int(iters)


def _final_global_fill(arr2d: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Guarantee finite output by replacing any remaining gaps with a global mean."""
    out = np.asarray(arr2d, dtype=np.float32).copy()
    nan_mask = ~np.isfinite(out)
    n_nan = int(nan_mask.sum())
    if n_nan == 0:
        return out, 0, float("nan")

    finite = out[np.isfinite(out)]
    if finite.size == 0:
        fallback = 0.0
    else:
        fallback = float(np.mean(finite, dtype=np.float64))
    out[nan_mask] = np.float32(fallback)
    return out, n_nan, float(fallback)


def _impute_grid_2d(arr2d: np.ndarray, cfg: ImputeConfig) -> tuple[np.ndarray, dict]:
    """Run the full imputation sequence and return the filled grid with diagnostics."""
    a = np.asarray(arr2d, dtype=np.float32)
    # Normalize all non-finite values to NaN before running the imputation stages.
    a = np.where(np.isfinite(a), a, np.nan).astype(np.float32, copy=False)
    nan_before = int((~np.isfinite(a)).sum())
    out = a.copy()

    # Stage 1: local kernel-mean filling
    out, filled_kernel = _kernel_fill_2d(out, max_passes=cfg.max_passes)

    # Stage 2: raster-based interpolation
    filled_rasterio = 0
    if cfg.use_rasterio_fillnodata:
        out, filled_rasterio = _rasterio_fill_2d(
            out,
            max_search_distance=cfg.rasterio_max_search_distance,
            smoothing_iterations=cfg.rasterio_smoothing_iterations,
        )

    # Stage 3: nearest-neighbor propagation
    filled_nearest = 0
    nearest_iters = 0
    if cfg.nearest_fill_fallback and (~np.isfinite(out)).any():
        out, filled_nearest, nearest_iters = _nearest_propagation_fill(out)

    # Stage 4: final global replacement to guarantee finite output
    filled_global = 0
    global_fallback = float("nan")
    if cfg.ensure_no_nan and (~np.isfinite(out)).any():
        out, filled_global, global_fallback = _final_global_fill(out)

    nan_after = int((~np.isfinite(out)).sum())
    details = {
        "nan_before": int(nan_before),
        "nan_after": int(nan_after),
        "filled_kernel": int(filled_kernel),
        "filled_rasterio": int(filled_rasterio),
        "filled_nearest": int(filled_nearest),
        "filled_global": int(filled_global),
        "nearest_iters": int(nearest_iters),
        "used_rasterio": bool(cfg.use_rasterio_fillnodata and HAS_RASTERIO),
        "global_fallback_value": float(global_fallback),
    }
    return out, details


def _copy_timestamp_if_present(src: zarr.Group, dst: zarr.Group) -> None:
    """Copy the timestamp array to the output store when it is available."""
    if "timestamp" not in src:
        return
    ts = src["timestamp"]
    ts_data = np.asarray(ts[:])
    kwargs = {
        "shape": ts_data.shape,
        "dtype": ts_data.dtype,
        "overwrite": True,
    }
    if getattr(ts, "chunks", None) is not None:
        kwargs["chunks"] = ts.chunks
    dst_ts = dst.create_dataset("timestamp", **kwargs)
    dst_ts[:] = ts_data


def _copy_attrs(src: zarr.Group, dst: zarr.Group) -> None:
    """Copy Zarr group attributes when they can be read safely."""
    try:
        dst.attrs.update(dict(src.attrs))
    except Exception:
        pass


def impute_one_feature(feature_path: Path, cfg: ImputeConfig) -> tuple[dict, pd.DataFrame]:
    """Impute one feature store and return summary and per-day diagnostics."""
    src = zarr.open_group(str(feature_path), mode="r")
    if "value" not in src:
        raise ValueError(f"{feature_path.name} missing 'value' array")

    value = src["value"]
    out_feature = cfg.output_dir / feature_path.name
    out_feature.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_group(str(out_feature), mode="w")

    _copy_attrs(src, dst)
    _copy_timestamp_if_present(src, dst)

    dtype_out = np.float32
    create_kwargs = {
        "shape": value.shape,
        "dtype": dtype_out,
        "overwrite": True,
    }
    if getattr(value, "chunks", None) is not None:
        create_kwargs["chunks"] = value.chunks
    dst_value = dst.create_dataset("value", **create_kwargs)

    rows = []

    if value.ndim == 3:
        t, h, w = value.shape
        it = range(t)
        it = _progress(it, enabled=cfg.show_progress, desc=f"impute::{feature_path.stem}", unit="day")

        total_before = 0
        total_after = 0
        total_filled = 0
        total_filled_kernel = 0
        total_filled_rasterio = 0
        total_filled_nearest = 0
        total_filled_global = 0
        nearest_iters_total = 0
        days_with_nan_before = 0
        days_with_nan_after = 0
        days_changed = 0

        for ti in it:
            arr = np.asarray(value[ti, :, :], dtype=np.float32)
            out, det = _impute_grid_2d(arr, cfg=cfg)
            nan_before = int(det["nan_before"])
            nan_after = int(det["nan_after"])
            filled = int(max(nan_before - nan_after, 0))
            dst_value[ti, :, :] = out

            total_before += nan_before
            total_after += nan_after
            total_filled += int(filled)
            total_filled_kernel += int(det["filled_kernel"])
            total_filled_rasterio += int(det["filled_rasterio"])
            total_filled_nearest += int(det["filled_nearest"])
            total_filled_global += int(det["filled_global"])
            nearest_iters_total += int(det["nearest_iters"])
            days_with_nan_before += int(nan_before > 0)
            days_with_nan_after += int(nan_after > 0)
            days_changed += int(filled > 0)

            rows.append(
                {
                    "feature": feature_path.stem,
                    "day_index": int(ti),
                    "nan_before": nan_before,
                    "nan_after": nan_after,
                    "filled": int(filled),
                    "changed": int(filled > 0),
                    "filled_kernel": int(det["filled_kernel"]),
                    "filled_rasterio": int(det["filled_rasterio"]),
                    "filled_nearest": int(det["filled_nearest"]),
                    "filled_global": int(det["filled_global"]),
                    "nearest_iters": int(det["nearest_iters"]),
                }
            )

        summary = {
            "feature": feature_path.stem,
            "source": str(feature_path),
            "output": str(out_feature),
            "kind": "dynamic",
            "shape": [int(t), int(h), int(w)],
            "kernel": int(cfg.kernel_size),
            "max_passes": int(cfg.max_passes),
            "cells_total": int(t * h * w),
            "nan_before_total": int(total_before),
            "nan_after_total": int(total_after),
            "filled_total": int(total_filled),
            "filled_kernel_total": int(total_filled_kernel),
            "filled_rasterio_total": int(total_filled_rasterio),
            "filled_nearest_total": int(total_filled_nearest),
            "filled_global_total": int(total_filled_global),
            "nearest_iters_total": int(nearest_iters_total),
            "used_rasterio": bool(cfg.use_rasterio_fillnodata and HAS_RASTERIO),
            "fill_rate_pct_of_nan_before": float(100.0 * total_filled / max(total_before, 1)),
            "days_with_nan_before": int(days_with_nan_before),
            "days_with_nan_after": int(days_with_nan_after),
            "days_changed": int(days_changed),
        }

    elif value.ndim == 2:
        h, w = value.shape
        arr = np.asarray(value[:, :], dtype=np.float32)
        out, det = _impute_grid_2d(arr, cfg=cfg)
        nan_before = int(det["nan_before"])
        nan_after = int(det["nan_after"])
        filled = int(max(nan_before - nan_after, 0))
        dst_value[:, :] = out

        rows.append(
            {
                "feature": feature_path.stem,
                "day_index": -1,
                "nan_before": nan_before,
                "nan_after": nan_after,
                "filled": int(filled),
                "changed": int(filled > 0),
                "filled_kernel": int(det["filled_kernel"]),
                "filled_rasterio": int(det["filled_rasterio"]),
                "filled_nearest": int(det["filled_nearest"]),
                "filled_global": int(det["filled_global"]),
                "nearest_iters": int(det["nearest_iters"]),
            }
        )

        summary = {
            "feature": feature_path.stem,
            "source": str(feature_path),
            "output": str(out_feature),
            "kind": "static",
            "shape": [int(h), int(w)],
            "kernel": int(cfg.kernel_size),
            "max_passes": int(cfg.max_passes),
            "cells_total": int(h * w),
            "nan_before_total": int(nan_before),
            "nan_after_total": int(nan_after),
            "filled_total": int(filled),
            "filled_kernel_total": int(det["filled_kernel"]),
            "filled_rasterio_total": int(det["filled_rasterio"]),
            "filled_nearest_total": int(det["filled_nearest"]),
            "filled_global_total": int(det["filled_global"]),
            "nearest_iters_total": int(det["nearest_iters"]),
            "used_rasterio": bool(cfg.use_rasterio_fillnodata and HAS_RASTERIO),
            "fill_rate_pct_of_nan_before": float(100.0 * filled / max(nan_before, 1)),
            "days_with_nan_before": int(nan_before > 0),
            "days_with_nan_after": int(nan_after > 0),
            "days_changed": int(filled > 0),
        }

    else:
        raise ValueError(f"Unsupported value.ndim={value.ndim} for {feature_path.name}")

    day_stats = pd.DataFrame(rows)
    return summary, day_stats


def run_imputation(cfg: ImputeConfig) -> dict[str, str]:
    """Run imputation for all selected features and write diagnostic outputs."""
    if int(cfg.kernel_size) != 3:
        raise ValueError("Only kernel_size=3 is supported in this implementation")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.stats_dir.mkdir(parents=True, exist_ok=True)

    features = _discover_features(
        source_dir=cfg.source_dir,
        include_features=cfg.include_features,
        exclude_features=cfg.exclude_features,
    )

    summaries: list[dict] = []
    day_stats_all: list[pd.DataFrame] = []

    n_jobs = max(1, int(cfg.n_jobs))
    if n_jobs == 1 or len(features) <= 1:
        iterator = _progress(features, enabled=cfg.show_progress, desc="features", unit="feat")
        for fp in iterator:
            s, d = impute_one_feature(fp, cfg)
            summaries.append(s)
            day_stats_all.append(d)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            fut_map = {ex.submit(impute_one_feature, fp, cfg): fp.stem for fp in features}
            iterator = as_completed(fut_map)
            iterator = _progress(iterator, enabled=cfg.show_progress, desc="features(parallel)", unit="feat", total=len(fut_map))
            for fut in iterator:
                s, d = fut.result()
                summaries.append(s)
                day_stats_all.append(d)

    summary_df = pd.DataFrame(summaries).sort_values("feature").reset_index(drop=True)
    day_df = pd.concat(day_stats_all, ignore_index=True) if day_stats_all else pd.DataFrame()

    p_summary_csv = cfg.stats_dir / "feature_impute_summary.csv"
    p_summary_json = cfg.stats_dir / "feature_impute_summary.json"
    p_day_csv = cfg.stats_dir / "feature_impute_day_stats.csv"

    summary_df.to_csv(p_summary_csv, index=False)
    p_summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    day_df.to_csv(p_day_csv, index=False)

    if cfg.copy_manifest_if_present:
        src_manifest = cfg.source_dir / "feature_manifest.json"
        if src_manifest.exists():
            manifest = json.loads(src_manifest.read_text(encoding="utf-8"))
            manifest_out = cfg.output_dir / "feature_manifest.json"
            manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_meta = {
        "source_dir": str(cfg.source_dir),
        "output_dir": str(cfg.output_dir),
        "stats_dir": str(cfg.stats_dir),
        "n_features": int(len(summary_df)),
        "kernel_size": int(cfg.kernel_size),
        "max_passes": int(cfg.max_passes),
        "use_rasterio_fillnodata": bool(cfg.use_rasterio_fillnodata),
        "rasterio_available": bool(HAS_RASTERIO),
        "rasterio_max_search_distance": float(cfg.rasterio_max_search_distance),
        "rasterio_smoothing_iterations": int(cfg.rasterio_smoothing_iterations),
        "nearest_fill_fallback": bool(cfg.nearest_fill_fallback),
        "ensure_no_nan": bool(cfg.ensure_no_nan),
        "n_jobs": int(cfg.n_jobs),
        "exclude_features": list(cfg.exclude_features),
        "include_features": list(cfg.include_features) if cfg.include_features else None,
        "outputs": {
            "summary_csv": str(p_summary_csv),
            "summary_json": str(p_summary_json),
            "day_stats_csv": str(p_day_csv),
        },
    }

    p_run = cfg.stats_dir / "impute_run_metadata.json"
    p_run.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return {
        "summary_csv": str(p_summary_csv),
        "summary_json": str(p_summary_json),
        "day_stats_csv": str(p_day_csv),
        "run_metadata": str(p_run),
        "output_dir": str(cfg.output_dir),
    }
