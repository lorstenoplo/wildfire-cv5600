"""Spatial imputation utilities for VIIRS-style band Zarr stores.

Expected input schema per band includes three imputed target arrays:
- `mean`: `[time, row, col]`
- `min`: `[time, row, col]`
- `max`: `[time, row, col]`

Auxiliary arrays such as time, coordinates, valid-observation counts, and
centroid grids are preserved. The output store keeps the same schema and
metadata; only the target data arrays are imputed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Any, Iterable

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
class ViirsImputeConfig:
    """Configuration for VIIRS band imputation and diagnostic export."""

    source_dir: Path
    output_dir: Path
    stats_dir: Path

    include_bands: tuple[str, ...] | None = None
    exclude_bands: tuple[str, ...] = tuple()

    arrays_to_impute: tuple[str, ...] = ("mean", "min", "max")
    preserve_aux_arrays: bool = True

    # Imputation settings
    kernel_size: int = 3
    max_passes: int = 1
    use_rasterio_fillnodata: bool = True
    rasterio_max_search_distance: float = 100.0
    rasterio_smoothing_iterations: int = 0
    nearest_fill_fallback: bool = True
    ensure_no_nan: bool = True

    n_jobs: int = max(1, min(8, (os.cpu_count() or 4)))
    show_progress: bool = True


def _progress(iterable, *, enabled: bool, **kwargs):
    """Return a tqdm-wrapped iterator when progress display is enabled."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _discover_band_paths(
    source_dir: Path,
    include_bands: Iterable[str] | None,
    exclude_bands: Iterable[str],
) -> list[Path]:
    """Enumerate VIIRS band Zarr directories selected for imputation."""
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    include = {x.strip() for x in include_bands} if include_bands else None
    exclude = {x.strip() for x in exclude_bands}

    out: list[Path] = []
    for p in sorted(source_dir.glob("*.zarr")):
        name = p.stem
        if include is not None and name not in include:
            continue
        if name in exclude:
            continue
        out.append(p)

    if not out:
        raise ValueError("No .zarr bands matched include/exclude")
    return out


def _kernel_fill_once_3x3(arr2d: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill missing values from valid entries in a 3x3 neighborhood."""
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


def _rasterio_fill_2d(arr2d: np.ndarray, *, max_search_distance: float, smoothing_iterations: int) -> tuple[np.ndarray, int]:
    """Fill missing values using `rasterio.fill.fillnodata` interpolation."""
    if not HAS_RASTERIO or _rio_fillnodata is None:
        return np.asarray(arr2d, dtype=np.float32), 0

    a = np.asarray(arr2d, dtype=np.float32)
    nan_before = int((~np.isfinite(a)).sum())
    if nan_before == 0:
        return a, 0

    img = np.where(np.isfinite(a), a, 0.0).astype(np.float32, copy=False)
    mask = np.isfinite(a).astype(np.uint8)
    out = _rio_fillnodata(
        image=img,
        mask=mask,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=int(smoothing_iterations),
    )
    out = np.asarray(out, dtype=np.float32)
    nan_after = int((~np.isfinite(out)).sum())
    return out, int(max(nan_before - nan_after, 0))


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
    fallback = 0.0 if finite.size == 0 else float(np.mean(finite, dtype=np.float64))
    out[nan_mask] = np.float32(fallback)
    return out, n_nan, float(fallback)


def _impute_grid_2d(arr2d: np.ndarray, cfg: ViirsImputeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the full imputation sequence and return the filled grid with diagnostics."""
    a = np.asarray(arr2d, dtype=np.float32)
    a = np.where(np.isfinite(a), a, np.nan).astype(np.float32, copy=False)
    nan_before = int((~np.isfinite(a)).sum())

    out = a.copy()
    out, filled_kernel = _kernel_fill_2d(out, max_passes=cfg.max_passes)

    filled_rasterio = 0
    if cfg.use_rasterio_fillnodata:
        out, filled_rasterio = _rasterio_fill_2d(
            out,
            max_search_distance=cfg.rasterio_max_search_distance,
            smoothing_iterations=cfg.rasterio_smoothing_iterations,
        )

    filled_nearest = 0
    nearest_iters = 0
    if cfg.nearest_fill_fallback and (~np.isfinite(out)).any():
        out, filled_nearest, nearest_iters = _nearest_propagation_fill(out)

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


def _copy_attrs(src_obj, dst_obj) -> None:
    """Copy Zarr attributes when they can be read safely."""
    try:
        dst_obj.attrs.update(dict(src_obj.attrs))
    except Exception:
        pass


def _copy_array(src_group: zarr.Group, dst_group: zarr.Group, name: str) -> None:
    """Copy one auxiliary array from the source band store to the destination."""
    arr = src_group[name]
    data = np.asarray(arr[:])
    kwargs: dict[str, Any] = {
        "shape": data.shape,
        "dtype": data.dtype,
        "overwrite": True,
    }
    if getattr(arr, "chunks", None) is not None:
        kwargs["chunks"] = arr.chunks
    dst_arr = dst_group.create_dataset(name, **kwargs)
    dst_arr[:] = data
    _copy_attrs(arr, dst_arr)


def _resolve_time_key(src_group: zarr.Group) -> str | None:
    """Return the time-coordinate key used by the band store, if present."""
    if "time" in src_group:
        return "time"
    if "timestamp" in src_group:
        return "timestamp"
    return None


def _validate_grid_and_time(src: zarr.Group, shape_3d: tuple[int, int, int], time_key: str | None) -> None:
    """Validate agreement between data-array, time, and coordinate dimensions."""
    t, h, w = shape_3d
    if time_key is not None:
        n_time = int(src[time_key].shape[0])
        if n_time != t:
            raise ValueError(f"time length mismatch: {time_key}={n_time} vs data time={t}")

    if "x" in src and int(src["x"].shape[0]) != w:
        raise ValueError(f"x length mismatch: x={src['x'].shape[0]} vs width={w}")
    if "y" in src and int(src["y"].shape[0]) != h:
        raise ValueError(f"y length mismatch: y={src['y'].shape[0]} vs height={h}")
    if "centroid_lat" in src:
        sh = tuple(int(v) for v in src["centroid_lat"].shape)
        if sh != (h, w):
            raise ValueError(f"centroid_lat shape mismatch: {sh} vs {(h,w)}")
    if "centroid_lon" in src:
        sh = tuple(int(v) for v in src["centroid_lon"].shape)
        if sh != (h, w):
            raise ValueError(f"centroid_lon shape mismatch: {sh} vs {(h,w)}")


def impute_one_band(band_path: Path, cfg: ViirsImputeConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    """Impute one VIIRS band store and return summary and per-time diagnostics."""
    src = zarr.open_group(str(band_path), mode="r")
    out_path = cfg.output_dir / band_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_group(str(out_path), mode="w")

    _copy_attrs(src, dst)

    target_arrays = [name for name in cfg.arrays_to_impute if name in src]
    if not target_arrays:
        raise ValueError(f"{band_path.name}: none of arrays_to_impute found: {cfg.arrays_to_impute}")

    ref = src[target_arrays[0]]
    if ref.ndim != 3:
        raise ValueError(f"{band_path.name}: {target_arrays[0]} ndim={ref.ndim} but expected 3")
    shape_3d = tuple(int(v) for v in ref.shape)

    for name in target_arrays[1:]:
        arr = src[name]
        if arr.ndim != 3:
            raise ValueError(f"{band_path.name}: {name} ndim={arr.ndim} but expected 3")
        if tuple(int(v) for v in arr.shape) != shape_3d:
            raise ValueError(f"{band_path.name}: {name} shape={arr.shape} differs from {target_arrays[0]} shape={shape_3d}")

    time_key = _resolve_time_key(src)
    _validate_grid_and_time(src, shape_3d, time_key)

    # Copy all auxiliary arrays before writing imputed target arrays.
    if cfg.preserve_aux_arrays:
        for key in src.array_keys():
            if key in target_arrays:
                continue
            _copy_array(src, dst, key)

    rows: list[dict[str, Any]] = []
    per_array_summary: list[dict[str, Any]] = []

    t, h, w = shape_3d
    for arr_name in target_arrays:
        src_arr = src[arr_name]
        kwargs: dict[str, Any] = {
            "shape": src_arr.shape,
            "dtype": np.float32,
            "overwrite": True,
        }
        if getattr(src_arr, "chunks", None) is not None:
            kwargs["chunks"] = src_arr.chunks
        dst_arr = dst.create_dataset(arr_name, **kwargs)
        _copy_attrs(src_arr, dst_arr)

        total_before = 0
        total_after = 0
        total_filled = 0
        total_kernel = 0
        total_rio = 0
        total_nearest = 0
        total_global = 0
        nearest_iters_total = 0
        days_changed = 0

        iterator = _progress(
            range(t),
            enabled=cfg.show_progress,
            desc=f"impute::{band_path.stem}.{arr_name}",
            unit="day",
        )
        for ti in iterator:
            grid = np.asarray(src_arr[ti, :, :], dtype=np.float32)
            out, det = _impute_grid_2d(grid, cfg)
            dst_arr[ti, :, :] = out

            nan_before = int(det["nan_before"])
            nan_after = int(det["nan_after"])
            filled = int(max(nan_before - nan_after, 0))

            total_before += nan_before
            total_after += nan_after
            total_filled += filled
            total_kernel += int(det["filled_kernel"])
            total_rio += int(det["filled_rasterio"])
            total_nearest += int(det["filled_nearest"])
            total_global += int(det["filled_global"])
            nearest_iters_total += int(det["nearest_iters"])
            days_changed += int(filled > 0)

            rows.append(
                {
                    "band": band_path.stem,
                    "array": arr_name,
                    "time_index": int(ti),
                    "nan_before": nan_before,
                    "nan_after": nan_after,
                    "filled": int(filled),
                    "filled_kernel": int(det["filled_kernel"]),
                    "filled_rasterio": int(det["filled_rasterio"]),
                    "filled_nearest": int(det["filled_nearest"]),
                    "filled_global": int(det["filled_global"]),
                    "nearest_iters": int(det["nearest_iters"]),
                }
            )

        per_array_summary.append(
            {
                "band": band_path.stem,
                "array": arr_name,
                "shape": [int(t), int(h), int(w)],
                "nan_before_total": int(total_before),
                "nan_after_total": int(total_after),
                "filled_total": int(total_filled),
                "filled_kernel_total": int(total_kernel),
                "filled_rasterio_total": int(total_rio),
                "filled_nearest_total": int(total_nearest),
                "filled_global_total": int(total_global),
                "nearest_iters_total": int(nearest_iters_total),
                "days_changed": int(days_changed),
                "fill_rate_pct_of_nan_before": float(100.0 * total_filled / max(total_before, 1)),
                "used_rasterio": bool(cfg.use_rasterio_fillnodata and HAS_RASTERIO),
            }
        )

    # Preserve root-level metadata and record the imputed arrays.
    dst.attrs["imputation_arrays"] = list(target_arrays)
    dst.attrs["imputation_note"] = "Only mean/min/max were imputed; time/grid arrays preserved."

    summary = {
        "band": band_path.stem,
        "source": str(band_path),
        "output": str(out_path),
        "schema": "viirs_band_aggregates",
        "time_key": time_key,
        "shape": [int(t), int(h), int(w)],
        "arrays_imputed": list(target_arrays),
        "per_array": per_array_summary,
    }
    day_df = pd.DataFrame(rows)
    return summary, day_df


def run_viirs_imputation(cfg: ViirsImputeConfig) -> dict[str, str]:
    """Run VIIRS band imputation for all selected inputs and write diagnostics."""
    if int(cfg.kernel_size) != 3:
        raise ValueError("Only kernel_size=3 is supported in this implementation")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.stats_dir.mkdir(parents=True, exist_ok=True)

    band_paths = _discover_band_paths(
        source_dir=cfg.source_dir,
        include_bands=cfg.include_bands,
        exclude_bands=cfg.exclude_bands,
    )

    summaries: list[dict[str, Any]] = []
    day_stats: list[pd.DataFrame] = []

    n_jobs = max(1, int(cfg.n_jobs))
    if n_jobs == 1 or len(band_paths) <= 1:
        iterator = _progress(band_paths, enabled=cfg.show_progress, desc="bands", unit="band")
        for bp in iterator:
            s, d = impute_one_band(bp, cfg)
            summaries.append(s)
            day_stats.append(d)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            fut_map = {ex.submit(impute_one_band, bp, cfg): bp.stem for bp in band_paths}
            iterator = as_completed(fut_map)
            iterator = _progress(iterator, enabled=cfg.show_progress, desc="bands(parallel)", unit="band", total=len(fut_map))
            for fut in iterator:
                s, d = fut.result()
                summaries.append(s)
                day_stats.append(d)

    # Flatten per-array summaries for CSV export.
    summary_rows: list[dict[str, Any]] = []
    for s in summaries:
        base = {
            "band": s["band"],
            "source": s["source"],
            "output": s["output"],
            "schema": s["schema"],
            "time_key": s["time_key"],
            "shape_t": s["shape"][0],
            "shape_h": s["shape"][1],
            "shape_w": s["shape"][2],
        }
        for r in s["per_array"]:
            summary_rows.append({**base, **r})

    summary_df = pd.DataFrame(summary_rows).sort_values(["band", "array"]).reset_index(drop=True)
    day_df = pd.concat(day_stats, ignore_index=True) if day_stats else pd.DataFrame()

    p_summary_csv = cfg.stats_dir / "viirs_impute_summary.csv"
    p_summary_json = cfg.stats_dir / "viirs_impute_summary.json"
    p_day_csv = cfg.stats_dir / "viirs_impute_day_stats.csv"

    summary_df.to_csv(p_summary_csv, index=False)
    p_summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    day_df.to_csv(p_day_csv, index=False)

    run_meta = {
        "source_dir": str(cfg.source_dir),
        "output_dir": str(cfg.output_dir),
        "stats_dir": str(cfg.stats_dir),
        "n_bands": int(len(band_paths)),
        "arrays_to_impute": list(cfg.arrays_to_impute),
        "preserve_aux_arrays": bool(cfg.preserve_aux_arrays),
        "kernel_size": int(cfg.kernel_size),
        "max_passes": int(cfg.max_passes),
        "use_rasterio_fillnodata": bool(cfg.use_rasterio_fillnodata),
        "rasterio_available": bool(HAS_RASTERIO),
        "rasterio_max_search_distance": float(cfg.rasterio_max_search_distance),
        "rasterio_smoothing_iterations": int(cfg.rasterio_smoothing_iterations),
        "nearest_fill_fallback": bool(cfg.nearest_fill_fallback),
        "ensure_no_nan": bool(cfg.ensure_no_nan),
        "n_jobs": int(cfg.n_jobs),
        "include_bands": list(cfg.include_bands) if cfg.include_bands else None,
        "exclude_bands": list(cfg.exclude_bands),
        "outputs": {
            "summary_csv": str(p_summary_csv),
            "summary_json": str(p_summary_json),
            "day_stats_csv": str(p_day_csv),
        },
    }
    p_run = cfg.stats_dir / "viirs_impute_run_metadata.json"
    p_run.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return {
        "summary_csv": str(p_summary_csv),
        "summary_json": str(p_summary_json),
        "day_stats_csv": str(p_day_csv),
        "run_metadata": str(p_run),
        "output_dir": str(cfg.output_dir),
    }
