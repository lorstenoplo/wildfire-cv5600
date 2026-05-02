"""Temporal fire-pattern search utilities for fixed-grid daily labels.

Window-based history logic (not exact lag-day lookup):
- window_1: [t-16, t)
- window_2: [t-32, t-16)
- window_3: [t-48, t-32)
- window_4: [t-64, t-48)

For each target day t and cell:
- fire_t
- fire_in_prev_0_16_days
- fire_in_prev_16_32_days
- fire_in_prev_32_48_days
- fire_in_prev_48_64_days
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import zarr

DAY_SEC = 86_400
WINDOW_DAYS_DEFAULT = 16
NUM_WINDOWS_DEFAULT = 4
WINDOW_FIRE_COUNT_THRESHOLD_DEFAULT = 3


@dataclass
class PatternSearchResult:
    fire_1: pd.DataFrame
    fire_5: pd.DataFrame
    fre_2: pd.DataFrame
    summary: dict


def _validate_arrays(timestamps: np.ndarray, labels: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError(f"timestamp array must be 1D, got shape={timestamps.shape}")
    if labels.ndim != 3:
        raise ValueError(f"label array must be 3D [N,H,W], got shape={labels.shape}")
    if labels.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"timestamp length ({timestamps.shape[0]}) and label length ({labels.shape[0]}) mismatch"
        )


def load_labels_from_zarr(
    labels_zarr_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamp and label arrays from labels.zarr."""
    z = zarr.open(str(labels_zarr_path), mode="r")

    if "timestamp" not in z or "label" not in z:
        raise KeyError("labels.zarr must contain arrays: 'timestamp' and 'label'")

    timestamps = np.asarray(z["timestamp"][:], dtype=np.int64)
    labels = np.asarray(z["label"][:], dtype=np.uint8)
    _validate_arrays(timestamps, labels)
    return timestamps, labels


def _window_indices_for_target(
    target_ts: int,
    ts_to_idx: dict[int, int],
    window_days: int,
    num_windows: int,
) -> list[np.ndarray] | None:
    """Return list of index arrays for history windows before target day.

    Returns None if any required history day is missing.
    """
    windows: list[np.ndarray] = []

    for w in range(num_windows):
        # w=0 => [t-16, t), w=1 => [t-32, t-16), ...
        end_offset = w * window_days
        start_offset = (w + 1) * window_days

        start_ts = target_ts - start_offset * DAY_SEC
        end_ts = target_ts - end_offset * DAY_SEC

        idxs = []
        cur = start_ts
        while cur < end_ts:
            i = ts_to_idx.get(cur)
            if i is None:
                return None
            idxs.append(i)
            cur += DAY_SEC

        windows.append(np.asarray(idxs, dtype=np.int64))

    return windows


def _build_valid_target_windows(
    timestamps: np.ndarray,
    window_days: int,
    num_windows: int,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build target-day entries with resolved day-index windows."""
    ts_to_idx = {int(ts): i for i, ts in enumerate(timestamps.tolist())}
    targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for i_t, ts in enumerate(timestamps.tolist()):
        target_ts = int(ts)
        win_idx = _window_indices_for_target(
            target_ts=target_ts,
            ts_to_idx=ts_to_idx,
            window_days=window_days,
            num_windows=num_windows,
        )
        if win_idx is None:
            continue
        idx_0_16, idx_16_32, idx_32_48, idx_48_64 = win_idx
        targets.append((i_t, idx_0_16, idx_16_32, idx_32_48, idx_48_64))

    return targets


def _chunk_targets(
    targets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_chunks: int,
) -> list[list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Split targets into balanced contiguous chunks."""
    if n_chunks <= 1 or len(targets) == 0:
        return [targets]

    n_chunks = min(n_chunks, len(targets))
    chunk_size = (len(targets) + n_chunks - 1) // n_chunks
    return [targets[i : i + chunk_size] for i in range(0, len(targets), chunk_size)]


def _records_for_mask(
    mask: np.ndarray,
    target_ts: int,
    width: int,
    pattern_type: str,
    fire_t: np.ndarray,
    prev_0_16: np.ndarray,
    prev_16_32: np.ndarray,
    prev_32_48: np.ndarray,
    prev_48_64: np.ndarray,
) -> pd.DataFrame:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return pd.DataFrame()

    rows = (idx // width).astype(np.int32)
    cols = (idx % width).astype(np.int32)
    target_date = pd.to_datetime(target_ts, unit="s", utc=True).strftime("%Y-%m-%d")

    return pd.DataFrame(
        {
            "target_date": target_date,
            "row": rows,
            "col": cols,
            "fire_t": fire_t[idx].astype(np.uint8),
            "fire_in_prev_0_16_days": prev_0_16[idx].astype(np.uint8),
            "fire_in_prev_16_32_days": prev_16_32[idx].astype(np.uint8),
            "fire_in_prev_32_48_days": prev_32_48[idx].astype(np.uint8),
            "fire_in_prev_48_64_days": prev_48_64[idx].astype(np.uint8),
            "pattern_type": pattern_type,
        }
    )


def _full_day_table(
    target_ts: int,
    height: int,
    width: int,
    fire_t: np.ndarray,
    prev_0_16: np.ndarray,
    prev_16_32: np.ndarray,
    prev_32_48: np.ndarray,
    prev_48_64: np.ndarray,
) -> pd.DataFrame:
    n_cells = height * width
    idx = np.arange(n_cells, dtype=np.int32)
    rows = (idx // width).astype(np.int32)
    cols = (idx % width).astype(np.int32)
    target_date = pd.to_datetime(target_ts, unit="s", utc=True).strftime("%Y-%m-%d")

    return pd.DataFrame(
        {
            "target_date": target_date,
            "row": rows,
            "col": cols,
            "fire_t": fire_t.astype(np.uint8),
            "fire_in_prev_0_16_days": prev_0_16.astype(np.uint8),
            "fire_in_prev_16_32_days": prev_16_32.astype(np.uint8),
            "fire_in_prev_32_48_days": prev_32_48.astype(np.uint8),
            "fire_in_prev_48_64_days": prev_48_64.astype(np.uint8),
        }
    )


def _process_target_batch(
    batch: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    timestamps: np.ndarray,
    labels_flat: np.ndarray,
    height: int,
    width: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Process one batch of target days and return class-matched DataFrame parts."""
    fire_1_parts: list[pd.DataFrame] = []
    fire_5_parts: list[pd.DataFrame] = []
    fre_2_parts: list[pd.DataFrame] = []

    for i_t, idx_0_16, idx_16_32, idx_32_48, idx_48_64 in batch:
        target_ts = int(timestamps[i_t])

        fire_t = labels_flat[i_t]
        count_0_16 = labels_flat[idx_0_16].sum(axis=0).astype(np.uint16)
        count_16_32 = labels_flat[idx_16_32].sum(axis=0).astype(np.uint16)
        count_32_48 = labels_flat[idx_32_48].sum(axis=0).astype(np.uint16)
        count_48_64 = labels_flat[idx_48_64].sum(axis=0).astype(np.uint16)

        prev_0_16 = (count_0_16 >= _WINDOW_FIRE_COUNT_THRESHOLD).astype(np.uint8)
        prev_16_32 = (count_16_32 >= _WINDOW_FIRE_COUNT_THRESHOLD).astype(np.uint8)
        prev_32_48 = (count_32_48 >= _WINDOW_FIRE_COUNT_THRESHOLD).astype(np.uint8)
        prev_48_64 = (count_48_64 >= _WINDOW_FIRE_COUNT_THRESHOLD).astype(np.uint8)

        m1 = (
            (fire_t == 1)
            & (prev_0_16 == 0)
            & (prev_16_32 == 0)
            & (prev_32_48 == 0)
            & (prev_48_64 == 0)
        )
        m2 = (
            (fire_t == 1)
            & (prev_0_16 == 1)
            & (prev_16_32 == 1)
            & (prev_32_48 == 1)
            & (prev_48_64 == 1)
        )
        m3 = (
            (fire_t == 1)
            & (prev_0_16 == 1)
            & (prev_16_32 == 0)
            & (prev_32_48 == 0)
            & (prev_48_64 == 0)
        )

        # Strict exclusivity across classes for same (target_date, row, col).
        overlap = (m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)) > 1
        if bool(np.any(overlap)):
            overlap_n = int(np.sum(overlap))
            raise RuntimeError(
                f"Class overlap detected for timestamp={target_ts}: {overlap_n} unit(s) match multiple classes."
            )

        df1 = _records_for_mask(
            m1,
            target_ts,
            width,
            "Fire-1",
            fire_t,
            prev_0_16,
            prev_16_32,
            prev_32_48,
            prev_48_64,
        )
        df2 = _records_for_mask(
            m2,
            target_ts,
            width,
            "Fire-5",
            fire_t,
            prev_0_16,
            prev_16_32,
            prev_32_48,
            prev_48_64,
        )
        df3 = _records_for_mask(
            m3,
            target_ts,
            width,
            "Fre-2",
            fire_t,
            prev_0_16,
            prev_16_32,
            prev_32_48,
            prev_48_64,
        )

        if not df1.empty:
            fire_1_parts.append(df1)
        if not df2.empty:
            fire_5_parts.append(df2)
        if not df3.empty:
            fre_2_parts.append(df3)

    return fire_1_parts, fire_5_parts, fre_2_parts


def search_fire_history_patterns(
    labels_zarr_path: str | Path,
    window_days: int = WINDOW_DAYS_DEFAULT,
    num_windows: int = NUM_WINDOWS_DEFAULT,
    window_fire_count_threshold: int = WINDOW_FIRE_COUNT_THRESHOLD_DEFAULT,
    export_full_table_csv: str | Path | None = None,
    n_jobs: int = 1,
) -> PatternSearchResult:
    """Greedily scan full dataset with 16-day history-window aggregation.

    Class definitions (mutually exclusive by construction):
    - Fire-1: fire_t=1 and all four windows are 0
    - Fire-5: fire_t=1 and all four windows are 1
    - Fre-2 : fire_t=1, prev_0_16=1, and other three windows are 0

    If `export_full_table_csv` is provided, writes all valid (target_date,row,col)
    rows (full 64-day-history-available table) incrementally to CSV.
    """
    if window_days <= 0:
        raise ValueError("window_days must be > 0")
    if num_windows != 4:
        raise ValueError(
            "This implementation expects exactly 4 windows for 64-day history"
        )
    if n_jobs <= 0:
        raise ValueError("n_jobs must be >= 1")
    if window_fire_count_threshold <= 0:
        raise ValueError("window_fire_count_threshold must be >= 1")

    global _WINDOW_FIRE_COUNT_THRESHOLD
    _WINDOW_FIRE_COUNT_THRESHOLD = int(window_fire_count_threshold)

    timestamps, labels = load_labels_from_zarr(labels_zarr_path)
    n_days, h, w = labels.shape
    n_cells = h * w

    labels_flat = labels.reshape(n_days, n_cells).astype(np.uint8)

    targets = _build_valid_target_windows(
        timestamps=timestamps,
        window_days=window_days,
        num_windows=num_windows,
    )
    valid_target_days = int(len(targets))
    full_rows_written = 0

    csv_path: Path | None = None
    wrote_header = False
    if export_full_table_csv is not None:
        csv_path = Path(export_full_table_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            csv_path.unlink()

    # Optional full table export (all valid target_date x all cells)
    if csv_path is not None:
        for i_t, idx_0_16, idx_16_32, idx_32_48, idx_48_64 in targets:
            target_ts = int(timestamps[i_t])
            fire_t = labels_flat[i_t]
            prev_0_16 = labels_flat[idx_0_16].max(axis=0).astype(np.uint8)
            prev_16_32 = labels_flat[idx_16_32].max(axis=0).astype(np.uint8)
            prev_32_48 = labels_flat[idx_32_48].max(axis=0).astype(np.uint8)
            prev_48_64 = labels_flat[idx_48_64].max(axis=0).astype(np.uint8)

            day_df = _full_day_table(
                target_ts=target_ts,
                height=h,
                width=w,
                fire_t=fire_t,
                prev_0_16=prev_0_16,
                prev_16_32=prev_16_32,
                prev_32_48=prev_32_48,
                prev_48_64=prev_48_64,
            )
            day_df.to_csv(csv_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            full_rows_written += len(day_df)

    # Parallelizable class matching across target days
    if n_jobs == 1:
        fire_1_parts, fire_5_parts, fre_2_parts = _process_target_batch(
            batch=targets,
            timestamps=timestamps,
            labels_flat=labels_flat,
            height=h,
            width=w,
        )
    else:
        fire_1_parts = []
        fire_5_parts = []
        fre_2_parts = []
        chunks = _chunk_targets(targets, n_jobs)
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _process_target_batch,
                    chunk,
                    timestamps,
                    labels_flat,
                    h,
                    w,
                )
                for chunk in chunks
            ]
            for fut in futures:
                p1, p2, p3 = fut.result()
                fire_1_parts.extend(p1)
                fire_5_parts.extend(p2)
                fre_2_parts.extend(p3)

    cols = [
        "target_date",
        "row",
        "col",
        "fire_t",
        "fire_in_prev_0_16_days",
        "fire_in_prev_16_32_days",
        "fire_in_prev_32_48_days",
        "fire_in_prev_48_64_days",
        "pattern_type",
    ]

    fire_1 = (
        pd.concat(fire_1_parts, ignore_index=True)
        if fire_1_parts
        else pd.DataFrame(columns=cols)
    )
    fire_5 = (
        pd.concat(fire_5_parts, ignore_index=True)
        if fire_5_parts
        else pd.DataFrame(columns=cols)
    )
    fre_2 = (
        pd.concat(fre_2_parts, ignore_index=True)
        if fre_2_parts
        else pd.DataFrame(columns=cols)
    )

    summary = {
        "window_days": int(window_days),
        "num_windows": int(num_windows),
        "window_fire_count_threshold": int(window_fire_count_threshold),
        "num_days": int(n_days),
        "grid_shape_rows_cols": [int(h), int(w)],
        "num_cells": int(n_cells),
        "num_target_days_with_full_history": int(valid_target_days),
        "full_table_rows_written": int(full_rows_written),
        "full_table_csv": str(csv_path) if csv_path is not None else None,
        "n_jobs_used": int(n_jobs),
        "counts": {
            "Fire-1": int(len(fire_1)),
            "Fire-5": int(len(fire_5)),
            "Fre-2": int(len(fre_2)),
        },
    }

    return PatternSearchResult(
        fire_1=fire_1,
        fire_5=fire_5,
        fre_2=fre_2,
        summary=summary,
    )


def save_pattern_outputs(
    result: PatternSearchResult, out_dir: str | Path
) -> dict[str, str]:
    """Save class-specific outputs to CSV + summary JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p1 = out / "pattern_fire_1.csv"
    p2 = out / "pattern_fire_5.csv"
    p3 = out / "pattern_fre_2.csv"
    ps = out / "pattern_summary.json"

    result.fire_1.to_csv(p1, index=False)
    result.fire_5.to_csv(p2, index=False)
    result.fre_2.to_csv(p3, index=False)
    ps.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")

    return {
        "fire_1_csv": str(p1),
        "fire_5_csv": str(p2),
        "fre_2_csv": str(p3),
        "summary_json": str(ps),
    }
