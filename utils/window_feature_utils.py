"""Utility helpers for label-driven window feature dataset creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Iterable

import numpy as np
import pandas as pd

DAY_SEC = 86_400


@dataclass(frozen=True)
class LabelSampleConfig:
    """Configuration for loading label sample keys."""

    label_csv: Path
    date_col: str = "target_date"
    row_col: str = "row"
    col_col: str = "col"
    split_col: str = "split"
    label_col: str = "label"
    class_col: str = "type_class"


def get_logger(
    name: str = "window_feature_pipeline", level: int = logging.INFO
) -> logging.Logger:
    """Return a configured console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _to_epoch_day_seconds(date_series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(date_series, utc=True, errors="coerce").dt.floor("D")
    ns = dt.to_numpy(dtype="datetime64[ns]").astype("int64")
    out = ns.astype(np.float64) / 1_000_000_000.0
    out[ns == np.iinfo(np.int64).min] = np.nan
    return out


def load_label_samples(
    config: LabelSampleConfig, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Load labels and return normalized sample table.

    Output columns:
    - target_date (YYYY-MM-DD)
    - target_ts (UTC epoch day seconds)
    - row, col (int32)
    - split (string; defaults to 'all' if missing in file)
    - label (float32; NaN if missing in file)
    - type_class (string; '' if missing in file)
    """
    lg = logger or get_logger()
    if not config.label_csv.exists():
        raise FileNotFoundError(config.label_csv)

    header_cols = pd.read_csv(config.label_csv, nrows=0).columns.tolist()
    required = [config.date_col, config.row_col, config.col_col]
    missing_required = sorted(set(required) - set(header_cols))
    if missing_required:
        raise ValueError(
            f"{config.label_csv} missing required columns: {missing_required}"
        )

    usecols = [config.date_col, config.row_col, config.col_col]
    for c in [config.split_col, config.label_col, config.class_col]:
        if c in header_cols:
            usecols.append(c)

    df = pd.read_csv(config.label_csv, usecols=usecols)
    df = df.rename(
        columns={
            config.date_col: "target_date",
            config.row_col: "row",
            config.col_col: "col",
        }
    )

    if config.split_col in df.columns:
        df["split"] = df[config.split_col].astype(str).str.strip().str.lower()
    else:
        df["split"] = "all"
    if config.label_col in df.columns:
        df["label"] = pd.to_numeric(df[config.label_col], errors="coerce")
    else:
        df["label"] = np.nan
    if config.class_col in df.columns:
        df["type_class"] = df[config.class_col].astype(str)
    else:
        df["type_class"] = ""

    ts = _to_epoch_day_seconds(df["target_date"])
    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["col"] = pd.to_numeric(df["col"], errors="coerce")

    ok = np.isfinite(ts) & df["row"].notna().to_numpy() & df["col"].notna().to_numpy()
    dropped = int((~ok).sum())
    if dropped > 0:
        lg.warning("Dropping %d invalid label rows (bad date/row/col)", dropped)

    out = df.loc[
        ok, ["target_date", "row", "col", "split", "label", "type_class"]
    ].copy()
    out["target_ts"] = ts[ok].astype(np.int64)
    out["target_date"] = pd.to_datetime(
        out["target_ts"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")
    out["row"] = out["row"].astype(np.int32)
    out["col"] = out["col"].astype(np.int32)
    out = out.drop_duplicates(
        subset=["target_date", "row", "col"], keep="first"
    ).reset_index(drop=True)

    lg.info(
        "Loaded labels | rows=%d | dates=%d | splits=%s",
        len(out),
        out["target_date"].nunique(),
        sorted(out["split"].dropna().unique().tolist()),
    )
    return out[
        ["target_date", "target_ts", "row", "col", "split", "label", "type_class"]
    ]


def collect_required_samples(
    samples_df: pd.DataFrame, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Return cleaned, deterministic sample key table."""
    lg = logger or get_logger()
    req_cols = [
        "target_date",
        "target_ts",
        "row",
        "col",
        "split",
        "label",
        "type_class",
    ]
    out = samples_df[req_cols].copy()
    out = out.sort_values(["target_ts", "row", "col"]).reset_index(drop=True)
    lg.info(
        "Required samples | rows=%d | date_min=%s | date_max=%s",
        len(out),
        out["target_date"].min() if len(out) else None,
        out["target_date"].max() if len(out) else None,
    )
    return out


def get_window_date_ranges(
    target_ts: int, window_size: int = 16, num_windows: int = 4
) -> list[tuple[int, int, int]]:
    """Return list of (window_id, start_ts, end_ts_exclusive) for a target day.

    Window ids are recent-first:
    - 1 => [t-16, t)
    - 2 => [t-32, t-16)
    - ...
    """
    target_ts = int(target_ts)
    ranges: list[tuple[int, int, int]] = []
    for window_id in range(1, int(num_windows) + 1):
        end_ts = target_ts - (window_id - 1) * int(window_size) * DAY_SEC
        start_ts = end_ts - int(window_size) * DAY_SEC
        ranges.append((window_id, start_ts, end_ts))
    return ranges


def save_output_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write CSV output, creating parent directories if required."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_summary_json(summary: dict, out_path: Path) -> None:
    """Persist JSON summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def validate_outputs(
    df_long: pd.DataFrame,
    num_windows: int,
    feature_cols: Iterable[str],
    logger: logging.Logger | None = None,
) -> dict:
    """Validate key constraints for long window-format output."""
    lg = logger or get_logger()

    key_cols = ["target_date", "row", "col", "window_id"]
    missing_required = [c for c in key_cols if c not in df_long.columns]
    if missing_required:
        raise ValueError(f"Output missing required columns: {missing_required}")

    feature_cols = list(feature_cols)
    missing_feature_cols = [c for c in feature_cols if c not in df_long.columns]
    if missing_feature_cols:
        raise ValueError(f"Output missing feature columns: {missing_feature_cols}")

    dup = int(df_long.duplicated(subset=key_cols).sum())
    if dup > 0:
        raise ValueError(f"Output has duplicate sample-window keys: {dup}")

    windows = sorted(
        pd.to_numeric(df_long["window_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    expected_windows = list(range(1, int(num_windows) + 1))
    if windows != expected_windows:
        raise ValueError(
            f"Unexpected window ids: {windows}; expected {expected_windows}"
        )

    per_key = df_long.groupby(["target_date", "row", "col"], sort=False)[
        "window_id"
    ].nunique()
    bad_keys = int((per_key != int(num_windows)).sum())
    if bad_keys > 0:
        raise ValueError(f"{bad_keys} keys do not have exactly {num_windows} windows")

    out = {
        "rows": int(len(df_long)),
        "unique_sample_keys": int(
            df_long[["target_date", "row", "col"]].drop_duplicates().shape[0]
        ),
        "num_windows": int(num_windows),
        "num_features": int(len(feature_cols)),
        "duplicate_key_rows": int(dup),
        "bad_window_count_keys": int(bad_keys),
    }
    lg.info(
        "Validation ok | rows=%d | keys=%d | features=%d",
        out["rows"],
        out["unique_sample_keys"],
        out["num_features"],
    )
    return out
