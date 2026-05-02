"""IO and validation helpers for window-level feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import Iterable

import numpy as np
import pandas as pd

DAY_SEC = 86_400


@dataclass(frozen=True)
class LabelLoadConfig:
    """Configuration for loading labeled sample CSVs."""

    label_dir: Path
    label_files: tuple[str, ...]
    date_col: str = "target_date"
    row_col: str = "row"
    col_col: str = "col"


def get_logger(
    name: str = "window_pipeline", level: int = logging.INFO
) -> logging.Logger:
    """Create or reuse a console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _to_utc_day_seconds(date_series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(date_series, utc=True, errors="coerce").dt.floor("D")
    dt64 = dt.to_numpy(dtype="datetime64[ns]")
    ns = dt64.astype("int64")
    out = ns.astype(np.float64) / 1_000_000_000.0
    out[ns == np.iinfo(np.int64).min] = np.nan
    return out.astype(np.float64)


def load_label_samples(
    config: LabelLoadConfig, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Load all required label sample files and return unified sample table.

    Output columns:
    - target_date (YYYY-MM-DD string)
    - target_ts (epoch day seconds)
    - row (int)
    - col (int)
    - split (train/val/test inferred from file name)
    - source_file
    """
    lg = logger or get_logger()

    frames: list[pd.DataFrame] = []
    for file_name in config.label_files:
        fp = config.label_dir / file_name
        if not fp.exists():
            raise FileNotFoundError(f"Label file not found: {fp}")

        df = pd.read_csv(fp, usecols=[config.date_col, config.row_col, config.col_col])
        df = df.rename(
            columns={
                config.date_col: "target_date",
                config.row_col: "row",
                config.col_col: "col",
            }
        )

        split = "unknown"
        lower_name = file_name.lower()
        if "train" in lower_name:
            split = "train"
        elif "val" in lower_name:
            split = "val"
        elif "test" in lower_name:
            split = "test"

        df["split"] = split
        df["source_file"] = file_name
        frames.append(df)
        lg.info("Loaded %s rows from %s", len(df), fp.name)

    if not frames:
        raise ValueError("No label files loaded")

    all_df = pd.concat(frames, ignore_index=True)

    all_df["row"] = pd.to_numeric(all_df["row"], errors="coerce").astype("Int64")
    all_df["col"] = pd.to_numeric(all_df["col"], errors="coerce").astype("Int64")
    ts = _to_utc_day_seconds(all_df["target_date"])

    ok = (
        np.isfinite(ts)
        & all_df["row"].notna().to_numpy()
        & all_df["col"].notna().to_numpy()
    )
    dropped = int((~ok).sum())
    if dropped > 0:
        lg.warning("Dropping %d invalid label rows (bad date/row/col)", dropped)

    all_df = all_df.loc[ok].copy()
    all_df["target_ts"] = ts[ok].astype(np.int64)
    all_df["target_date"] = pd.to_datetime(
        all_df["target_ts"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")
    all_df["row"] = all_df["row"].astype(np.int32)
    all_df["col"] = all_df["col"].astype(np.int32)

    # Keep first source assignment for duplicated samples across files.
    all_df = all_df.drop_duplicates(
        subset=["target_date", "row", "col"], keep="first"
    ).reset_index(drop=True)
    lg.info("Unified unique samples: %d", len(all_df))

    return all_df[["target_date", "target_ts", "row", "col", "split", "source_file"]]


def collect_required_samples(
    samples_df: pd.DataFrame, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Return cleaned sample keys sorted for stable downstream processing."""
    lg = logger or get_logger()

    req = samples_df[
        ["target_date", "target_ts", "row", "col", "split", "source_file"]
    ].copy()
    req = req.sort_values(["target_ts", "row", "col"]).reset_index(drop=True)

    lg.info(
        "Required sample stats | rows=%d | dates=%d | rows_range=[%d,%d] | cols_range=[%d,%d]",
        len(req),
        req["target_date"].nunique(),
        int(req["row"].min()),
        int(req["row"].max()),
        int(req["col"].min()),
        int(req["col"].max()),
    )
    return req


def discover_feature_zarrs(
    feature_zarr_dir: Path,
    include_features: Iterable[str] | None = None,
    exclude_features: Iterable[str] | None = None,
) -> list[Path]:
    """Discover feature zarr stores to process."""
    if not feature_zarr_dir.exists():
        raise FileNotFoundError(f"Feature zarr directory not found: {feature_zarr_dir}")

    include = {x.strip() for x in include_features} if include_features else None
    exclude = {x.strip() for x in exclude_features} if exclude_features else set()

    zarr_paths: list[Path] = []
    for p in sorted(feature_zarr_dir.glob("*.zarr")):
        name = p.stem
        if name.startswith("_"):
            continue
        if include is not None and name not in include:
            continue
        if name in exclude:
            continue
        zarr_paths.append(p)

    if not zarr_paths:
        raise ValueError("No feature zarr files matched include/exclude rules")
    return zarr_paths


def save_output_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Save a DataFrame to CSV with parent creation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def validate_outputs(
    output_paths: list[Path],
    expected_samples: int,
    num_windows: int,
    logger: logging.Logger | None = None,
) -> dict:
    """Validate output CSVs and return summary stats."""
    lg = logger or get_logger()

    summary: dict[str, dict] = {}
    expected_rows = int(expected_samples * num_windows)

    for p in output_paths:
        df = pd.read_csv(p)
        key_cols = ["target_date", "row", "col", "window_id"]

        missing_cols = [c for c in key_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{p.name} missing required columns: {missing_cols}")

        n_rows = int(len(df))
        dup = int(df.duplicated(subset=key_cols).sum())
        wid = sorted(df["window_id"].dropna().unique().tolist())
        summary[p.name] = {
            "rows": n_rows,
            "expected_rows": expected_rows,
            "duplicate_key_rows": dup,
            "window_ids": wid,
            "ok_rows": (n_rows == expected_rows),
            "ok_no_duplicates": (dup == 0),
        }

        lg.info(
            "Validate %s | rows=%d expected=%d dup=%d windows=%s",
            p.name,
            n_rows,
            expected_rows,
            dup,
            wid,
        )

    return summary


def write_summary_json(summary: dict, out_path: Path) -> None:
    """Write JSON summary file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_feature_manifest(feature_zarr_dir: Path) -> dict | None:
    """Load optional feature manifest if available."""
    fp = feature_zarr_dir / "feature_manifest.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))
