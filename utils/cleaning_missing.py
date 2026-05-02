"""Utilities for split-wise missing-value analysis and key-based cleaning.

Cleaning policy:
- Never drop full feature columns.
- Drop sample keys (target_date,row,col) when any feature is missing.
- Keep original input files untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd

KEY_COLS = ["target_date", "row", "col"]
META_COLS = ["window_id", "split", "source_file"]


@dataclass(frozen=True)
class CleaningConfig:
    features_train_path: Path
    features_val_path: Path
    features_test_path: Path
    fire_train_path: Path
    fire_val_path: Path
    fire_test_path: Path
    nofire_train_path: Path
    nofire_val_path: Path
    nofire_test_path: Path
    output_dir: Path
    clean_train: bool = False
    treat_inf_as_missing: bool = True
    impute_train_missing: bool = True
    train_impute_strategy: str = "mean"


def identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns from a FEATURES_* dataframe."""
    excluded = set(KEY_COLS + META_COLS)
    return [c for c in df.columns if c not in excluded]


def _missing_mask(
    df: pd.DataFrame, feature_cols: list[str], treat_inf_as_missing: bool
) -> pd.DataFrame:
    vals = df[feature_cols]
    if treat_inf_as_missing:
        # numeric inf/-inf treated as missing
        m = vals.isna() | vals.apply(
            lambda s: np.isinf(pd.to_numeric(s, errors="coerce")), axis=0
        )
    else:
        m = vals.isna()
    return m


def feature_missing_stats(
    df: pd.DataFrame,
    split_name: str,
    feature_cols: list[str],
    treat_inf_as_missing: bool = True,
) -> dict:
    """Compute missing stats for one feature split dataframe."""
    m = _missing_mask(df, feature_cols, treat_inf_as_missing=treat_inf_as_missing)
    n_rows = int(len(df))

    per_feature = []
    for c in feature_cols:
        miss = int(m[c].sum())
        per_feature.append(
            {
                "feature": c,
                "missing_rows": miss,
                "missing_ratio": float(miss / n_rows) if n_rows else 0.0,
            }
        )

    per_window = []
    if "window_id" in df.columns:
        for wid, gidx in df.groupby("window_id").groups.items():
            gmask = m.loc[gidx]
            any_miss = gmask.any(axis=1)
            n = int(len(gmask))
            miss = int(any_miss.sum())
            per_window.append(
                {
                    "window_id": int(wid),
                    "rows": n,
                    "rows_with_any_missing": miss,
                    "rows_with_any_missing_ratio": float(miss / n) if n else 0.0,
                }
            )

    any_missing_row = m.any(axis=1)
    bad_keys = (
        df.loc[any_missing_row, KEY_COLS]
        .drop_duplicates(subset=KEY_COLS)
        .reset_index(drop=True)
    )

    return {
        "split": split_name,
        "n_rows": n_rows,
        "n_unique_keys": int(df[KEY_COLS].drop_duplicates().shape[0]),
        "rows_with_any_missing": int(any_missing_row.sum()),
        "rows_with_any_missing_ratio": float(any_missing_row.mean()) if n_rows else 0.0,
        "n_bad_keys": int(len(bad_keys)),
        "per_feature": per_feature,
        "per_window": per_window,
        "bad_keys_df": bad_keys,
    }


def drop_bad_keys(df: pd.DataFrame, bad_keys_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where (target_date,row,col) is in bad_keys_df."""
    if bad_keys_df.empty:
        return df.copy()

    bad = bad_keys_df.copy()
    bad["__drop__"] = 1
    out = df.merge(bad, on=KEY_COLS, how="left")
    cleaned = (
        out.loc[out["__drop__"].isna()]
        .drop(columns=["__drop__"])
        .reset_index(drop=True)
    )
    return cleaned


def impute_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    strategy: str = "mean",
    treat_inf_as_missing: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Impute missing values in feature columns only.

    Returns
    -------
    (imputed_df, report_dict)
    """
    out = df.copy()
    report: dict[str, dict] = {}

    strategy = strategy.lower().strip()
    if strategy not in {"mean"}:
        raise ValueError(f"Unsupported train impute strategy: {strategy}")

    for c in feature_cols:
        s = out[c]
        s_num = pd.to_numeric(s, errors="coerce")
        if treat_inf_as_missing:
            s_num = s_num.replace([np.inf, -np.inf], np.nan)

        missing_before = int(s_num.isna().sum())
        if missing_before == 0:
            report[c] = {"missing_before": 0, "filled": 0, "fill_value": None}
            out[c] = s_num
            continue

        fill_value = float(s_num.mean(skipna=True))
        # if entire column is NaN, keep NaN (no fake fill)
        if np.isnan(fill_value):
            filled = 0
            out[c] = s_num
        else:
            out[c] = s_num.fillna(fill_value)
            filled = missing_before

        report[c] = {
            "missing_before": missing_before,
            "filled": int(filled),
            "fill_value": None if np.isnan(fill_value) else float(fill_value),
        }

    return out, report


def split_key_intersection_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Count unique key overlap between two dataframes."""
    ka = a[KEY_COLS].drop_duplicates()
    kb = b[KEY_COLS].drop_duplicates()
    return int(ka.merge(kb, on=KEY_COLS, how="inner").shape[0])


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_cleaning(config: CleaningConfig) -> dict:
    """Run analysis + cleaning for train/val/test based on missing features.

    Rules:
    - For each split, find keys with any missing in FEATURES split.
    - Drop those keys from FEATURES split and matching FIRE/NOFIRE split.
    - Never drop columns, only rows identified by KEY_COLS.
    - Originals untouched; write *_clean.csv outputs.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all files
    f_train = pd.read_csv(config.features_train_path)
    f_val = pd.read_csv(config.features_val_path)
    f_test = pd.read_csv(config.features_test_path)

    fire_train = pd.read_csv(config.fire_train_path)
    fire_val = pd.read_csv(config.fire_val_path)
    fire_test = pd.read_csv(config.fire_test_path)

    nofire_train = pd.read_csv(config.nofire_train_path)
    nofire_val = pd.read_csv(config.nofire_val_path)
    nofire_test = pd.read_csv(config.nofire_test_path)

    feature_cols = identify_feature_columns(f_train)

    # Stats + bad keys per split
    st_train = feature_missing_stats(
        f_train, "train", feature_cols, treat_inf_as_missing=config.treat_inf_as_missing
    )
    st_val = feature_missing_stats(
        f_val, "val", feature_cols, treat_inf_as_missing=config.treat_inf_as_missing
    )
    st_test = feature_missing_stats(
        f_test, "test", feature_cols, treat_inf_as_missing=config.treat_inf_as_missing
    )

    bad_train = st_train.pop("bad_keys_df")
    bad_val = st_val.pop("bad_keys_df")
    bad_test = st_test.pop("bad_keys_df")

    # Apply cleaning policy
    train_impute_report: dict[str, dict] = {}
    if config.clean_train:
        f_train_clean = drop_bad_keys(f_train, bad_train)
        fire_train_clean = drop_bad_keys(fire_train, bad_train)
        nofire_train_clean = drop_bad_keys(nofire_train, bad_train)
    else:
        f_train_clean = f_train.copy()
        if config.impute_train_missing:
            f_train_clean, train_impute_report = impute_feature_columns(
                f_train_clean,
                feature_cols=feature_cols,
                strategy=config.train_impute_strategy,
                treat_inf_as_missing=config.treat_inf_as_missing,
            )
        fire_train_clean = fire_train.copy()
        nofire_train_clean = nofire_train.copy()

    f_val_clean = drop_bad_keys(f_val, bad_val)
    fire_val_clean = drop_bad_keys(fire_val, bad_val)
    nofire_val_clean = drop_bad_keys(nofire_val, bad_val)

    f_test_clean = drop_bad_keys(f_test, bad_test)
    fire_test_clean = drop_bad_keys(fire_test, bad_test)
    nofire_test_clean = drop_bad_keys(nofire_test, bad_test)

    # Save cleaned outputs
    save_df(f_train_clean, config.output_dir / "FEATURES_train_clean.csv")
    save_df(f_val_clean, config.output_dir / "FEATURES_val_clean.csv")
    save_df(f_test_clean, config.output_dir / "FEATURES_test_clean.csv")

    save_df(fire_train_clean, config.output_dir / "FIRE_train_clean.csv")
    save_df(fire_val_clean, config.output_dir / "FIRE_val_clean.csv")
    save_df(fire_test_clean, config.output_dir / "FIRE_test_clean.csv")

    save_df(nofire_train_clean, config.output_dir / "NOFIRE_train_clean.csv")
    save_df(nofire_val_clean, config.output_dir / "NOFIRE_val_clean.csv")
    save_df(nofire_test_clean, config.output_dir / "NOFIRE_test_clean.csv")

    # Save dropped key lists for traceability
    save_df(bad_train, config.output_dir / "dropped_keys_train.csv")
    save_df(bad_val, config.output_dir / "dropped_keys_val.csv")
    save_df(bad_test, config.output_dir / "dropped_keys_test.csv")

    summary = {
        "policy": {
            "drop_level": "sample_key_only",
            "key_columns": KEY_COLS,
            "drop_columns": False,
            "clean_train": bool(config.clean_train),
            "impute_train_missing": bool(config.impute_train_missing),
            "train_impute_strategy": str(config.train_impute_strategy),
            "clean_val": True,
            "clean_test": True,
            "treat_inf_as_missing": bool(config.treat_inf_as_missing),
        },
        "feature_columns_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "missing_stats": {
            "train": st_train,
            "val": st_val,
            "test": st_test,
        },
        "dropped_keys": {
            "train": int(len(bad_train)),
            "val": int(len(bad_val)),
            "test": int(len(bad_test)),
        },
        "train_imputation": {
            "features_touched": int(
                sum(1 for v in train_impute_report.values() if v.get("filled", 0) > 0)
            ),
            "total_filled_values": int(
                sum(v.get("filled", 0) for v in train_impute_report.values())
            ),
            "per_feature": train_impute_report,
        },
        "shape_before_after": {
            "FEATURES_train": {
                "before": int(len(f_train)),
                "after": int(len(f_train_clean)),
            },
            "FEATURES_val": {"before": int(len(f_val)), "after": int(len(f_val_clean))},
            "FEATURES_test": {
                "before": int(len(f_test)),
                "after": int(len(f_test_clean)),
            },
            "FIRE_train": {
                "before": int(len(fire_train)),
                "after": int(len(fire_train_clean)),
            },
            "FIRE_val": {
                "before": int(len(fire_val)),
                "after": int(len(fire_val_clean)),
            },
            "FIRE_test": {
                "before": int(len(fire_test)),
                "after": int(len(fire_test_clean)),
            },
            "NOFIRE_train": {
                "before": int(len(nofire_train)),
                "after": int(len(nofire_train_clean)),
            },
            "NOFIRE_val": {
                "before": int(len(nofire_val)),
                "after": int(len(nofire_val_clean)),
            },
            "NOFIRE_test": {
                "before": int(len(nofire_test)),
                "after": int(len(nofire_test_clean)),
            },
        },
        "key_overlap_after_cleaning": {
            "val_fire_vs_features": split_key_intersection_count(
                fire_val_clean, f_val_clean
            ),
            "val_nofire_vs_features": split_key_intersection_count(
                nofire_val_clean, f_val_clean
            ),
            "test_fire_vs_features": split_key_intersection_count(
                fire_test_clean, f_test_clean
            ),
            "test_nofire_vs_features": split_key_intersection_count(
                nofire_test_clean, f_test_clean
            ),
        },
        "outputs": {
            "dir": str(config.output_dir),
            "files": [
                "FEATURES_train_clean.csv",
                "FEATURES_val_clean.csv",
                "FEATURES_test_clean.csv",
                "FIRE_train_clean.csv",
                "FIRE_val_clean.csv",
                "FIRE_test_clean.csv",
                "NOFIRE_train_clean.csv",
                "NOFIRE_val_clean.csv",
                "NOFIRE_test_clean.csv",
                "dropped_keys_train.csv",
                "dropped_keys_val.csv",
                "dropped_keys_test.csv",
                "cleaning_summary.json",
            ],
        },
    }

    save_json(summary, config.output_dir / "cleaning_summary.json")
    return summary
