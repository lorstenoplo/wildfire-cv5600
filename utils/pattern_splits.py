"""Utilities to build strict date-disjoint 1:3:2 splits from pattern CSVs.

Input class files expected from fire-pattern mining:
- pattern_fire_1.csv
- pattern_fire_5.csv
- pattern_fre_2.csv

Each output split keeps exact class ratio (Fire-1:Fire-5:Fre-2) using
no-replacement sampling by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

DEFAULT_SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2012-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2023-12-31"),
}


@dataclass
class RatioSplitConfig:
    fire_1_csv: str | Path
    fire_5_csv: str | Path
    fre_2_csv: str | Path
    out_dir: str | Path
    splits: dict[str, tuple[str, str]] | None = None
    ratio: tuple[int, int, int] = (1, 3, 2)
    seed: int = 42
    allow_replacement_if_needed: bool = False
    key_cols: tuple[str, str, str] = ("target_date", "row", "col")


def _read_class_csv(path: str | Path, key_cols: tuple[str, str, str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing class CSV: {p}")

    df = pd.read_csv(p)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} missing required columns: {missing}")

    # Normalize date for split filtering and uniqueness checks.
    df = df.copy()
    df["target_date"] = pd.to_datetime(df["target_date"], errors="raise").dt.date
    df = df.drop_duplicates(list(key_cols)).reset_index(drop=True)
    return df


def _slice_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    return df[(df["target_date"] >= s) & (df["target_date"] <= e)].copy()


def _sample_exact(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    replace: bool,
) -> pd.DataFrame:
    if n < 0:
        raise ValueError("Sample size cannot be negative")
    if n == 0:
        return df.iloc[0:0].copy()
    if not replace and len(df) < n:
        raise ValueError(f"Need {n} rows, found {len(df)} with replace=False")

    idx = rng.choice(len(df), size=n, replace=replace)
    return df.iloc[idx].copy()


def _assert_disjoint_dates(split_outputs: dict[str, pd.DataFrame]) -> None:
    split_names = list(split_outputs.keys())
    date_sets = {k: set(v["target_date"].tolist()) for k, v in split_outputs.items()}

    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            if not date_sets[a].isdisjoint(date_sets[b]):
                overlap = sorted(date_sets[a].intersection(date_sets[b]))
                preview = overlap[:5]
                raise ValueError(
                    f"Date overlap detected between '{a}' and '{b}'. "
                    f"Examples: {preview}"
                )


def _assert_unique_units(
    df: pd.DataFrame, key_cols: tuple[str, str, str], split_name: str
) -> None:
    dup_n = int(df.duplicated(list(key_cols)).sum())
    if dup_n != 0:
        raise ValueError(
            f"{split_name}: found {dup_n} duplicate units by keys={key_cols}"
        )


def build_ratio_splits(config: RatioSplitConfig) -> dict[str, Path]:
    """Build strict date-disjoint ratio splits and save CSV + summary JSON.

    Returns dict of output paths.
    """
    splits = config.splits or DEFAULT_SPLITS
    r1, r5, r2 = config.ratio
    if min(r1, r5, r2) <= 0:
        raise ValueError(f"Invalid ratio {config.ratio}; all values must be > 0.")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fire_1 = _read_class_csv(config.fire_1_csv, config.key_cols)
    fire_5 = _read_class_csv(config.fire_5_csv, config.key_cols)
    fre_2 = _read_class_csv(config.fre_2_csv, config.key_cols)

    rng = np.random.default_rng(config.seed)
    split_outputs: dict[str, pd.DataFrame] = {}
    summary: dict = {
        "seed": int(config.seed),
        "ratio": {"Fire-1": int(r1), "Fire-5": int(r5), "Fre-2": int(r2)},
        "allow_replacement_if_needed": bool(config.allow_replacement_if_needed),
        "splits": {},
    }

    for split_name, (start_date, end_date) in splits.items():
        s1 = _slice_by_date(fire_1, start_date, end_date)
        s5 = _slice_by_date(fire_5, start_date, end_date)
        s2 = _slice_by_date(fre_2, start_date, end_date)

        n1, n5, n2 = len(s1), len(s5), len(s2)
        k = min(n1 // r1, n5 // r5, n2 // r2)
        if k <= 0:
            raise ValueError(
                f"{split_name}: insufficient rows for ratio {config.ratio}. "
                f"Counts=(Fire-1={n1}, Fire-5={n5}, Fre-2={n2})"
            )

        t1, t5, t2 = r1 * k, r5 * k, r2 * k

        p1 = _sample_exact(s1, t1, rng=rng, replace=False)
        p5 = _sample_exact(s5, t5, rng=rng, replace=False)
        p2 = _sample_exact(s2, t2, rng=rng, replace=False)

        # Optional fallback for users who still want exact ratio even on short class subsets.
        if config.allow_replacement_if_needed:
            if len(p1) < t1:
                p1 = _sample_exact(s1, t1, rng=rng, replace=True)
            if len(p5) < t5:
                p5 = _sample_exact(s5, t5, rng=rng, replace=True)
            if len(p2) < t2:
                p2 = _sample_exact(s2, t2, rng=rng, replace=True)

        out = pd.concat([p1, p5, p2], ignore_index=True)
        out = out.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
        _assert_unique_units(out, config.key_cols, split_name)

        split_outputs[split_name] = out
        summary["splits"][split_name] = {
            "date_range": {"start": start_date, "end": end_date},
            "input_counts": {"Fire-1": int(n1), "Fire-5": int(n5), "Fre-2": int(n2)},
            "k_base": int(k),
            "selected_counts": {"Fire-1": int(t1), "Fire-5": int(t5), "Fre-2": int(t2)},
            "rows_total": int(len(out)),
        }

    _assert_disjoint_dates(split_outputs)

    out_paths: dict[str, Path] = {}
    for split_name, df in split_outputs.items():
        p = out_dir / f"{split_name}_132.csv"
        save_df = df.copy()
        save_df["target_date"] = pd.to_datetime(save_df["target_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        save_df.to_csv(p, index=False)
        out_paths[f"{split_name}_csv"] = p

    summary_path = out_dir / "split_132_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_paths["summary_json"] = summary_path
    return out_paths
