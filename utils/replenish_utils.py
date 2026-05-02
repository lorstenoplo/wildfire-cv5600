"""Reusable helpers for split cleaning + replenishment pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

KEY_COLS = ["target_date", "row", "col"]


@dataclass(frozen=True)
class SplitLabelTables:
    fire_train: pd.DataFrame
    fire_val: pd.DataFrame
    fire_test: pd.DataFrame
    nofire_train: pd.DataFrame
    nofire_val: pd.DataFrame
    nofire_test: pd.DataFrame


def normalize_key_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_date"] = pd.to_datetime(
        out["target_date"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    out["row"] = pd.to_numeric(out["row"], errors="coerce").astype("Int64")
    out["col"] = pd.to_numeric(out["col"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["target_date", "row", "col"])
    out["row"] = out["row"].astype(np.int32)
    out["col"] = out["col"].astype(np.int32)
    return out


def key_set_from_df(df: pd.DataFrame) -> set[tuple[str, int, int]]:
    return set(
        map(tuple, df[KEY_COLS].drop_duplicates().itertuples(index=False, name=None))
    )


def drop_keys(df: pd.DataFrame, keys_to_drop: pd.DataFrame) -> pd.DataFrame:
    if keys_to_drop.empty:
        return df.copy()
    k = keys_to_drop[KEY_COLS].drop_duplicates().copy()
    k["__drop__"] = 1
    out = df.merge(k, on=KEY_COLS, how="left")
    out = out[out["__drop__"].isna()].drop(columns=["__drop__"]).reset_index(drop=True)
    return out


def pick_feature_cols(features_df: pd.DataFrame) -> list[str]:
    excluded = set(KEY_COLS + ["window_id", "split", "source_file"])
    return [c for c in features_df.columns if c not in excluded]


def bad_keys_from_features(
    features_df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    miss = features_df[feature_cols].isna().any(axis=1)
    bad = features_df.loc[miss, KEY_COLS].drop_duplicates().reset_index(drop=True)
    return bad


def load_split_labels(label_dir: Path) -> SplitLabelTables:
    def _r(name: str) -> pd.DataFrame:
        p = label_dir / name
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        return normalize_key_df(df)

    return SplitLabelTables(
        fire_train=_r("FIRE_train.csv"),
        fire_val=_r("FIRE_val.csv"),
        fire_test=_r("FIRE_test.csv"),
        nofire_train=_r("NOFIRE_train.csv"),
        nofire_val=_r("NOFIRE_val.csv"),
        nofire_test=_r("NOFIRE_test.csv"),
    )


def load_and_combine_raw_fire_pool(
    label_dir: Path, fire_files: tuple[str, ...]
) -> pd.DataFrame:
    frames = []
    for fn in fire_files:
        p = label_dir / fn
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df = normalize_key_df(df)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    return out


def load_and_combine_raw_nofire_pool(
    label_dir: Path, nofire_files: tuple[str, ...]
) -> pd.DataFrame:
    frames = []
    for fn in nofire_files:
        p = label_dir / fn
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df = normalize_key_df(df)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    return out


def attach_split_column(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split
    return out


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def key_df_to_tuples(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    k = df[KEY_COLS].drop_duplicates().reset_index(drop=True)
    return list(map(tuple, k.itertuples(index=False, name=None)))


def tuples_to_key_df(keys: list[tuple[str, int, int]]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=KEY_COLS)
    return pd.DataFrame(keys, columns=KEY_COLS)
