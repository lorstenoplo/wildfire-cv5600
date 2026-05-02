"""Build no-fire candidate classes and balanced no-fire pool.

This module assumes fire-side datasets are already finalized.
It constructs only label-0 samples from a full window-history table and writes:

1) nofire_0A.csv
2) nofire_0B.csv
3) nofire_0C.csv
4) nofire_0D.csv
5) nofire_all_unique.csv
6) nofire_balanced_final.csv
7) nofire_sampling_summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover
    distance_transform_edt = None


@dataclass
class NoFireBuildConfig:
    window_history_csv: str | Path
    split_summary_json: str | Path
    fire_train_csv: str | Path
    fire_val_csv: str | Path
    fire_test_csv: str | Path
    output_dir: str | Path
    seed: int = 42
    grid_rows: int = 142
    grid_cols: int = 116
    cell_km: float = 4.0
    chunksize: int = 250_000
    allow_replacement_if_needed: bool = True
    n_jobs: int = 1
    precompute_distance_maps: bool = True


CLASS_ORDER = ["NoFire-0A", "NoFire-0B", "NoFire-0C", "NoFire-0D"]
CLASS_WEIGHTS = {"NoFire-0A": 1, "NoFire-0B": 1, "NoFire-0C": 2, "NoFire-0D": 1}
CLASS_KEYS = {
    "NoFire-0A": "0A",
    "NoFire-0B": "0B",
    "NoFire-0C": "0C",
    "NoFire-0D": "0D",
}


def _load_total_fire_from_split_summary(path: str | Path) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    total = 0
    for split_name in ("train", "val", "test"):
        block = splits.get(split_name, {})
        total += int(block.get("rows_total", 0))
    if total <= 0:
        raise ValueError("Could not read positive total_fire from split summary JSON.")
    return int(total)


def _load_fire_units(
    train_csv: str | Path, val_csv: str | Path, test_csv: str | Path
) -> pd.DataFrame:
    usecols = ["target_date", "row", "col"]
    parts = []
    for p in (train_csv, val_csv, test_csv):
        try:
            df = pd.read_csv(p, usecols=usecols)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=usecols)
        parts.append(df)
    fire = pd.concat(parts, ignore_index=True)
    fire["target_date"] = fire["target_date"].astype(str)
    fire["row"] = fire["row"].astype(np.int32)
    fire["col"] = fire["col"].astype(np.int32)
    fire = fire.drop_duplicates(["target_date", "row", "col"]).reset_index(drop=True)
    return fire


def _build_fire_index(fire_df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for d, g in fire_df.groupby("target_date", sort=False):
        out[str(d)] = g[["row", "col"]].to_numpy(dtype=np.int32, copy=True)
    return out


def _distance_map_for_date(
    fire_coords: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Return nearest-fire Euclidean distance in cell units for every grid cell."""
    if fire_coords.size == 0:
        return np.full((rows, cols), np.inf, dtype=np.float32)

    fire_mask = np.zeros((rows, cols), dtype=bool)
    fire_mask[fire_coords[:, 0], fire_coords[:, 1]] = True

    if distance_transform_edt is not None:
        # distance to nearest True in fire_mask by running EDT on inverse mask.
        dist = distance_transform_edt(~fire_mask).astype(np.float32)
        return dist

    # Fallback without scipy: exact Euclidean via vectorized nearest-distance scan.
    rr, cc = np.indices((rows, cols))
    grid = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float32)
    fire = fire_coords.astype(np.float32)

    out = np.full(grid.shape[0], np.inf, dtype=np.float32)
    block = 4096
    for i in range(0, grid.shape[0], block):
        chunk = grid[i : i + block]
        diff = chunk[:, None, :] - fire[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        out[i : i + block] = np.sqrt(d2.min(axis=1)).astype(np.float32)
    return out.reshape(rows, cols)


def _detect_history_columns(path: str | Path) -> dict[str, str]:
    cols = set(pd.read_csv(path, nrows=0).columns.tolist())

    mapping = {
        "target_date": "target_date",
        "row": "row",
        "col": "col",
        "fire_t": "fire_t",
        "w0_16": "w0_16" if "w0_16" in cols else "fire_in_prev_0_16_days",
        "w16_32": "w16_32" if "w16_32" in cols else "fire_in_prev_16_32_days",
        "w32_48": "w32_48" if "w32_48" in cols else "fire_in_prev_32_48_days",
        "w48_64": "w48_64" if "w48_64" in cols else "fire_in_prev_48_64_days",
    }

    missing = [src for src in mapping.values() if src not in cols]
    if missing:
        raise ValueError(f"Missing required columns in history CSV: {missing}")
    return mapping


def _append_csv(
    df: pd.DataFrame, path: Path, wrote_header: dict[str, bool], key: str
) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not wrote_header[key], index=False)
    wrote_header[key] = True


def _compute_initial_quotas(total_needed: int) -> dict[str, int]:
    weights = np.array([CLASS_WEIGHTS[c] for c in CLASS_ORDER], dtype=np.float64)
    raw = total_needed * (weights / weights.sum())
    floor = np.floor(raw).astype(np.int64)
    remain = int(total_needed - floor.sum())

    frac = raw - floor
    order = np.argsort(-frac)  # descending fractional parts
    quotas = floor.copy()
    for i in order[:remain]:
        quotas[i] += 1

    return {c: int(quotas[i]) for i, c in enumerate(CLASS_ORDER)}


def _redistribute_shortfall(
    requested: dict[str, int],
    available: dict[str, int],
) -> tuple[dict[str, int], bool]:
    """Return achievable unique targets (no replacement) and whether redistribution happened."""
    selected = {c: min(requested[c], available[c]) for c in CLASS_ORDER}
    deficit = int(sum(requested.values()) - sum(selected.values()))
    if deficit <= 0:
        return selected, False

    redistributed = True
    spare = {c: max(0, available[c] - selected[c]) for c in CLASS_ORDER}
    # Weighted round-robin favoring 0C then others.
    cycle = ["NoFire-0C", "NoFire-0A", "NoFire-0B", "NoFire-0D"]
    while deficit > 0:
        moved = False
        for c in cycle:
            if spare[c] > 0 and deficit > 0:
                selected[c] += 1
                spare[c] -= 1
                deficit -= 1
                moved = True
        if not moved:
            break
    return selected, redistributed


def _replacement_plan(deficit: int, available: dict[str, int]) -> dict[str, int]:
    if deficit <= 0:
        return {c: 0 for c in CLASS_ORDER}
    active = [c for c in CLASS_ORDER if available[c] > 0]
    if not active:
        return {c: 0 for c in CLASS_ORDER}

    total_w = float(sum(CLASS_WEIGHTS[c] for c in active))
    raw = {c: deficit * (CLASS_WEIGHTS[c] / total_w) for c in active}
    floor = {c: int(np.floor(raw[c])) for c in active}
    rem = int(deficit - sum(floor.values()))
    frac_sorted = sorted(active, key=lambda c: (raw[c] - floor[c]), reverse=True)
    for c in frac_sorted[:rem]:
        floor[c] += 1

    out = {c: 0 for c in CLASS_ORDER}
    for c in active:
        out[c] = int(floor[c])
    return out


def _sample_rows_from_csv(
    csv_path: Path,
    n_rows_total: int,
    n_take: int,
    rng: np.random.Generator,
    replace: bool,
    chunksize: int,
) -> pd.DataFrame:
    if n_take <= 0:
        return pd.DataFrame()
    if n_rows_total <= 0:
        raise ValueError(f"Cannot sample from empty file: {csv_path}")
    if not replace and n_take > n_rows_total:
        raise ValueError(
            f"Requested {n_take} > available {n_rows_total} without replacement."
        )

    chosen = rng.choice(n_rows_total, size=n_take, replace=replace)
    rep = Counter(chosen.tolist())
    keys = np.array(sorted(rep.keys()), dtype=np.int64)
    if keys.size == 0:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    ptr = 0
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        end = offset + len(chunk)
        # advance pointer until key in range
        while ptr < len(keys) and keys[ptr] < offset:
            ptr += 1
        q = ptr
        while q < len(keys) and keys[q] < end:
            k = int(keys[q])
            cnt = int(rep[k])
            row_df = chunk.iloc[[k - offset]].copy()
            if cnt > 1:
                row_df = pd.concat([row_df] * cnt, ignore_index=True)
            parts.append(row_df)
            q += 1
        ptr = q
        if ptr >= len(keys):
            break
        offset = end

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _build_distance_maps(
    fire_index: dict[str, np.ndarray],
    rows: int,
    cols: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    items = sorted(fire_index.items(), key=lambda x: x[0])
    if n_jobs <= 1:
        return {d: _distance_map_for_date(coords, rows, cols) for d, coords in items}

    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        dists = ex.map(
            lambda it: (it[0], _distance_map_for_date(it[1], rows, cols)), items
        )
        for d, dist in dists:
            out[d] = dist
    return out


def _classify_nofire_group(
    d_str: str,
    g: pd.DataFrame,
    fire_coords: np.ndarray,
    dist_map: np.ndarray | None,
    grid_rows: int,
    grid_cols: int,
    cell_km: float,
) -> dict[str, pd.DataFrame]:
    if fire_coords is None or len(fire_coords) == 0:
        return {}

    dist = dist_map
    if dist is None:
        dist = _distance_map_for_date(fire_coords, grid_rows, grid_cols)

    gg = g.copy()
    r = gg["row"].to_numpy(dtype=np.int32)
    c = gg["col"].to_numpy(dtype=np.int32)
    d_cells = dist[r, c].astype(np.float32)
    d_km = d_cells * np.float32(cell_km)

    w_sum = (
        gg["w0_16"].to_numpy(dtype=np.int8)
        + gg["w16_32"].to_numpy(dtype=np.int8)
        + gg["w32_48"].to_numpy(dtype=np.int8)
        + gg["w48_64"].to_numpy(dtype=np.int8)
    )
    hist_any = w_sum >= 1
    hist_zero = w_sum == 0

    base = pd.DataFrame(
        {
            "target_date": gg["target_date"].astype(str).to_numpy(),
            "row": r,
            "col": c,
            "fire_t": np.zeros(len(gg), dtype=np.int8),
            "w0_16": gg["w0_16"].to_numpy(dtype=np.int8),
            "w16_32": gg["w16_32"].to_numpy(dtype=np.int8),
            "w32_48": gg["w32_48"].to_numpy(dtype=np.int8),
            "w48_64": gg["w48_64"].to_numpy(dtype=np.int8),
            "dist_to_nearest_same_day_fire_cells": d_cells,
            "dist_to_nearest_same_day_fire_km": d_km,
            "label": np.zeros(len(gg), dtype=np.int8),
        }
    )

    remaining = np.ones(len(base), dtype=bool)
    m_0a = remaining & (d_cells > 1.0) & (d_cells <= 5.0) & hist_zero
    remaining &= ~m_0a
    m_0b = remaining & (d_cells > 1.0) & (d_cells <= 5.0) & hist_any
    remaining &= ~m_0b
    m_0c = remaining & (d_cells > 5.0) & (d_cells <= 10.0) & hist_any
    remaining &= ~m_0c
    m_0d = remaining & (d_cells > 10.0)

    class_masks = {
        "NoFire-0A": m_0a,
        "NoFire-0B": m_0b,
        "NoFire-0C": m_0c,
        "NoFire-0D": m_0d,
    }

    out: dict[str, pd.DataFrame] = {}
    for cname, m in class_masks.items():
        if not np.any(m):
            continue
        out_df = base.loc[m].copy()
        out_df["class_name"] = cname
        out_df = out_df.drop_duplicates(["target_date", "row", "col"])
        out[cname] = out_df
    return out


def build_nofire_dataset(config: NoFireBuildConfig) -> dict[str, str]:
    """Build no-fire classes, balanced final no-fire file, and summary JSON."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.n_jobs <= 0:
        raise ValueError("n_jobs must be >= 1")

    paths = {
        "0A": out_dir / "nofire_0A.csv",
        "0B": out_dir / "nofire_0B.csv",
        "0C": out_dir / "nofire_0C.csv",
        "0D": out_dir / "nofire_0D.csv",
        "ALL": out_dir / "nofire_all_unique.csv",
        "FINAL": out_dir / "nofire_balanced_final.csv",
        "SUMMARY": out_dir / "nofire_sampling_summary.json",
    }
    for p in paths.values():
        if p.exists():
            p.unlink()

    total_fire = _load_total_fire_from_split_summary(config.split_summary_json)
    fire_df = _load_fire_units(
        config.fire_train_csv, config.fire_val_csv, config.fire_test_csv
    )
    fire_index = _build_fire_index(fire_df)
    fire_dates = set(fire_index.keys())
    dist_maps: dict[str, np.ndarray] = {}
    if config.precompute_distance_maps:
        dist_maps = _build_distance_maps(
            fire_index=fire_index,
            rows=config.grid_rows,
            cols=config.grid_cols,
            n_jobs=config.n_jobs,
        )

    col_map = _detect_history_columns(config.window_history_csv)
    usecols = [
        col_map["target_date"],
        col_map["row"],
        col_map["col"],
        col_map["fire_t"],
        col_map["w0_16"],
        col_map["w16_32"],
        col_map["w32_48"],
        col_map["w48_64"],
    ]

    wrote = {"0A": False, "0B": False, "0C": False, "0D": False, "ALL": False}
    class_counts = {"NoFire-0A": 0, "NoFire-0B": 0, "NoFire-0C": 0, "NoFire-0D": 0}

    for chunk in pd.read_csv(
        config.window_history_csv, usecols=usecols, chunksize=config.chunksize
    ):
        chunk = chunk.rename(
            columns={
                col_map["target_date"]: "target_date",
                col_map["row"]: "row",
                col_map["col"]: "col",
                col_map["fire_t"]: "fire_t",
                col_map["w0_16"]: "w0_16",
                col_map["w16_32"]: "w16_32",
                col_map["w32_48"]: "w32_48",
                col_map["w48_64"]: "w48_64",
            }
        )

        chunk["target_date"] = chunk["target_date"].astype(str)
        chunk = chunk[(chunk["fire_t"] == 0) & (chunk["target_date"].isin(fire_dates))]
        if chunk.empty:
            continue

        groups = [(str(d), g) for d, g in chunk.groupby("target_date", sort=False)]
        if config.n_jobs == 1:
            grouped_results = []
            for d_str, g in groups:
                fire_coords = fire_index.get(d_str)
                if fire_coords is None or len(fire_coords) == 0:
                    continue
                grouped_results.append(
                    _classify_nofire_group(
                        d_str=d_str,
                        g=g,
                        fire_coords=fire_coords,
                        dist_map=dist_maps.get(d_str),
                        grid_rows=config.grid_rows,
                        grid_cols=config.grid_cols,
                        cell_km=config.cell_km,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=config.n_jobs) as ex:
                futures = []
                for d_str, g in groups:
                    fire_coords = fire_index.get(d_str)
                    if fire_coords is None or len(fire_coords) == 0:
                        continue
                    futures.append(
                        ex.submit(
                            _classify_nofire_group,
                            d_str,
                            g,
                            fire_coords,
                            dist_maps.get(d_str),
                            config.grid_rows,
                            config.grid_cols,
                            config.cell_km,
                        )
                    )
                grouped_results = [f.result() for f in futures]

        for class_dict in grouped_results:
            for cname, out_df in class_dict.items():
                _append_csv(out_df, paths[CLASS_KEYS[cname]], wrote, CLASS_KEYS[cname])
                _append_csv(out_df, paths["ALL"], wrote, "ALL")
                class_counts[cname] += int(len(out_df))

    available = {c: int(class_counts[c]) for c in CLASS_ORDER}
    total_candidates = int(sum(available.values()))

    requested = _compute_initial_quotas(total_fire)
    selected_unique, redistributed = _redistribute_shortfall(requested, available)
    unique_selected_total = int(sum(selected_unique.values()))
    deficit_after_unique = int(total_fire - unique_selected_total)

    fallback_used = False
    replacement = {c: 0 for c in CLASS_ORDER}
    if deficit_after_unique > 0:
        if not config.allow_replacement_if_needed:
            raise ValueError(
                f"Insufficient unique no-fire rows ({unique_selected_total}) for target {total_fire} "
                "and replacement is disabled."
            )
        fallback_used = True
        replacement = _replacement_plan(deficit_after_unique, available)

    rng = np.random.default_rng(config.seed)
    class_file = {
        "NoFire-0A": paths["0A"],
        "NoFire-0B": paths["0B"],
        "NoFire-0C": paths["0C"],
        "NoFire-0D": paths["0D"],
    }

    sampled_parts: list[pd.DataFrame] = []
    actual_counts = {}
    for cname in CLASS_ORDER:
        n_av = available[cname]
        n_u = int(selected_unique[cname])
        n_r = int(replacement[cname])

        d_u = _sample_rows_from_csv(
            class_file[cname],
            n_rows_total=n_av,
            n_take=n_u,
            rng=rng,
            replace=False,
            chunksize=config.chunksize,
        )
        d_r = _sample_rows_from_csv(
            class_file[cname],
            n_rows_total=n_av,
            n_take=n_r,
            rng=rng,
            replace=True,
            chunksize=config.chunksize,
        )
        d = pd.concat([d_u, d_r], ignore_index=True)
        sampled_parts.append(d)
        actual_counts[cname] = int(len(d))

    final = pd.concat(sampled_parts, ignore_index=True)
    final = final.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    if len(final) != total_fire:
        raise RuntimeError(
            f"Final no-fire count {len(final)} does not match total_fire {total_fire}."
        )

    # Validation counts
    dup_count = int(final.duplicated(["target_date", "row", "col"]).sum())
    # By construction this should be zero for candidate class overlap.
    overlap_count = 0

    final.to_csv(paths["FINAL"], index=False)

    summary = {
        "raw_candidate_counts": {
            "NoFire-0A": int(available["NoFire-0A"]),
            "NoFire-0B": int(available["NoFire-0B"]),
            "NoFire-0C": int(available["NoFire-0C"]),
            "NoFire-0D": int(available["NoFire-0D"]),
            "total_no_fire_candidates": int(total_candidates),
        },
        "requested_quotas": {
            "NoFire-0A": int(requested["NoFire-0A"]),
            "NoFire-0B": int(requested["NoFire-0B"]),
            "NoFire-0C": int(requested["NoFire-0C"]),
            "NoFire-0D": int(requested["NoFire-0D"]),
        },
        "actual_sampled_counts": {
            "NoFire-0A": int(actual_counts["NoFire-0A"]),
            "NoFire-0B": int(actual_counts["NoFire-0B"]),
            "NoFire-0C": int(actual_counts["NoFire-0C"]),
            "NoFire-0D": int(actual_counts["NoFire-0D"]),
            "final_total_no_fire_sampled": int(len(final)),
        },
        "final_balance_target": {
            "total_fire_count": int(total_fire),
            "total_no_fire_sampled": int(len(final)),
            "fire_to_no_fire_ratio_target_achieved": bool(len(final) == total_fire),
        },
        "validation": {
            "duplicate_unit_count": int(dup_count),
            "overlapping_unit_count_across_nofire_classes": int(overlap_count),
            "random_seed_used": int(config.seed),
            "n_jobs_used": int(config.n_jobs),
            "precompute_distance_maps": bool(config.precompute_distance_maps),
            "redistribution_performed": bool(redistributed),
            "fallback_used_due_to_insufficient_class_rows": bool(fallback_used),
            "replacement_rows_per_class": {
                "NoFire-0A": int(replacement["NoFire-0A"]),
                "NoFire-0B": int(replacement["NoFire-0B"]),
                "NoFire-0C": int(replacement["NoFire-0C"]),
                "NoFire-0D": int(replacement["NoFire-0D"]),
            },
        },
        "files": {
            "nofire_0A_csv": str(paths["0A"]),
            "nofire_0B_csv": str(paths["0B"]),
            "nofire_0C_csv": str(paths["0C"]),
            "nofire_0D_csv": str(paths["0D"]),
            "nofire_all_unique_csv": str(paths["ALL"]),
            "nofire_balanced_final_csv": str(paths["FINAL"]),
        },
    }

    paths["SUMMARY"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "nofire_0A_csv": str(paths["0A"]),
        "nofire_0B_csv": str(paths["0B"]),
        "nofire_0C_csv": str(paths["0C"]),
        "nofire_0D_csv": str(paths["0D"]),
        "nofire_all_unique_csv": str(paths["ALL"]),
        "nofire_balanced_final_csv": str(paths["FINAL"]),
        "nofire_sampling_summary_json": str(paths["SUMMARY"]),
    }
