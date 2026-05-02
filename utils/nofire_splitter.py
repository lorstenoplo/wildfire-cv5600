"""Split no-fire classes into train/val/test with per-split 1:1:2:1 ratio.

Also writes renamed fire split files:
- FIRE_train.csv
- FIRE_val.csv
- FIRE_test.csv

And no-fire split files:
- NOFIRE_train.csv
- NOFIRE_val.csv
- NOFIRE_test.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd

NOFIRE_CLASSES = ["NoFire-0A", "NoFire-0B", "NoFire-0C", "NoFire-0D"]
NOFIRE_RATIO = {"NoFire-0A": 1, "NoFire-0B": 1, "NoFire-0C": 2, "NoFire-0D": 1}


@dataclass
class NoFireSplitConfig:
    nofire_0a_csv: str | Path
    nofire_0b_csv: str | Path
    nofire_0c_csv: str | Path
    nofire_0d_csv: str | Path
    split_summary_json: str | Path
    fire_train_csv: str | Path
    fire_val_csv: str | Path
    fire_test_csv: str | Path
    output_dir: str | Path
    seed: int = 42
    chunksize: int = 250_000
    allow_replacement_if_needed: bool = True
    date_col: str = "target_date"


def _read_split_targets(path: str | Path) -> dict[str, dict[str, str | int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    out: dict[str, dict[str, str | int]] = {}
    for name in ("train", "val", "test"):
        blk = splits.get(name, {})
        dr = blk.get("date_range", {})
        start = str(dr.get("start"))
        end = str(dr.get("end"))
        rows_total = int(blk.get("rows_total", 0))
        if not start or not end or rows_total <= 0:
            raise ValueError(f"Invalid split block for '{name}' in {path}")
        out[name] = {"start": start, "end": end, "rows_total": rows_total}
    return out


def _count_by_split(
    csv_path: Path,
    splits: dict[str, dict[str, str | int]],
    date_col: str,
    chunksize: int,
) -> dict[str, int]:
    counts = {k: 0 for k in splits.keys()}
    usecols = [date_col]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        d = chunk[date_col].astype(str)
        for s, cfg in splits.items():
            start = str(cfg["start"])
            end = str(cfg["end"])
            counts[s] += int(((d >= start) & (d <= end)).sum())
    return counts


def _compute_quotas(total_needed: int) -> dict[str, int]:
    w = np.array([NOFIRE_RATIO[c] for c in NOFIRE_CLASSES], dtype=np.float64)
    raw = total_needed * (w / w.sum())
    base = np.floor(raw).astype(np.int64)
    rem = int(total_needed - base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in order[:rem]:
        base[i] += 1
    return {c: int(base[i]) for i, c in enumerate(NOFIRE_CLASSES)}


def _redistribute_shortfall(
    requested: dict[str, int],
    available: dict[str, int],
) -> tuple[dict[str, int], bool]:
    selected = {c: min(requested[c], available[c]) for c in NOFIRE_CLASSES}
    deficit = int(sum(requested.values()) - sum(selected.values()))
    if deficit <= 0:
        return selected, False

    redistributed = True
    spare = {c: max(0, available[c] - selected[c]) for c in NOFIRE_CLASSES}
    # prioritize 0C first to preserve 1:1:2:1 intent
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
    out = {c: 0 for c in NOFIRE_CLASSES}
    if deficit <= 0:
        return out
    active = [c for c in NOFIRE_CLASSES if available[c] > 0]
    if not active:
        return out

    total_w = float(sum(NOFIRE_RATIO[c] for c in active))
    raw = {c: deficit * (NOFIRE_RATIO[c] / total_w) for c in active}
    base = {c: int(np.floor(raw[c])) for c in active}
    rem = int(deficit - sum(base.values()))
    frac_sorted = sorted(active, key=lambda c: (raw[c] - base[c]), reverse=True)
    for c in frac_sorted[:rem]:
        base[c] += 1
    for c in active:
        out[c] = int(base[c])
    return out


def _build_position_counter(
    n_available: int,
    n_unique: int,
    n_repl: int,
    rng: np.random.Generator,
) -> Counter:
    picks: list[int] = []
    if n_unique > 0:
        if n_unique > n_available:
            raise ValueError("n_unique > n_available with no replacement")
        picks.extend(rng.choice(n_available, size=n_unique, replace=False).tolist())
    if n_repl > 0:
        picks.extend(rng.choice(n_available, size=n_repl, replace=True).tolist())
    return Counter(picks)


def _extract_split_samples_from_class_file(
    csv_path: Path,
    splits: dict[str, dict[str, str | int]],
    counters: dict[str, Counter],
    date_col: str,
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    """Extract sampled rows for each split from one class CSV in one pass."""
    split_names = list(splits.keys())
    key_arrays = {
        s: np.array(sorted(counters[s].keys()), dtype=np.int64) for s in split_names
    }
    ptr = {s: 0 for s in split_names}
    offset = {s: 0 for s in split_names}
    out_parts = {s: [] for s in split_names}

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        d = chunk[date_col].astype(str)
        for s in split_names:
            start = str(splits[s]["start"])
            end = str(splits[s]["end"])
            mask = (d >= start) & (d <= end)
            if not bool(mask.any()):
                continue

            sub = chunk.loc[mask].reset_index(drop=True)
            n = len(sub)
            if n == 0:
                continue

            karr = key_arrays[s]
            p = ptr[s]
            lo = offset[s]
            hi = lo + n
            local_idx: list[int] = []

            while p < len(karr) and karr[p] < hi:
                k = int(karr[p])
                if k >= lo:
                    cnt = int(counters[s][k])
                    local = k - lo
                    local_idx.extend([local] * cnt)
                p += 1

            ptr[s] = p
            offset[s] = hi

            if local_idx:
                out_parts[s].append(sub.iloc[local_idx].copy())

    out = {}
    for s in split_names:
        if out_parts[s]:
            out[s] = pd.concat(out_parts[s], ignore_index=True)
        else:
            out[s] = pd.DataFrame()
    return out


def _copy_fire_splits(
    fire_train_csv: Path,
    fire_val_csv: Path,
    fire_test_csv: Path,
    out_dir: Path,
) -> dict[str, str]:
    fire_paths = {
        "train": fire_train_csv,
        "val": fire_val_csv,
        "test": fire_test_csv,
    }
    out = {}
    for s, src in fire_paths.items():
        df = pd.read_csv(src)
        dst = out_dir / f"FIRE_{s}.csv"
        df.to_csv(dst, index=False)
        out[f"FIRE_{s}_csv"] = str(dst)
    return out


def split_nofire_with_ratio(config: NoFireSplitConfig) -> dict[str, str]:
    """Create NOFIRE_train/val/test with 1:1:2:1 class ratio per split."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_files = {
        "NoFire-0A": Path(config.nofire_0a_csv),
        "NoFire-0B": Path(config.nofire_0b_csv),
        "NoFire-0C": Path(config.nofire_0c_csv),
        "NoFire-0D": Path(config.nofire_0d_csv),
    }
    splits = _read_split_targets(config.split_summary_json)

    # Count availability per class per split.
    available: dict[str, dict[str, int]] = {
        s: {c: 0 for c in NOFIRE_CLASSES} for s in splits
    }
    for c in NOFIRE_CLASSES:
        c_counts = _count_by_split(
            class_files[c], splits, config.date_col, config.chunksize
        )
        for s in splits:
            available[s][c] = int(c_counts[s])

    # Build sampling plan per split.
    requested: dict[str, dict[str, int]] = {}
    selected_unique: dict[str, dict[str, int]] = {}
    replacement: dict[str, dict[str, int]] = {}
    redistribution_flags: dict[str, bool] = {}
    fallback_flags: dict[str, bool] = {}

    for s, scfg in splits.items():
        target = int(scfg["rows_total"])
        req = _compute_quotas(target)
        sel, red = _redistribute_shortfall(req, available[s])
        deficit = int(target - sum(sel.values()))
        rep = {c: 0 for c in NOFIRE_CLASSES}
        fallback = False
        if deficit > 0:
            if not config.allow_replacement_if_needed:
                raise ValueError(
                    f"{s}: cannot reach target rows_total={target} without replacement. "
                    f"Selected unique={sum(sel.values())}"
                )
            rep = _replacement_plan(deficit, available[s])
            fallback = True

        requested[s] = req
        selected_unique[s] = sel
        replacement[s] = rep
        redistribution_flags[s] = red
        fallback_flags[s] = fallback

    # Sample rows from each class file, split-wise, in one pass per class.
    rng = np.random.default_rng(config.seed)
    split_parts: dict[str, list[pd.DataFrame]] = {s: [] for s in splits}
    actual_counts: dict[str, dict[str, int]] = {
        s: {c: 0 for c in NOFIRE_CLASSES} for s in splits
    }

    for c in NOFIRE_CLASSES:
        counters_by_split: dict[str, Counter] = {}
        for s in splits:
            n_av = int(available[s][c])
            n_u = int(selected_unique[s][c])
            n_r = int(replacement[s][c])
            counters_by_split[s] = _build_position_counter(
                n_available=n_av,
                n_unique=n_u,
                n_repl=n_r,
                rng=rng,
            )

        extracted = _extract_split_samples_from_class_file(
            csv_path=class_files[c],
            splits=splits,
            counters=counters_by_split,
            date_col=config.date_col,
            chunksize=config.chunksize,
        )
        for s in splits:
            df = extracted[s]
            if not df.empty:
                split_parts[s].append(df)
            actual_counts[s][c] = int(len(df))

    out_paths: dict[str, str] = {}
    for s in splits:
        if split_parts[s]:
            df = pd.concat(split_parts[s], ignore_index=True)
        else:
            df = pd.DataFrame()
        df = df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

        # enforce target count
        target = int(splits[s]["rows_total"])
        if len(df) != target:
            raise RuntimeError(
                f"{s}: NOFIRE rows {len(df)} != target fire rows {target}"
            )

        dst = out_dir / f"NOFIRE_{s}.csv"
        df.to_csv(dst, index=False)
        out_paths[f"NOFIRE_{s}_csv"] = str(dst)

    out_paths.update(
        _copy_fire_splits(
            fire_train_csv=Path(config.fire_train_csv),
            fire_val_csv=Path(config.fire_val_csv),
            fire_test_csv=Path(config.fire_test_csv),
            out_dir=out_dir,
        )
    )

    summary = {
        "seed": int(config.seed),
        "ratio_per_split": {
            "NoFire-0A": 1,
            "NoFire-0B": 1,
            "NoFire-0C": 2,
            "NoFire-0D": 1,
        },
        "splits": {},
    }

    for s in splits:
        target = int(splits[s]["rows_total"])
        summary["splits"][s] = {
            "date_range": {
                "start": str(splits[s]["start"]),
                "end": str(splits[s]["end"]),
            },
            "target_rows_from_fire_split": target,
            "available_counts": {c: int(available[s][c]) for c in NOFIRE_CLASSES},
            "requested_quotas": {c: int(requested[s][c]) for c in NOFIRE_CLASSES},
            "actual_unique_plan": {
                c: int(selected_unique[s][c]) for c in NOFIRE_CLASSES
            },
            "replacement_plan": {c: int(replacement[s][c]) for c in NOFIRE_CLASSES},
            "actual_sampled_counts": {
                c: int(actual_counts[s][c]) for c in NOFIRE_CLASSES
            },
            "actual_total_rows": int(sum(actual_counts[s].values())),
            "redistribution_performed": bool(redistribution_flags[s]),
            "fallback_with_replacement_used": bool(fallback_flags[s]),
        }

    summary_path = out_dir / "nofire_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_paths["nofire_split_summary_json"] = str(summary_path)
    return out_paths
