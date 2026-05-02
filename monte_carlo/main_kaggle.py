from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch


# ------------------------
# Project path setup
# ------------------------
candidate_roots = [
    Path("/kaggle/working/RealWork"),
    Path("/kaggle/working"),
    Path.cwd(),
]
project_root = None
for p in candidate_roots:
    if (p / "Futures").exists():
        project_root = p
        break
if project_root is None:
    raise RuntimeError("Could not find project root containing Futures/ directory")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Project root:", project_root)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"gpu[{i}]:", torch.cuda.get_device_name(i))

from Futures.future_mc_forecast import (  # noqa: E402
    CellTrainingPoint,
    StreamingMCTrainConfig,
    run_recursive_future_forecast,
    train_future_probability_mlp_streaming_mc,
)


# ------------------------
# User config (Kaggle)
# ------------------------
# Example expected files inside dataset:
#   FEATURES_train.csv
#   FEATURES_train_viirs.csv (optional)
#   LABELS_train.csv
TRAIN_DATA_DIR = Path("/kaggle/input/YOUR_DATASET_NAME")  # change this
USE_SYNTHETIC_IF_MISSING = True

FEATURES_BASE_CSV = TRAIN_DATA_DIR / "FEATURES_train.csv"
FEATURES_VIIRS_CSV = TRAIN_DATA_DIR / "FEATURES_train_viirs.csv"  # optional
LABELS_CSV = TRAIN_DATA_DIR / "LABELS_train.csv"

# If current probability from your existing method is already in a feature column,
# set that column name here. Fallback is ignition_prob_clim.
CURRENT_PROBABILITY_COLUMN = "ignition_prob_clim"

# Optional: if you already have variables in memory (vars), set these manually in notebook
# and call build_points_from_dataframe() directly.
EXTRA_FEATURE_KEY_COLS = ("target_date", "row", "col", "window_id")


def configure_paths_from_directory(
    train_data_dir: str | Path,
    *,
    features_base_name: str = "FEATURES_train.csv",
    features_viirs_name: str = "FEATURES_train_viirs.csv",
    labels_name: str = "LABELS_train.csv",
) -> None:
    """Set global CSV paths from a local/Kaggle input directory."""
    global TRAIN_DATA_DIR, FEATURES_BASE_CSV, FEATURES_VIIRS_CSV, LABELS_CSV
    TRAIN_DATA_DIR = Path(train_data_dir)
    FEATURES_BASE_CSV = TRAIN_DATA_DIR / features_base_name
    FEATURES_VIIRS_CSV = TRAIN_DATA_DIR / features_viirs_name
    LABELS_CSV = TRAIN_DATA_DIR / labels_name
    print("Configured dataset dir:", TRAIN_DATA_DIR)
    print("FEATURES_BASE_CSV:", FEATURES_BASE_CSV)
    print("FEATURES_VIIRS_CSV:", FEATURES_VIIRS_CSV)
    print("LABELS_CSV:", LABELS_CSV)


def configure_paths_from_hf_repo(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str | None = None,
    repo_type: str = "dataset",
    features_base_name: str = "FEATURES_train.csv",
    features_viirs_name: str = "FEATURES_train_viirs.csv",
    labels_name: str = "LABELS_train.csv",
    cache_dir: str | Path = "/kaggle/working/hf_cache",
) -> None:
    """Download required CSVs from Hugging Face Hub and set global paths.

    Example:
        configure_paths_from_hf_repo(
            repo_id="username/your-dataset",
            token=os.getenv("HF_TOKEN"),
        )
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required. Install with: pip install -U huggingface_hub"
        ) from exc

    base_fp = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=features_base_name,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir),
        )
    )
    labels_fp = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=labels_name,
            repo_type=repo_type,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir),
        )
    )

    viirs_fp: Path | None = None
    try:
        viirs_fp = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=features_viirs_name,
                repo_type=repo_type,
                revision=revision,
                token=token,
                cache_dir=str(cache_dir),
            )
        )
    except Exception:
        viirs_fp = None

    global TRAIN_DATA_DIR, FEATURES_BASE_CSV, FEATURES_VIIRS_CSV, LABELS_CSV
    TRAIN_DATA_DIR = base_fp.parent
    FEATURES_BASE_CSV = base_fp
    LABELS_CSV = labels_fp
    FEATURES_VIIRS_CSV = viirs_fp if viirs_fp is not None else Path("/__missing_viirs__")

    print("Configured from Hugging Face repo:", repo_id)
    print("FEATURES_BASE_CSV:", FEATURES_BASE_CSV)
    print("LABELS_CSV:", LABELS_CSV)
    if viirs_fp is not None:
        print("FEATURES_VIIRS_CSV:", FEATURES_VIIRS_CSV)
    else:
        print("FEATURES_VIIRS_CSV: not found (continuing without VIIRS merge)")


def _norm_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_date"] = pd.to_datetime(out["target_date"], errors="coerce").dt.date
    out["row"] = pd.to_numeric(out["row"], errors="coerce").astype("Int64")
    out["col"] = pd.to_numeric(out["col"], errors="coerce").astype("Int64")
    out["window_id"] = pd.to_numeric(out["window_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["target_date", "row", "col", "window_id"])
    out["row"] = out["row"].astype(np.int64)
    out["col"] = out["col"].astype(np.int64)
    out["window_id"] = out["window_id"].astype(np.int64)
    return out


def load_window_feature_tables(
    features_base_csv: Path,
    labels_csv: Path,
    features_viirs_csv: Path | None = None,
    extra_feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load long-format window tables and merge labels + optional VIIRS."""
    if not features_base_csv.exists():
        raise FileNotFoundError(features_base_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    feat = pd.read_csv(features_base_csv)
    feat = _norm_key_columns(feat)

    key_cols = ["target_date", "row", "col", "window_id"]
    meta_cols = {"target_date", "row", "col", "window_id", "split", "source_file"}

    if features_viirs_csv is not None and features_viirs_csv.exists():
        viirs = pd.read_csv(features_viirs_csv)
        viirs = _norm_key_columns(viirs)
        viirs_drop = [c for c in key_cols if c in viirs.columns]
        viirs_feats = [c for c in viirs.columns if c not in viirs_drop and c not in {"split", "source_file"}]
        viirs = viirs[key_cols + viirs_feats]
        feat = feat.merge(viirs, on=key_cols, how="left", suffixes=("", "_viirsdup"))
        dup_cols = [c for c in feat.columns if c.endswith("_viirsdup")]
        if dup_cols:
            feat = feat.drop(columns=dup_cols)
        print(f"Merged VIIRS features from: {features_viirs_csv}")

    if extra_feature_df is not None:
        ex = _norm_key_columns(extra_feature_df)
        ex_feats = [c for c in ex.columns if c not in EXTRA_FEATURE_KEY_COLS]
        ex = ex[list(EXTRA_FEATURE_KEY_COLS) + ex_feats]
        feat = feat.merge(ex, on=key_cols, how="left", suffixes=("", "_extradup"))
        dup_cols = [c for c in feat.columns if c.endswith("_extradup")]
        if dup_cols:
            feat = feat.drop(columns=dup_cols)
        print("Merged extra in-memory feature vars.")

    labels = pd.read_csv(labels_csv)
    need_y = {"target_date", "row", "col", "label"}
    missing_y = [c for c in need_y if c not in labels.columns]
    if missing_y:
        raise ValueError(f"{labels_csv} missing label columns: {missing_y}")
    labels["target_date"] = pd.to_datetime(labels["target_date"], errors="coerce").dt.date
    labels["row"] = pd.to_numeric(labels["row"], errors="coerce")
    labels["col"] = pd.to_numeric(labels["col"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["target_date", "row", "col", "label"]).copy()
    labels["row"] = labels["row"].astype(np.int64)
    labels["col"] = labels["col"].astype(np.int64)
    labels = labels[["target_date", "row", "col", "label"]].drop_duplicates(
        subset=["target_date", "row", "col"],
        keep="first",
    )

    df = feat.merge(labels, on=["target_date", "row", "col"], how="inner")
    # Keep only full 4-window keys.
    win_counts = df.groupby(["target_date", "row", "col"], sort=False)["window_id"].nunique()
    keep_keys = win_counts[win_counts == 4].index.to_frame(index=False)
    df = df.merge(keep_keys, on=["target_date", "row", "col"], how="inner")
    df = df.sort_values(["target_date", "row", "col", "window_id"]).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in meta_cols and c not in {"label"}]
    print("Loaded merged long table.")
    print("rows:", len(df), "| unique keys:", df[["target_date", "row", "col"]].drop_duplicates().shape[0])
    print("feature count:", len(feat_cols))
    return df


def build_points_from_dataframe(
    df_long: pd.DataFrame,
    current_probability_col: str = CURRENT_PROBABILITY_COLUMN,
    horizon_days: int = 7,
    window_size: int = 16,
) -> list[CellTrainingPoint]:
    """Convert long window CSV rows into CellTrainingPoint list.

    Note:
    - Your table has 4 windows (window_id=1..4), not daily 64-day rows.
    - We reconstruct pseudo-daily history by repeating each 16-day window vector
      for 16 days (oldest->latest: 4,3,2,1) to get [F,64].
    """
    key_cols = ["target_date", "row", "col", "window_id"]
    meta_cols = set(key_cols + ["split", "source_file", "label"])
    feat_cols = [c for c in df_long.columns if c not in meta_cols]
    if not feat_cols:
        raise ValueError("No feature columns found in long dataframe.")
    if current_probability_col not in feat_cols:
        raise ValueError(
            f"current_probability_col='{current_probability_col}' not found. "
            f"Available examples: {feat_cols[:10]}"
        )

    points: list[CellTrainingPoint] = []
    grouped = df_long.groupby(["target_date", "row", "col"], sort=False)
    for (t_date, row, col), g in grouped:
        g = g.sort_values("window_id")
        wids = g["window_id"].tolist()
        if wids != [1, 2, 3, 4]:
            continue

        # Feature window matrix [4,F] in window_id order 1..4.
        mat_w = g[feat_cols].to_numpy(dtype=np.float32)
        mat_w = np.nan_to_num(mat_w, nan=0.0, posinf=0.0, neginf=0.0)

        # current_features uses most recent window (window_id=1).
        current_features = mat_w[0].copy()  # [F]

        # Build pseudo daily history [F,64] from windows oldest->latest: 4,3,2,1.
        # Each 16-day window summary is repeated 16 times.
        oldest_to_recent = np.stack([mat_w[3], mat_w[2], mat_w[1], mat_w[0]], axis=0)  # [4,F]
        seq_64_f = np.repeat(oldest_to_recent, repeats=window_size, axis=0)  # [64,F]
        history_f64 = seq_64_f.T  # [F,64]

        label_val = float(g["label"].iloc[0])
        current_prob = float(g.loc[g["window_id"] == 1, current_probability_col].iloc[0])

        # Here label date is target_date. To keep training tuple semantics for t+h,
        # we map anchor_date = target_date - horizon_days.
        target_dt = date.fromisoformat(str(t_date))
        anchor_dt = target_dt - timedelta(days=int(horizon_days))

        points.append(
            CellTrainingPoint(
                anchor_date=anchor_dt.isoformat(),
                target_date=target_dt.isoformat(),
                row=int(row),
                col=int(col),
                history_features=torch.as_tensor(history_f64, dtype=torch.float32),
                current_features=torch.as_tensor(current_features, dtype=torch.float32),
                current_probability_at_t=current_prob,
                observed_t_plus_h_label_or_probability=label_val,
                future_day_of_year_t_plus_h=int(target_dt.timetuple().tm_yday),
            )
        )
    print("Built CellTrainingPoint list:", len(points))
    return points


def make_synthetic_points(n: int = 1200, f: int = 8) -> list[CellTrainingPoint]:
    points: list[CellTrainingPoint] = []
    for i in range(n):
        points.append(
            CellTrainingPoint(
                anchor_date="2026-04-01",
                target_date="2026-04-08",
                row=i,
                col=i % 100,
                history_features=torch.randn(f, 64),
                current_features=torch.randn(f),
                current_probability_at_t=float(np.random.rand()),
                observed_t_plus_h_label_or_probability=float(np.random.rand() > 0.8),
                future_day_of_year_t_plus_h=99,
            )
        )
    return points


def load_points() -> list[CellTrainingPoint]:
    if FEATURES_BASE_CSV.exists() and LABELS_CSV.exists():
        viirs = FEATURES_VIIRS_CSV if FEATURES_VIIRS_CSV.exists() else None
        df_long = load_window_feature_tables(
            features_base_csv=FEATURES_BASE_CSV,
            labels_csv=LABELS_CSV,
            features_viirs_csv=viirs,
            extra_feature_df=None,
        )
        return build_points_from_dataframe(
            df_long=df_long,
            current_probability_col=CURRENT_PROBABILITY_COLUMN,
            horizon_days=7,
            window_size=16,
        )

    if not USE_SYNTHETIC_IF_MISSING:
        raise FileNotFoundError(
            "Could not find FEATURES/LABELS CSVs. "
            "Set TRAIN_DATA_DIR correctly or enable synthetic fallback."
        )

    print("Using synthetic demo points (replace with real Kaggle data).")
    points = make_synthetic_points(n=1200, f=8)
    print("Synthetic points:", len(points))
    return points


def _save_model_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, ckpt_path)
    print("Saved checkpoint:", ckpt_path)


def main() -> None:
    points = load_points()
    if not points:
        raise RuntimeError("No training points were created from input data.")

    # Requested setup: use only 1000 points and run 10000 MC simulations each.
    stream_cfg = StreamingMCTrainConfig(
        max_train_points=1000,
        mc_samples_per_point=10000,
        horizon_days=7,
        mc_batch_size=512,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=(64, 32),
        activation="gelu",
        dropout=0.0,
        sigma_floor=1e-6,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        selection_mode="random",
        use_data_parallel=True,
        gpu_ids=(0, 1),
    )

    train_out = train_future_probability_mlp_streaming_mc(points, stream_cfg)
    print("selected_points_count:", train_out["selected_points_count"])
    print("total_simulations:", train_out["total_simulations"])
    print("history tail:", train_out["history"][-1])

    ckpt_path = Path("/kaggle/working/futures_ckpt/future_mlp.pt")
    _save_model_checkpoint(train_out["model"], ckpt_path)

    # Single-cell inference example
    p0 = points[0]
    infer_out = run_recursive_future_forecast(
        history_features=p0.history_features,
        current_features=p0.current_features,
        current_probability=p0.current_probability_at_t,
        future_day_of_year=int(p0.future_day_of_year_t_plus_h),
        model_checkpoint=ckpt_path,
        horizon_days=7,
        num_mc_samples=100000,
        mc_batch_size=1024,
        output_dir="/kaggle/working/future_forecast_outputs",
        seed=42,
        allow_random_model=False,
        use_data_parallel=True,
        gpu_ids=(0, 1),
    )
    print("probability_mean:", infer_out["probability_mean"])
    print("probability_std:", infer_out["probability_std"])


if __name__ == "__main__":
    main()
