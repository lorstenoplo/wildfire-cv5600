"""
Transform GridMET NetCDF files → standardised Parquet.
Schema: time (daily), lat, lon, <feature_columns>
Output: data_processed/gridmet.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC, GRIDMET_VARS

import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "gridmet.parquet"
if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

COLS_TO_KEEP = {"time", "lat", "lon"} | set(GRIDMET_VARS.values())
CHUNK_DAYS = 30  # ~60 MB per chunk in memory

# ── Open all variables lazily ─────────────────────────────────────────────────
datasets = []
for var, label in GRIDMET_VARS.items():
    nc = DIRS["gridmet"] / f"{var}.nc"
    if not nc.exists():
        print(f"[SKIP] {nc} not found")
        continue
    print(f"Opening {var} ...")
    ds = xr.open_dataset(nc)

    nc_var_names = [v for v in ds.data_vars if v not in ("day", "time", "lat", "lon", "crs")]
    nc_var_name = nc_var_names[0] if nc_var_names else var

    time_dim = "day" if "day" in ds.dims else "time"
    ds_var = ds[[nc_var_name]].rename({time_dim: "time", nc_var_name: label})
    datasets.append(ds_var)

if not datasets:
    print("[ERROR] No GridMET files found. Run download/01 first.")
    sys.exit(1)

print("Merging NetCDF files (lazy)...")
merged = xr.merge(datasets, compat="override")
n_times = len(merged.time)
print(f"Processing {n_times} time steps in chunks of {CHUNK_DAYS} days...")

writer = None
total_rows = 0
try:
    for start in range(0, n_times, CHUNK_DAYS):
        end = min(start + CHUNK_DAYS, n_times)
        print(f"  chunk {start}–{end} / {n_times} ...", end="\r")

        chunk = merged.isel(time=slice(start, end)).load()

        # downcast float64 → float32 to halve memory
        for label in GRIDMET_VARS.values():
            if label in chunk and chunk[label].dtype == np.float64:
                chunk[label] = chunk[label].astype(np.float32)

        df = chunk.to_dataframe().dropna().reset_index()
        df = df[[c for c in df.columns if c in COLS_TO_KEEP]]
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        df.sort_values(["time", "lat", "lon"], inplace=True)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, table.schema)
        writer.write_table(table)

        total_rows += len(df)
        del chunk, df, table
finally:
    if writer:
        writer.close()

print(f"\n[DONE] {OUT}  total_rows={total_rows}")
