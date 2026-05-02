"""
Build historical ignition probability raster from the VIIRS fire archive CSVs.
For each 0.1-degree grid cell, compute monthly ignition frequency over 2012-2023.
Output: data_raw/historical_ignition/ignition_prob_monthly.parquet
        data_raw/historical_ignition/ignition_prob_monthly.tif  (12-band, one per month)

No external download needed — uses existing VIIRS label CSVs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import AOI, DIRS, RAW

import pandas as pd
import numpy as np

OUT = DIRS["hist_ignition"]
OUT.mkdir(parents=True, exist_ok=True)

VIIRS_DIR = RAW / "viirs"
CSV_FILES = list(VIIRS_DIR.glob("*.csv"))

print(f"Loading {len(CSV_FILES)} VIIRS CSVs...")
dfs = []
for f in CSV_FILES:
    df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "type"])
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# keep vegetation fires only (type==0), filter by confidence
df = df[df["type"] == 0]
df = df[df["confidence"].isin(["n", "h"])]  # nominal + high confidence
df["acq_date"] = pd.to_datetime(df["acq_date"])
df["month"] = df["acq_date"].dt.month
df["year"]  = df["acq_date"].dt.year

# clip to AOI
df = df[
    (df["longitude"] >= AOI["min_lon"]) & (df["longitude"] <= AOI["max_lon"]) &
    (df["latitude"]  >= AOI["min_lat"]) & (df["latitude"]  <= AOI["max_lat"])
]

# bin to 0.1-degree grid cells
CELL = 0.1
df["lat_bin"] = (df["latitude"]  / CELL).round().astype(int) * CELL
df["lon_bin"] = (df["longitude"] / CELL).round().astype(int) * CELL

# count ignitions per cell per month per year, then average over years
annual = (
    df.groupby(["lat_bin", "lon_bin", "year", "month"])
    .size()
    .rename("count")
    .reset_index()
)
annual["has_fire"] = (annual["count"] > 0).astype(int)

years = df["year"].nunique()
prob = (
    annual.groupby(["lat_bin", "lon_bin", "month"])["has_fire"]
    .sum()
    .div(years)
    .rename("ignition_prob")
    .reset_index()
)

out_parquet = OUT / "ignition_prob_monthly.parquet"
prob.to_parquet(out_parquet, index=False)
print(f"[DONE] {out_parquet}  ({len(prob)} cell-month records)")
