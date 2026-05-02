"""
Transform VIIRS fire archive CSVs → historical ignition probability parquet.
Already computed in download/07 — this just validates and re-exports
to the standard data_processed/ location.

Output schema: lat_bin, lon_bin, month, ignition_prob
(static — no time dimension; join on lat/lon + calendar month at model time)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC

import pandas as pd

PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "historical_ignition.parquet"
SRC = DIRS["hist_ignition"] / "ignition_prob_monthly.parquet"

if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

if not SRC.exists():
    print("[ERROR] Source not found. Run download/07 first.")
    sys.exit(1)

df = pd.read_parquet(SRC)
print(f"Loaded {len(df)} cell-month records")

# rename to standard lat/lon for consistent joins
df = df.rename(columns={"lat_bin": "lat", "lon_bin": "lon"})

# validate range
assert df["ignition_prob"].between(0, 1).all(), "probabilities out of range"
assert df["month"].between(1, 12).all(), "month out of range"

df = df.sort_values(["month", "lat", "lon"]).reset_index(drop=True)
df.to_parquet(OUT, index=False)
print(f"[DONE] {OUT}  shape={df.shape}")
