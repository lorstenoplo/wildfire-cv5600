"""
Transform VIIRS spectral GeoTIFFs (from GEE) → standardised Parquet.

VNP09GA  — monthly TIFs: vnp09ga_<YYYY>_<MM>.tif
             bands named YYYYMMDD_I1, YYYYMMDD_I2, ..., YYYYMMDD_EVI
VNP21A1D — yearly TIFs:  vnp21a1d_<YYYY>.tif  → YYYYMMDD_LST_1KM
VNP21A1N — yearly TIFs:  vnp21a1n_<YYYY>.tif  → YYYYMMDD_LST_1KM

Output schema: time, lat, lon, i1, i2, i3, m11, ndvi, evi, lst_day, lst_night
Output: data_processed/viirs_spectral.parquet
"""

import sys, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC

import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "viirs_spectral.parquet"
SRC = DIRS["viirs_spectral"]

if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

VNP09_BANDS = ["I1", "I2", "I3", "M11", "NDVI", "EVI"]


def _coords(src):
    rows, cols = np.indices((src.height, src.width))
    lons, lats = rio_xy(src.transform, rows.ravel(), cols.ravel())
    return np.array(lats, dtype=np.float32), np.array(lons, dtype=np.float32)


def read_daily_bands(tif_path, wanted_suffixes):
    """
    Read a multi-band TIF with bands named YYYYMMDD_<suffix>.
    Returns:
        coords : (lats, lons) float32 flat arrays
        day_data: {date_str: {suffix: float32 array}}
    """
    day_data = defaultdict(dict)
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        coords = _coords(src)
        for i, desc in enumerate(src.descriptions, start=1):
            if not desc:
                continue
            m = re.match(r"(\d{8})_(.+)", desc)
            if not m:
                continue
            date_str, suffix = m.group(1), m.group(2)
            if suffix not in wanted_suffixes:
                continue
            arr = src.read(i).ravel().astype(np.float32)
            if nodata is not None:
                arr[arr == nodata] = np.nan
            day_data[date_str][suffix] = arr
    return coords, dict(day_data)


# ── Pre-load LST into memory (small — 1 band/day, yearly TIFs) ───────────────
# lst_lookup[date_str] = {"lst_day": (lats, lons, arr), "lst_night": ...}
lst_lookup = {}

for kind, pattern, col in [
    ("lst_day",   "vnp21a1d_*.tif", "lst_day"),
    ("lst_night", "vnp21a1n_*.tif", "lst_night"),
]:
    for tif in sorted(SRC.glob(pattern)):
        print(f"  Loading {tif.stem} ...")
        coords, day_data = read_daily_bands(tif, {"LST_1KM"})
        for date_str, bands in day_data.items():
            if date_str not in lst_lookup:
                lst_lookup[date_str] = {}
            lst_lookup[date_str][col] = (coords[0], coords[1], bands["LST_1KM"])

# Build a lookup DataFrame per date for fast joining
# Round to 4 dp so 500m and 1km grids can match on lat/lon
lst_dfs = {}
for date_str, kinds in lst_lookup.items():
    parts = {}
    for col, (lats, lons, arr) in kinds.items():
        df = pd.DataFrame({
            "lat_r": np.round(lats, 4),
            "lon_r": np.round(lons, 4),
            col:     arr,
        }).dropna(subset=[col])
        parts[col] = df
    if parts:
        from functools import reduce
        lst_dfs[date_str] = reduce(
            lambda a, b: a.merge(b, on=["lat_r", "lon_r"], how="outer"),
            parts.values()
        )
del lst_lookup

# ── Stream VNP09GA monthly TIFs → Parquet ────────────────────────────────────
vnp09_tifs = sorted(SRC.glob("vnp09ga_*.tif"))
if not vnp09_tifs:
    print("[ERROR] No VNP09GA TIFs found. Run download/03 first.")
    sys.exit(1)

print(f"Processing {len(vnp09_tifs)} VNP09GA TIF(s) ...")
writer     = None
total_rows = 0

try:
    for tif in vnp09_tifs:
        print(f"  {tif.stem}")
        (lats, lons), day_data = read_daily_bands(tif, set(VNP09_BANDS))

        for date_str, bands in sorted(day_data.items()):
            if len(bands) < len(VNP09_BANDS):
                continue

            ts = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
            df = pd.DataFrame({
                "time": ts,
                "lat":  lats,
                "lon":  lons,
                "i1":   bands["I1"],
                "i2":   bands["I2"],
                "i3":   bands["I3"],
                "m11":  bands["M11"],
                "ndvi": bands["NDVI"],
                "evi":  bands["EVI"],
            }).dropna(subset=["i1"])

            if df.empty:
                continue

            # Join LST at 1km resolution via rounded lat/lon
            lst = lst_dfs.get(date_str)
            if lst is not None:
                df["lat_r"] = df["lat"].round(4)
                df["lon_r"] = df["lon"].round(4)
                df = df.merge(lst, on=["lat_r", "lon_r"], how="left").drop(
                    columns=["lat_r", "lon_r"]
                )
            else:
                df["lst_day"]   = np.nan
                df["lst_night"] = np.nan

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT, table.schema)
            writer.write_table(table)
            total_rows += len(df)

        print(f"    running total: {total_rows:,} rows", end="\r")

finally:
    if writer:
        writer.close()

print(f"\n[DONE] {OUT}  total_rows={total_rows:,}")
