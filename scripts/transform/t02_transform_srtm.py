"""
Derive slope and aspect from the SRTM DEM.
Output: data_processed/terrain.parquet  — columns: lat, lon, elevation, slope, aspect
(static — no time dimension)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC

import numpy as np
import rasterio
from rasterio.transform import rowcol
import pandas as pd

PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "terrain.parquet"
DEM = DIRS["srtm"] / "srtm_dem.tif"

if OUT.exists():
    print(f"[SKIP] {OUT}")
    sys.exit(0)

if not DEM.exists():
    print(f"[ERROR] DEM not found at {DEM}. Run download/02 first.")
    sys.exit(1)

print("Computing slope and aspect from SRTM DEM ...")

with rasterio.open(DEM) as src:
    elev = src.read(1).astype(float)
    transform = src.transform
    nodata = src.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan

    # pixel size in metres (approximate for geographic CRS)
    res_x_deg = abs(transform.a)
    res_y_deg = abs(transform.e)
    lat_centre = (src.bounds.top + src.bounds.bottom) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_centre))
    dx = res_x_deg * m_per_deg_lon
    dy = res_y_deg * m_per_deg_lat

    # gradient → slope / aspect
    gy, gx = np.gradient(elev, dy, dx)
    slope  = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360

    rows, cols = np.indices(elev.shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())

df = pd.DataFrame({
    "lon":       xs,
    "lat":       ys,
    "elevation": elev.ravel(),
    "slope":     slope.ravel(),
    "aspect":    aspect.ravel(),
}).dropna()

df.to_parquet(OUT, index=False)
print(f"[DONE] {OUT}  shape={df.shape}")
