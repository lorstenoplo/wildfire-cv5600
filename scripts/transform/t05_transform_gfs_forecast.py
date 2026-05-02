"""
Transform GFS forecast NetCDF files → standardised Parquet.
Input : data_raw/gfs_forecast/gfs_<YYYYMMDD>.nc  (one file per day, 24h forecast)
Output: data_processed/gfs_forecast.parquet
Schema: time (daily), lat, lon,
        forecast_wind_speed, forecast_wind_direction,
        forecast_temp, forecast_precip, forecast_specific_humidity
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC

import numpy as np
import pandas as pd
import xarray as xr

PROC.mkdir(parents=True, exist_ok=True)
OUT  = PROC / "gfs_forecast.parquet"
SRC  = DIRS["gfs"]

if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

# GFS variable names vary slightly between versions; try multiple aliases
VAR_MAP = {
    "forecast_temp":             ["Temperature_height_above_ground",
                                  "TMP_P0_L103_GLL0"],
    "forecast_specific_humidity":["Specific_humidity_height_above_ground",
                                  "SPFH_P0_L103_GLL0"],
    "forecast_precip":           ["Total_precipitation_surface",
                                  "APCP_P8_L1_GLL0_acc"],
    "u_wind":                    ["u-component_of_wind_height_above_ground",
                                  "UGRD_P0_L103_GLL0"],
    "v_wind":                    ["v-component_of_wind_height_above_ground",
                                  "VGRD_P0_L103_GLL0"],
}


def pick_var(ds: xr.Dataset, aliases: list[str]):
    for a in aliases:
        if a in ds:
            return ds[a]
    return None


def first_level(da: xr.DataArray) -> xr.DataArray:
    """Select the lowest atmospheric level if a level dimension exists."""
    for dim in da.dims:
        if "level" in dim or "height" in dim or "lv_" in dim:
            da = da.isel({dim: 0})
    return da


files = sorted(SRC.glob("gfs_????????.nc"))
if not files:
    print("[ERROR] No GFS NetCDF files found. Run download/05 first.")
    sys.exit(1)

print(f"Processing {len(files)} GFS files ...")
dfs = []

for nc in files:
    date_str = nc.stem.replace("gfs_", "")
    try:
        date = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    except ValueError:
        continue

    try:
        ds = xr.open_dataset(nc, engine="netcdf4")
    except Exception:
        try:
            ds = xr.open_dataset(nc, engine="cfgrib")
        except Exception as e:
            print(f"  [WARN] cannot open {nc.name}: {e}")
            continue

    # flatten lat/lon — GFS uses 'lat' or 'latitude'
    lat_dim = "lat" if "lat" in ds.coords else "latitude"
    lon_dim = "lon" if "lon" in ds.coords else "longitude"

    row = {}
    for col, aliases in VAR_MAP.items():
        da = pick_var(ds, aliases)
        if da is None:
            row[col] = np.nan
            continue
        da = first_level(da)
        # take time-mean if multiple forecast steps present
        if "time" in da.dims or "time1" in da.dims:
            t = "time" if "time" in da.dims else "time1"
            da = da.isel({t: 0})
        row[col] = da.values  # 2-D array (lat × lon)

    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    df = pd.DataFrame({
        "time": date,
        "lat":  lat2d.ravel(),
        "lon":  lon2d.ravel(),
    })
    for col in VAR_MAP:
        if isinstance(row.get(col), np.ndarray):
            df[col] = row[col].ravel()
        else:
            df[col] = row.get(col, np.nan)

    # derive wind speed and direction from u/v components
    if isinstance(row.get("u_wind"), np.ndarray) and isinstance(row.get("v_wind"), np.ndarray):
        u = row["u_wind"].ravel()
        v = row["v_wind"].ravel()
        df["forecast_wind_speed"]     = np.sqrt(u**2 + v**2)
        df["forecast_wind_direction"] = (np.degrees(np.arctan2(-u, -v)) % 360)
    else:
        df["forecast_wind_speed"]     = np.nan
        df["forecast_wind_direction"] = np.nan

    df = df.drop(columns=["u_wind", "v_wind"], errors="ignore")
    dfs.append(df)
    ds.close()

out = pd.concat(dfs, ignore_index=True).sort_values(["time", "lat", "lon"]).reset_index(drop=True)
out.to_parquet(OUT, index=False)
print(f"[DONE] {OUT}  shape={out.shape}")
