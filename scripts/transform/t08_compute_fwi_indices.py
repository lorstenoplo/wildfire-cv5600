"""
Compute Canadian FWI System indices from GridMET weather using cffdrs/cffwis.py.

Indices computed (daily, per grid cell):
  FFMC → Fine Fuel Moisture Code
  DMC  → Duff Moisture Code
  DC   → Drought Code
  ISI  → Initial Spread Index
  BUI  → Build Up Index
  FWI  → Fire Weather Index

Each index is stateful (today depends on yesterday), so we iterate
day-by-day while keeping per-cell state arrays in memory.

GridMET unit conversions applied here:
  tmmx / tmmn : Kelvin → Celsius  (subtract 273.15)
  vs          : m/s    → km/h     (multiply by 3.6)
  pr          : mm     (no change)
  rmin / rmax : %      (no change)

FWI noon-obs convention:
  temp  ← tmmx  (daily max ≈ noon peak)
  rh    ← rmin  (daily min RH ≈ noon)
  wind  ← vs    (daily mean, best available)
  precip← pr    (24-h total)

Output: data_processed/fwi_indices.parquet
Schema: time, lat, lon, ffmc, dmc, dc, isi, bui, fwi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROC

import numpy as np
import pandas as pd
from cffdrs.cffwis import dailyFFMC, dailyDMC, dailyDC, dailyISI, dailyBUI, dailyFWI

PROC.mkdir(parents=True, exist_ok=True)
OUT  = PROC / "fwi_indices.parquet"
SRC  = PROC / "gridmet.parquet"

if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

if not SRC.exists():
    print("[ERROR] gridmet.parquet not found. Run t01 first.")
    sys.exit(1)

# ── Load GridMET and pivot to (time × cell) layout ───────────────────────────
print("Loading GridMET parquet ...")
gm = pd.read_parquet(SRC, columns=["time", "lat", "lon",
                                    "max_temp", "min_temp",
                                    "rh_min", "rh_max",
                                    "wind_speed", "precipitation"])
gm["time"] = pd.to_datetime(gm["time"])

# unit conversions
gm["temp_c"]    = gm["max_temp"]   - 273.15          # K → °C  (noon ≈ max)
gm["rh_pct"]    = gm["rmin"] if "rmin" in gm.columns else gm["rh_min"]
gm["wind_kmh"]  = gm["wind_speed"] * 3.6             # m/s → km/h
gm["precip_mm"] = gm["precipitation"]                # already mm

# clamp physically impossible values
gm["temp_c"]   = gm["temp_c"].clip(-50, 60)
gm["rh_pct"]   = gm["rh_pct"].clip(0, 100)
gm["wind_kmh"] = gm["wind_kmh"].clip(0, None)
gm["precip_mm"]= gm["precip_mm"].clip(0, None)

# build an ordered cell index so we can work with flat numpy arrays
cells = gm[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
cells["cell_id"] = cells.index
gm = gm.merge(cells, on=["lat", "lon"])
n_cells = len(cells)

dates = sorted(gm["time"].unique())
print(f"  {n_cells} grid cells × {len(dates)} days")

# ── Initialise state arrays ───────────────────────────────────────────────────
# Standard FWI startup values (Van Wagner 1987)
FFMC0 = np.full(n_cells, 85.0)
DMC0  = np.full(n_cells,  6.0)
DC0   = np.full(n_cells, 15.0)

rows = []

print("Computing FWI indices day-by-day ...")
for i, date in enumerate(dates):
    day_df = gm[gm["time"] == date].set_index("cell_id")

    # align to full cell array (NaN for missing cells on this day)
    temp   = day_df.reindex(range(n_cells))["temp_c"].values
    rh     = day_df.reindex(range(n_cells))["rh_pct"].values
    wind   = day_df.reindex(range(n_cells))["wind_kmh"].values
    precip = day_df.reindex(range(n_cells))["precip_mm"].values
    month  = int(pd.Timestamp(date).month)
    lat_arr= cells["lat"].values

    # compute today's indices (vectorised over all cells)
    ffmc = dailyFFMC(ffmc0=FFMC0, temp=temp, rh=rh,   wind=wind, precip=precip)
    dmc  = dailyDMC( dmc0=DMC0,   temp=temp, rh=rh,   precip=precip,
                     month=month, lat=lat_arr, lat_adjust=True)
    dc   = dailyDC(  dc0=DC0,     temp=temp, precip=precip,
                     month=month, lat=lat_arr, lat_adjust=True)
    isi  = dailyISI( wind=wind,   ffmc=ffmc)
    bui  = dailyBUI( dmc=dmc,     dc=dc)
    fwi  = dailyFWI( isi=isi,     bui=bui)

    # store results
    out_df = cells.copy()
    out_df["time"] = date
    out_df["ffmc"] = ffmc
    out_df["dmc"]  = dmc
    out_df["dc"]   = dc
    out_df["isi"]  = isi
    out_df["bui"]  = bui
    out_df["fwi"]  = fwi
    rows.append(out_df[["time", "lat", "lon", "ffmc", "dmc", "dc", "isi", "bui", "fwi"]])

    # carry state forward (use nan-safe fallback to previous value)
    FFMC0 = np.where(np.isnan(ffmc), FFMC0, ffmc)
    DMC0  = np.where(np.isnan(dmc),  DMC0,  dmc)
    DC0   = np.where(np.isnan(dc),   DC0,   dc)

    if (i + 1) % 365 == 0:
        print(f"  {i+1}/{len(dates)} days done")

out = pd.concat(rows, ignore_index=True).sort_values(["time", "lat", "lon"]).reset_index(drop=True)
out.to_parquet(OUT, index=False)
print(f"[DONE] {OUT}  shape={out.shape}")
