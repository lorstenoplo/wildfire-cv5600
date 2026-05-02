"""
Download SRTM 30m DEM for the AOI using the `elevation` package (CGIAR SRTM).
Output: data_raw/srtm/srtm_dem.tif
Slope and aspect are derived in the transform step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import AOI, DIRS

import elevation  # pip install elevation (wraps CGIAR SRTM tiles)
import subprocess

OUT = DIRS["srtm"]
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "srtm_dem.tif"

if OUT_FILE.exists():
    print(f"[SKIP] {OUT_FILE} already exists")
else:
    bounds = (AOI["min_lon"], AOI["min_lat"], AOI["max_lon"], AOI["max_lat"])
    print(f"Downloading SRTM for bounds {bounds} ...")
    elevation.clip(bounds=bounds, output=str(OUT_FILE), product="SRTM3")
    elevation.clean()
    print(f"[DONE] {OUT_FILE}")
