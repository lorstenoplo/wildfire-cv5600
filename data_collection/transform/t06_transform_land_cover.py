"""
Transform MRLC land cover GeoTIFF → standardised Parquet.
Also maps NLCD class → Anderson 13 fuel type.
Input : data_raw/land_cover/*.tif / *.tiff  (MRLC NLCD, single file)
Output: data_processed/land_cover.parquet
Schema: lat, lon, land_cover_class, land_cover_label, fuel_type  (static)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DIRS, PROC, AOI

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import Transformer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROC.mkdir(parents=True, exist_ok=True)
OUT = PROC / "land_cover.parquet"

if OUT.exists():
    print(f"[SKIP] {OUT} already exists")
    sys.exit(0)

NLCD_LABELS = {
    11: "Open Water",        12: "Perennial Ice/Snow",
    21: "Developed Open Space", 22: "Developed Low Intensity",
    23: "Developed Medium Intensity", 24: "Developed High Intensity",
    31: "Barren Land",       41: "Deciduous Forest",
    42: "Evergreen Forest",  43: "Mixed Forest",
    52: "Shrub/Scrub",       71: "Grassland/Herbaceous",
    81: "Pasture/Hay",       82: "Cultivated Crops",
    90: "Woody Wetlands",    95: "Emergent Herbaceous Wetlands",
}
NLCD_TO_FUEL = {
    11: 98, 12: 99, 21: 99, 22: 99, 23: 99, 24: 99, 31: 99,
    41: 8,  42: 9,  43: 10, 52: 6,  71: 3,  81: 3,
    82: 4,  90: 11, 95: 3,
}

tifs = list(DIRS["land_cover"].glob("*.tif")) + list(DIRS["land_cover"].glob("*.tiff"))
if not tifs:
    print("[ERROR] No land cover raster found.")
    sys.exit(1)

tif = tifs[0]
print(f"Processing {tif.name} ...")

CHUNK_ROWS = 512  # read this many raster rows at a time

with rasterio.open(tif) as src:
    # Transform AOI lat/lon bbox → raster CRS so we can read only that window
    to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    left,  bottom = to_src.transform(AOI["min_lon"], AOI["min_lat"])
    right, top    = to_src.transform(AOI["max_lon"], AOI["max_lat"])

    win = window_from_bounds(left, bottom, right, top, src.transform)
    win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

    col_off = int(win.col_off)
    row_off = int(win.row_off)
    win_w   = int(np.ceil(win.width))
    win_h   = int(np.ceil(win.height))

    # Transformer from raster CRS → WGS84
    to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    nodata   = src.nodata
    transform = src.transform

    print(f"AOI window: {win_h} rows × {win_w} cols  (full raster: {src.height}×{src.width})")

    writer      = None
    total_rows  = 0

    try:
        for r_start in range(0, win_h, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, win_h)
            chunk_win = rasterio.windows.Window(col_off, row_off + r_start, win_w, r_end - r_start)
            data = src.read(1, window=chunk_win)  # uint8, shape (rows, cols)

            rows_h, cols_w = data.shape
            # pixel-centre coordinates in the raster CRS
            col_idx = np.arange(cols_w) + col_off
            row_idx = np.arange(rows_h) + row_off + r_start
            col_grid, row_grid = np.meshgrid(col_idx, row_idx)

            xs = transform.c + (col_grid + 0.5) * transform.a
            ys = transform.f + (row_grid + 0.5) * transform.e
            lons, lats = to_wgs84.transform(xs.ravel(), ys.ravel())

            lc = data.ravel()
            valid = lc != int(nodata) if nodata is not None else np.ones(len(lc), dtype=bool)
            # also clip to exact AOI lat/lon
            valid &= (
                (lats >= AOI["min_lat"]) & (lats <= AOI["max_lat"]) &
                (lons >= AOI["min_lon"]) & (lons <= AOI["max_lon"])
            )

            if not valid.any():
                continue

            lc_v  = lc[valid].astype(np.uint8)
            lat_v = lats[valid].astype(np.float32)
            lon_v = lons[valid].astype(np.float32)

            df = pd.DataFrame({
                "lat":               lat_v,
                "lon":               lon_v,
                "land_cover_class":  lc_v,
            })
            df["land_cover_label"] = df["land_cover_class"].map(NLCD_LABELS).fillna("Unknown")
            df["fuel_type"]        = df["land_cover_class"].map(NLCD_TO_FUEL).fillna(99).astype(np.uint8)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT, table.schema)
            writer.write_table(table)

            total_rows += len(df)
            print(f"  rows {row_off + r_start}–{row_off + r_end} → {total_rows:,} kept", end="\r")

    finally:
        if writer:
            writer.close()

print(f"\n[DONE] {OUT}  total_rows={total_rows:,}")
