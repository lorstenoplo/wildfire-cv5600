#!/usr/bin/env bash
# Run once before any download script
pip install \
  requests tqdm numpy pandas xarray netCDF4 \
  rasterio rioxarray geopandas shapely pyproj \
  earthengine-api geemap elevation \
  aiohttp aiofiles dask
