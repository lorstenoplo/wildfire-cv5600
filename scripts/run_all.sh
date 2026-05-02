#!/usr/bin/env bash
# Run all downloads in parallel, then all transforms sequentially.
# Usage: bash scripts/run_all.sh
# GEE scripts (03, 04) require `ee.Authenticate()` done once first.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DL="$ROOT/scripts/download"
TR="$ROOT/scripts/transform"

# echo "=== STEP 0: Install dependencies ==="
# bash "$DL/00_install_deps.sh"

echo "=== STEP 1: Parallel downloads ==="
python3 "$DL/01_download_gridmet.py"          &  PID_GRIDMET=$!
python3 "$DL/02_download_srtm.py"             &  PID_SRTM=$!
python3 "$DL/03_download_viirs_spectral_gee.py" & PID_VIIRS=$!
python3 "$DL/05_download_gfs_forecast.py"     &  PID_GFS=$!
python3 "$DL/07_download_historical_ignition.py" & PID_HIST=$!

wait $PID_GRIDMET && echo "[DONE] GridMET"      || echo "[FAIL] GridMET"
wait $PID_SRTM    && echo "[DONE] SRTM"         || echo "[FAIL] SRTM"
wait $PID_VIIRS   && echo "[DONE] VIIRS spectral" || echo "[FAIL] VIIRS spectral"
wait $PID_GFS     && echo "[DONE] GFS forecast"  || echo "[FAIL] GFS forecast"
wait $PID_HIST    && echo "[DONE] Historical ignition" || echo "[FAIL] Historical ignition"

echo "=== STEP 2: Transforms (sequential) ==="
python3 "$TR/t01_transform_gridmet.py"
python3 "$TR/t02_transform_srtm.py"
python3 "$TR/t03_transform_viirs_spectral.py"
python3 "$TR/t04_transform_veg_indices.py"
python3 "$TR/t05_transform_gfs_forecast.py"
python3 "$TR/t06_transform_land_cover.py"
python3 "$TR/t07_transform_historical_ignition.py"
python3 "$TR/t08_compute_fwi_indices.py"

echo "=== ALL DONE ==="
echo "Processed files in: $ROOT/data_processed/"
