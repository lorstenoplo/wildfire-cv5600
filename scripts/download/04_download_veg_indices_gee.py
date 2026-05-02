"""
NDVI and EVI are now computed daily inside 03_download_viirs_spectral_gee.py
directly from VNP09GA I1/I2/I3 bands (NDVI = (I2-I1)/(I2+I1), EVI from I1/I2/I3).

This script is intentionally left as a no-op placeholder.
Delete or ignore it — veg indices will appear as NDVI/EVI columns
in data_processed/viirs_spectral.parquet after running transform/t03.
"""

print("NDVI/EVI are produced in 03_download_viirs_spectral_gee.py — nothing to do here.")
