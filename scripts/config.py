from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Your Google Cloud project ID with Earth Engine API enabled
# Set this once here — all GEE scripts read it from config
GEE_PROJECT = "gen-lang-client-0293562798"

AOI = {
    "min_lon": -124.135,
    "max_lon": -118.963,
    "min_lat": 36.993,
    "max_lat": 42.01,
}
# ee/opendap bounding box shorthand [W, S, E, N]
BBOX = [AOI["min_lon"], AOI["min_lat"], AOI["max_lon"], AOI["max_lat"]]

START_DATE = "2012-01-01"
END_DATE   = "2023-12-31"

RAW   = ROOT / "data_raw"
PROC  = ROOT / "data_processed"

DIRS = {
    "gridmet":           RAW / "gridmet",
    "gfs":               RAW / "gfs_forecast",
    "viirs_spectral":    RAW / "viirs_spectral",
    "srtm":              RAW / "srtm",
    "land_cover":        RAW / "land_cover",
    "veg_indices":       RAW / "vegetation_indices",
    "hist_ignition":     RAW / "historical_ignition",
}

# GridMET variable codes  → human label
GRIDMET_VARS = {
    "pr":    "precipitation",
    "tmmx":  "max_temp",
    "tmmn":  "min_temp",
    "vs":    "wind_speed",
    "th":    "wind_direction",
    "srad":  "solar_radiation",
    "rmax":  "rh_max",
    "rmin":  "rh_min",
    "sph":   "specific_humidity",
    "vpd":   "vapor_pressure_deficit",
    "erc":   "energy_release_component",
    "bi":    "burning_index",
    "fm100": "fuel_moisture_100hr",
    "fm1000":"fuel_moisture_1000hr",
    "pdsi":  "palmer_drought_severity_index",
    "pet":   "potential_evapotranspiration",
    "etr":   "actual_evapotranspiration",
}

# VIIRS spectral band names used in GEE (VNP09GA product)
VIIRS_BANDS = ["I1", "I2", "I3", "I4", "I5", "M11"]
