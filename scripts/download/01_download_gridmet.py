"""
Download all 17 GridMET variables for the AOI via THREDDS OPeNDAP.
Parallelised: all variables download concurrently.
Output: data_raw/gridmet/<var>.nc  (one file per variable)
"""

import sys
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import AOI, START_DATE, END_DATE, DIRS, GRIDMET_VARS

OUT = DIRS["gridmet"]
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://thredds.northwestknowledge.net:443/thredds/ncss/agg_met_{var}_1979_CurrentYear_CONUS.nc"

GRIDMET_NC_VAR_NAMES = {
    "pr": "precipitation_amount",
    "tmmx": "daily_maximum_temperature",
    "tmmn": "daily_minimum_temperature",
    "vs": "daily_mean_wind_speed",
    "th": "daily_mean_wind_direction",
    "srad": "daily_mean_shortwave_radiation_at_surface",
    "rmax": "daily_maximum_relative_humidity",
    "rmin": "daily_minimum_relative_humidity",
    "sph": "daily_mean_specific_humidity",
    "vpd": "daily_mean_vapor_pressure_deficit",
    "erc": "daily_mean_energy_release_component-g",
    "bi": "daily_mean_burning_index_g",
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "pdsi": "daily_mean_palmer_drought_severity_index",
    "pet": "daily_mean_reference_evapotranspiration_grass",
    "etr": "daily_mean_reference_evapotranspiration_alfalfa"
}

def build_url(var: str) -> str:
    nc_var = GRIDMET_NC_VAR_NAMES[var]
    return (
        f"{BASE.format(var=var)}"
        f"?var={nc_var}"
        f"&north={AOI['max_lat']}&south={AOI['min_lat']}"
        f"&west={AOI['min_lon']}&east={AOI['max_lon']}"
        f"&disableProjSubset=on&horizStride=1"
        f"&time_start={START_DATE}T00%3A00%3A00Z"
        f"&time_end={END_DATE}T00%3A00%3A00Z"
        f"&timeStride=1&accept=netcdf"
    )


async def download_var(session: aiohttp.ClientSession, var: str, sem: asyncio.Semaphore):
    out_path = OUT / f"{var}.nc"
    if out_path.exists():
        print(f"[SKIP] {var} already exists")
        return

    url = build_url(var)
    async with sem:
        print(f"[START] {var}")
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                async with aiofiles.open(out_path, "wb") as f:
                    with tqdm(total=total, unit="B", unit_scale=True, desc=var, leave=False) as bar:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            await f.write(chunk)
                            bar.update(len(chunk))
            print(f"[DONE] {var}")
        except Exception as e:
            print(f"[ERROR] {var}: {e}")
            out_path.unlink(missing_ok=True)


async def main():
    # max 4 concurrent to avoid rate-limiting
    sem = asyncio.Semaphore(4)
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_var(session, var, sem) for var in GRIDMET_VARS]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
