"""
Download daily VIIRS surface reflectance + LST from GEE.

Products:
  VNP09GA  — daily 500m surface reflectance: I1, I2, I3, M11 + NDVI/EVI
              exported per-month (365 days × 6 bands = 2190 > GEE 1024-band limit)
  VNP21A1D — daily 1km LST (day),   exported per-year (365 × 1 = fine)
  VNP21A1N — daily 1km LST (night), exported per-year

Band names encoded as YYYYMMDD_<band> so the transform step can reconstruct time.

Output folder: data_raw/viirs_spectral/
  vnp09ga_<YYYY>_<MM>.tif
  vnp21a1d_<YYYY>.tif
  vnp21a1n_<YYYY>.tif
"""

import sys, time, calendar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import AOI, START_DATE, END_DATE, DIRS, GEE_PROJECT

import ee
import ee.batch

OUT = DIRS["viirs_spectral"]
OUT.mkdir(parents=True, exist_ok=True)

ee.Initialize(project=GEE_PROJECT)

_all_tasks    = ee.batch.Task.list()
# Skip resubmitting tasks that are in-progress OR completed successfully.
# Failed/errored tasks (state=COMPLETED but with error) are NOT skipped so they resubmit.
_active_descs = {
    t.config["description"]
    for t in _all_tasks
    if t.state in ("READY", "RUNNING")
    or (t.state == "COMPLETED" and not t.status().get("error_message"))
}

region = ee.Geometry.Rectangle([
    AOI["min_lon"], AOI["min_lat"],
    AOI["max_lon"], AOI["max_lat"],
])
years = list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))


def _prefix_date_fn(bands_after):
    def prefix_date(img):
        date_str = img.date().format("YYYYMMdd")
        new_names = ee.List([
            ee.String(date_str).cat("_").cat(ee.String(b))
            for b in bands_after
        ])
        return img.rename(new_names)
    return prefix_date


def submit_period(collection_id, select_bands, label, scale,
                  start, end, compute_fn=None, output_bands=None):
    """Submit one export task for [start, end). Returns task or None if skipped."""
    desc = f"{label}_{start.replace('-', '')[:6]}" if len(start) > 7 else f"{label}_{start[:4]}"
    # Use full start string as part of desc to be unique per period
    desc = f"{label}_{start[:7].replace('-', '_')}" if "-" in start[5:] else f"{label}_{start[:4]}"

    local_file = OUT / f"{desc}.tif"
    if local_file.exists() or desc in _active_descs:
        print(f"[SKIP] {desc}")
        return None

    col = (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .filterBounds(region)
        .select(select_bands)
    )
    if compute_fn:
        col = col.map(compute_fn)

    bands_after = output_bands if output_bands is not None else select_bands
    col = col.map(_prefix_date_fn(bands_after))

    img = col.toBands().clip(region)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder="wildfire_viirs",
        fileNamePrefix=desc,
        region=region,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"[SUBMITTED] {desc}  ({start} → {end})")
    return task


def add_ndvi_evi(img):
    ndvi = img.normalizedDifference(["I2", "I1"]).rename("NDVI").toFloat()
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {"NIR": img.select("I2"), "RED": img.select("I1"), "BLUE": img.select("I3")},
    ).rename("EVI").toFloat()
    return img.select(["I1", "I2", "I3", "M11"]).toFloat().addBands([ndvi, evi])


tasks = []
for year in years:
    # VNP09GA: submit monthly (31 days × 6 bands = 186 — well under 1024 limit)
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        # end is exclusive in GEE filterDate
        next_month = month + 1 if month < 12 else 1
        next_year  = year if month < 12 else year + 1
        end = f"{next_year}-{next_month:02d}-01"

        t = submit_period(
            "NASA/VIIRS/002/VNP09GA",
            ["I1", "I2", "I3", "M11"],
            "vnp09ga", 500,
            start, end,
            compute_fn=add_ndvi_evi,
            output_bands=["I1", "I2", "I3", "M11", "NDVI", "EVI"],
        )
        if t:
            tasks.append(t)

    # LST day — yearly is fine (365 × 1 band)
    t = submit_period("NASA/VIIRS/002/VNP21A1D", ["LST_1KM"],
                      "vnp21a1d", 1000, f"{year}-01-01", f"{year+1}-01-01")
    if t: tasks.append(t)

    # LST night
    t = submit_period("NASA/VIIRS/002/VNP21A1N", ["LST_1KM"],
                      "vnp21a1n", 1000, f"{year}-01-01", f"{year+1}-01-01")
    if t: tasks.append(t)


def download_from_drive(folder_name="wildfire_viirs"):
    """Download all completed TIFs from the GEE export Drive folder."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    creds = ee.data._credentials
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    res = drive.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id,name)",
    ).execute()
    folders = res.get("files", [])
    if not folders:
        print(f"[WARN] Drive folder '{folder_name}' not found.")
        return
    folder_id = folders[0]["id"]

    page_token = None
    files = []
    while True:
        res = drive.files().list(
            q=f"'{folder_id}' in parents and name contains '.tif' and trashed=false",
            fields="files(id,name,size),nextPageToken",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(files)} TIF(s) in Drive folder '{folder_name}'")
    for f in files:
        dest = OUT / f["name"]
        if dest.exists():
            print(f"  [SKIP] {f['name']} already local")
            continue
        size_mb = int(f.get("size", 0)) / 1e6
        print(f"  Downloading {f['name']} ({size_mb:.0f} MB) ...")
        request = drive.files().get_media(fileId=f["id"])
        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=32 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"    {int(status.progress() * 100)}%", end="\r")
        print(f"  [DONE] {f['name']}")


if tasks:
    print(f"\nSubmitted {len(tasks)} new task(s). Waiting...")
    while True:
        running = [t for t in ee.batch.Task.list() if t.state in ("READY", "RUNNING")]
        if not running:
            break
        print(f"  {len(running)} task(s) still running ...")
        time.sleep(60)
    print("All GEE tasks done.")
else:
    print("\nNo new tasks to submit.")

print("\nDownloading completed TIFs from Google Drive...")
download_from_drive("wildfire_viirs")
print("[ALL DONE]", OUT)
