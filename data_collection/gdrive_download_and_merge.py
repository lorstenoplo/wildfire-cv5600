"""
gdrive_download_and_merge.py
-----------------------------
Downloads monthly checkpoint parquets from Google Drive using a service account
and merges them into a single viirs_spectral.parquet.

Setup:
  pip install pyarrow tqdm google-api-python-client google-auth-httplib2 google-auth

Usage:
  python gdrive_download_and_merge.py --sa-key sa.json --ckpt-dir ./checkpoints --out ./viirs_spectral.parquet
  python gdrive_download_and_merge.py --skip-download --ckpt-dir ./checkpoints --out ./viirs_spectral.parquet
"""

import argparse, gc, sys
from pathlib import Path

def _check_deps():
    missing = []
    for mod in ["googleapiclient", "google.oauth2", "pyarrow", "tqdm"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod.split(".")[0])
    if missing:
        print(f"[ERROR] Missing: {' '.join(set(missing))}")
        print("  pip install pyarrow tqdm google-api-python-client google-auth-httplib2 google-auth")
        sys.exit(1)

_check_deps()

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

SCOPES                = ["https://www.googleapis.com/auth/drive.readonly"]
CHECKPOINTS_FOLDER_ID = "1uOcAPG-viOk914bQPXruYHckcPfSuDkd"

# ── auth ──────────────────────────────────────────────────────────────────────
def get_service(sa_key: str):
    creds = service_account.Credentials.from_service_account_file(sa_key, scopes=SCOPES)
    print(f"[auth] {creds.service_account_email}")
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ── Drive helpers ─────────────────────────────────────────────────────────────
def list_files(svc, folder_id: str, suffix: str = "") -> list[dict]:
    items, page_token = [], None
    while True:
        r = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, size)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        items.extend(r.get("files", []))
        page_token = r.get("nextPageToken")
        if not page_token:
            break
    if suffix:
        items = [f for f in items if f["name"].endswith(suffix)]
    return items


def download_file(svc, file_id: str, dest: Path, size: int | None = None):
    if dest.exists() and size and dest.stat().st_size == size:
        print(f"  [skip] {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = svc.files().get_media(fileId=file_id)
    bar = tqdm(total=size or 0, unit="B", unit_scale=True, desc=dest.name, leave=False)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            status, done = dl.next_chunk()
            if status:
                bar.update(int(status.resumable_progress) - bar.n)
    bar.close()
    print(f"  [ok] {dest.name}  ({dest.stat().st_size/1e9:.2f} GB)")

# ── merge ─────────────────────────────────────────────────────────────────────
def merge(ckpt_dir: Path, out_path: Path):
    files = sorted(ckpt_dir.glob("*.parquet"))
    if not files:
        print(f"[ERROR] No parquet files in {ckpt_dir}")
        sys.exit(1)

    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"\n[MERGE] {len(files)} files  |  {total_gb:.1f} GB")

    out_path.unlink(missing_ok=True)
    writer, rows = None, 0

    for cp in tqdm(files, desc="Merging"):
        pf = pq.ParquetFile(cp)
        for batch in pf.iter_batches(batch_size=2_000_000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path, table.schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table)
            rows += len(table)
            del table, batch
        del pf; gc.collect()

    if writer:
        writer.close()
    print(f"\n[DONE] {out_path}")
    print(f"       {rows:,} rows  |  {out_path.stat().st_size/1e9:.2f} GB")

# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sa-key",        required=False, metavar="KEY.json",
                   help="Service account JSON key")
    p.add_argument("--folder-id",     default=CHECKPOINTS_FOLDER_ID)
    p.add_argument("--ckpt-dir",      default="./checkpoints")
    p.add_argument("--out",           default="./viirs_spectral.parquet")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)

    if not args.skip_download:
        if not args.sa_key:
            print("[ERROR] --sa-key is required for download")
            sys.exit(1)
        svc   = get_service(args.sa_key)
        files = list_files(svc, args.folder_id, suffix=".parquet")
        print(f"[DRIVE] {len(files)} files  |  {sum(int(f.get('size',0)) for f in files)/1e9:.1f} GB")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(sorted(files, key=lambda x: x["name"]), desc="Downloading", unit="file"):
            download_file(svc, f["id"], ckpt_dir / f["name"], int(f.get("size") or 0) or None)

    merge(ckpt_dir, out_path)


if __name__ == "__main__":
    main()
