"""Download the public datasets into ``data/raw/`` (resumable) and record SHA-256 checksums in ``data/MANIFEST.json``.

    python scripts/download_data.py                 # DRIVE, STARE, CHASE_DB1, HRF  (~140 MB)
    python scripts/download_data.py --fives         # + FIVES (1.8 GB rar, optional extra target domain)
    python scripts/download_data.py --only hrf stare

Sources (all free for research use; please cite the original papers, see README):
  HRF       FAU Erlangen          https://www5.cs.fau.de/research/data/fundus-images/            (official, CC BY 4.0)
  STARE     Clemson Univ.         https://cecas.clemson.edu/~ahoover/stare/                      (official)
  CHASE_DB1 Kingston Univ.        https://researchdata.kingston.ac.uk/96/                        (official; if the server is
            unreachable we fall back to the Hugging Face mirror MedOtter/CHASE_DB1)
  DRIVE     Utrecht Univ.         https://drive.grand-challenge.org/   (official download needs a free account, so the
            script uses the PaddleSeg mirror; to use the official files instead unzip them to data/raw/drive/ -- see README)
  FIVES     figshare              https://doi.org/10.6084/m9.figshare.19688169                   (official, CC BY 4.0)

Mirrors are third-party copies: ``prepare_data.py`` validates image counts/sizes and ``make_data_overview.py`` renders a
contact sheet so you can eyeball image/label alignment before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.json"
UA = {"User-Agent": "tglite-data-downloader/1.0 (academic coursework)"}

HF = "https://huggingface.co/datasets/MedOtter/CHASE_DB1/resolve/main/data/"
FIGSHARE_FIVES = "https://ndownloader.figshare.com/files/34969398"

# name -> list of (relative path, [candidate urls in priority order], provenance note)
SOURCES: dict[str, list[tuple[str, list[str], str]]] = {
    "hrf": [("hrf/all.zip", ["https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"], "official")],
    "stare": [
        ("stare/stare-images.tar", ["https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar"], "official"),
        ("stare/labels-ah.tar", ["https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar"], "official"),
        ("stare/labels-vk.tar", ["https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar"], "official"),
    ],
    "chase_db1": [
        ("chase_hf/train.parquet", [HF + "train-00000-of-00001.parquet"], "Hugging Face mirror MedOtter/CHASE_DB1"),
        ("chase_hf/test.parquet", [HF + "test-00000-of-00001.parquet"], "Hugging Face mirror MedOtter/CHASE_DB1"),
    ],
    "drive": [("drive_mirror/drive.zip", ["https://bj.bcebos.com/paddleseg/dataset/drive/drive.zip"], "PaddleSeg mirror of DRIVE")],
    "fives": [("fives/FIVES.rar", [FIGSHARE_FIVES], "official (figshare)")],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_parallel(url: str, dest: Path, connections: int = 8, segment: int = 16 << 20, retries: int = 8) -> None:
    """Segmented download with several concurrent ``Range`` requests (some hosts throttle each connection).

    Progress is kept in ``<dest>.segments.json`` so an interrupted download resumes; an existing sequential partial
    file is treated as a finished prefix.  Falls back to the plain downloader if the server does not support ranges.
    """
    import concurrent.futures as cf
    import threading

    head = requests.get(url, headers=dict(UA, Range="bytes=0-0"), stream=True, timeout=60, allow_redirects=True)
    head.close()
    if head.status_code != 206 or "/" not in head.headers.get("Content-Range", ""):
        return download(url, dest)
    total = int(head.headers["Content-Range"].rsplit("/", 1)[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    side = dest.with_suffix(dest.suffix + ".segments.json")
    nseg = -(-total // segment)
    if side.exists():
        done = set(json.loads(side.read_text())["done"])
    else:
        have = dest.stat().st_size if dest.exists() else 0
        done = set(range(min(nseg, have // segment)))          # sequential prefix from an earlier plain download
    if not dest.exists() or dest.stat().st_size != total:
        with open(dest, "r+b" if dest.exists() else "wb") as fh:  # pre-allocate (sparse) so segments can be written anywhere
            fh.truncate(total)
    lock, t0 = threading.Lock(), time.time()
    base_done = len(done)

    def fetch(i: int) -> int:
        a, b = i * segment, min(total, (i + 1) * segment) - 1
        for attempt in range(retries):
            try:
                with requests.get(url, headers=dict(UA, Range=f"bytes={a}-{b}"), stream=True, timeout=90) as r:
                    if r.status_code != 206:
                        raise IOError(f"status {r.status_code}")
                    buf = b"".join(r.iter_content(1 << 20))
                if len(buf) != b - a + 1:
                    raise IOError("short segment")
                with lock, open(dest, "r+b") as fh:
                    fh.seek(a)
                    fh.write(buf)
                    done.add(i)
                    side.write_text(json.dumps({"total": total, "done": sorted(done)}))
                    n = len(done) - base_done
                    print(f"\r  {dest.name}: {len(done)}/{nseg} segments, {n * segment / max(time.time() - t0, 1e-6) / 1e6:5.2f} MB/s", end="", flush=True)
                return i
            except Exception:  # noqa: BLE001
                time.sleep(min(20, 2 * (attempt + 1)))
        raise RuntimeError(f"segment {i} failed")

    for _round in range(6):                      # failed segments are retried in later rounds instead of aborting everything
        todo = [i for i in range(nseg) if i not in done]
        if not todo:
            break
        with cf.ThreadPoolExecutor(connections) as ex:
            for fut in [ex.submit(fetch, i) for i in todo]:
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"\n  {exc}; will retry")
    print()
    if len(done) != nseg or dest.stat().st_size != total:
        raise RuntimeError("incomplete download")
    side.unlink(missing_ok=True)


def download(url: str, dest: Path, retries: int = 6) -> None:
    """Resumable streaming download with a few retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        headers = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:      # already complete
                    return
                r.raise_for_status()
                if have and r.status_code == 200:   # server ignored Range -> restart
                    have = 0
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    done = have
                    t0 = time.time()
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"\r  {dest.name}: {done/1e6:8.1f}/{total/1e6:.1f} MB  {rate:5.1f} MB/s", end="", flush=True)
                print()
                if total and dest.stat().st_size < total:
                    raise IOError("short read")
                return
        except requests.exceptions.SSLError as exc:
            # Some university servers (e.g. Clemson, for STARE) send an incomplete certificate chain that Python's CA bundle
            # rejects but the operating system trust store accepts: let curl (which uses the OS store) do the transfer.
            import shutil
            import subprocess
            curl = shutil.which("curl")
            if not curl:
                raise RuntimeError(f"TLS verification failed for {url} and curl is not installed: {exc}") from exc
            print("\n  TLS chain not accepted by Python's CA bundle; retrying with curl (OS trust store, still verified)")
            subprocess.run([curl, "-sSL", "--retry", "5", "-C", "-", "-o", str(dest), url], check=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"\n  attempt {attempt + 1}/{retries} failed: {exc}")
            time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="subset of datasets")
    ap.add_argument("--fives", action="store_true", help="also download FIVES (1.8 GB, optional)")
    ap.add_argument("--connections", type=int, default=8, help="parallel range requests for large files (default 8)")
    args = ap.parse_args()

    names = args.only or [n for n in SOURCES if n != "fives" or args.fives]
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for name in names:
        print(f"== {name}")
        for rel, urls, note in SOURCES[name]:
            dest = RAW / rel
            if dest.exists() and dest.stat().st_size > 0 and rel in manifest:
                print(f"  {rel}: already present ({dest.stat().st_size/1e6:.1f} MB)")
                continue
            for url in urls:
                try:
                    if note.startswith("official (figshare)") and args.connections > 1:
                        download_parallel(url, dest, args.connections)
                    else:
                        download(url, dest)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  {url}: {exc}")
            else:
                sys.exit(f"failed to fetch {rel}")
            manifest[rel] = {"sha256": sha256(dest), "bytes": dest.stat().st_size, "url": urls[0], "provenance": note}
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("done. Next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
