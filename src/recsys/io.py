"""Download only interactions; never extract arbitrary archive paths."""

import hashlib
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


REPOSITORY = "https://github.com/irsafilo/KION_DATASET"
KION_REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
KION_EXPECTED_ROWS = 5_476_251


def _json_scalar(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_scalar) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_interactions(archive, destination):
    """Accept both standalone interactions.zip and original dataset bundles."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            name for name in bundle.namelist()
            if Path(name).name.lower() == "interactions.csv" and "__MACOSX" not in name
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one interactions.csv; found {candidates}")
        temporary = destination.with_suffix(".csv.partial")
        try:
            with bundle.open(candidates[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def download_kion(directory, revision=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Use the complete bundle referenced by the RecTools KION tutorial.
    # The standalone interactions.csv.zip observed in Colab contained a
    # truncated CSV (1,594,787 rows, final date '20') despite valid ZIP framing.
    if revision is None:
        revision = KION_REVISION
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full lowercase Git commit SHA (40 hex characters)")
    url = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{revision}/data_en.zip"
    archive = directory / "data_en.zip"
    partial = archive.with_suffix(".zip.partial")
    # Do not silently trust an archive left by another revision or interrupted run.
    try:
        with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        if not zipfile.is_zipfile(partial):
            raise ValueError("Downloaded response is not a ZIP archive")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    csv_path = extract_interactions(archive, directory / "interactions.csv")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        rows = sum(1 for _ in reader)
    if revision == KION_REVISION and rows != KION_EXPECTED_ROWS:
        raise ValueError(f"Incomplete pinned KION CSV: expected {KION_EXPECTED_ROWS} rows, got {rows}")
    write_json(directory / "source.json", {
        "repository": REPOSITORY, "revision": revision, "url": url,
        "archive_sha256": sha256(archive), "csv_sha256": sha256(csv_path),
        "rows": rows,
    })
    return csv_path
