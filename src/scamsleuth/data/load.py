"""Download and validate the Mendeley SMS Phishing dataset (DOI 10.17632/f45bkkt8pr.1).

The source is Mishra & Soni (2022), licensed CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd

LABELS: Final = ("ham", "spam", "smishing")

SOURCE_URL: Final = (
    "https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
    "edb361de-918d-469f-9106-e84823830665/file_downloaded"
)
SOURCE_SHA256: Final = "9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3"
SOURCE_MEMBER: Final = "Dataset_5971.csv"

_RAW_COLUMNS: Final = ("LABEL", "TEXT", "URL", "EMAIL", "PHONE")
_FLAG_VALUES: Final = {"yes": True, "no": False}


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected schema."""


def fetch_verified(url: str, sha256: str) -> bytes:
    """Download ``url`` and check its SHA-256, so every build uses identical source data."""
    # Some hosts reject urllib's default User-Agent with HTTP 403.
    request = urllib.request.Request(url, headers={"User-Agent": "scamsleuth/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload: bytes = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise SchemaError(f"checksum mismatch for {url}: expected {sha256}, got {digest}")
    return payload


def download_raw(dest_dir: Path, *, url: str = SOURCE_URL) -> Path:
    """Download the dataset archive, verify its checksum and extract the CSV.

    Returns the path of the extracted CSV. An existing file is reused.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / SOURCE_MEMBER
    if csv_path.exists():
        return csv_path

    payload = fetch_verified(url, SOURCE_SHA256)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_path.write_bytes(archive.read(SOURCE_MEMBER))
    return csv_path


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV into a clean, validated frame.

    Output columns: ``text`` (str), ``label`` (one of ``LABELS``) and the boolean
    source annotations ``has_url``, ``has_email``, ``has_phone``.
    """
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    missing = set(_RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")

    df = pd.DataFrame(
        {
            "text": raw["TEXT"].str.strip(),
            "label": raw["LABEL"].str.strip().str.lower(),
        }
    )
    for src, dst in (("URL", "has_url"), ("EMAIL", "has_email"), ("PHONE", "has_phone")):
        flags = raw[src].str.strip().str.lower()
        unknown = set(flags) - set(_FLAG_VALUES)
        if unknown:
            raise SchemaError(f"unexpected values in {src}: {sorted(unknown)}")
        df[dst] = flags.map(_FLAG_VALUES).astype(bool)

    unknown_labels = set(df["label"]) - set(LABELS)
    if unknown_labels:
        raise SchemaError(f"unexpected labels: {sorted(unknown_labels)}")
    if (df["text"] == "").any():
        raise SchemaError("empty message text")

    return df
