import hashlib
import io
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from scamsleuth.data import load

RAW_CSV = (
    "LABEL,TEXT,URL,EMAIL,PHONE\n"
    "ham,  See you at 6  ,No,No,No\n"
    "Smishing,\tYour parcel is on hold: http://x.top,yes,No,No\n"
    "Spam,50% off pizza call 08001234567,No,No,yes\n"
)


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "raw.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_raw_normalises_labels_text_and_flags(tmp_path: Path) -> None:
    df = load.load_raw(write(tmp_path, RAW_CSV))

    assert list(df["label"]) == ["ham", "smishing", "spam"]
    assert list(df["text"]) == [
        "See you at 6",
        "Your parcel is on hold: http://x.top",
        "50% off pizza call 08001234567",
    ]
    assert list(df["has_url"]) == [False, True, False]
    assert list(df["has_phone"]) == [False, False, True]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("LABEL,TEXT\nham,hi\n", "missing columns"),
        ("LABEL,TEXT,URL,EMAIL,PHONE\nphishy,hi,No,No,No\n", "unexpected labels"),
        ("LABEL,TEXT,URL,EMAIL,PHONE\nham,hi,maybe,No,No\n", "unexpected values in URL"),
        ("LABEL,TEXT,URL,EMAIL,PHONE\nham,  ,No,No,No\n", "empty message text"),
    ],
)
def test_load_raw_rejects_bad_schema(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(load.SchemaError, match=message):
        load.load_raw(write(tmp_path, content))


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(load.SOURCE_MEMBER, RAW_CSV)
    return buffer.getvalue()


def _fake_urlopen(payload: bytes) -> mock.MagicMock:
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = payload
    return response


def test_download_raw_verifies_checksum_and_extracts(tmp_path: Path) -> None:
    payload = _zip_bytes()
    with (
        mock.patch.object(load, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest()),
        mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(payload)) as urlopen,
    ):
        path = load.download_raw(tmp_path)
        load.download_raw(tmp_path)  # second call reuses the extracted file

    assert path.read_text(encoding="utf-8") == RAW_CSV
    assert urlopen.call_count == 1


def test_download_raw_rejects_tampered_archive(tmp_path: Path) -> None:
    with (
        mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(_zip_bytes())),
        pytest.raises(load.SchemaError, match="checksum mismatch"),
    ):
        load.download_raw(tmp_path)
    assert not (tmp_path / load.SOURCE_MEMBER).exists()
