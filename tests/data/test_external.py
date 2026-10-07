from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from scamsleuth.data import external, load


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pay now <URL> or call <PHONE_NUMBER>", "Pay now xxurl or call xxphone"),
        ("Dear <NAMED_ENTITY>, refund of Rs. <DATE_TIME> due", "Dear, refund of Rs. due"),
        ("visit http://<IP_ADDRESS>/login today", "visit http:// xxurl /login today"),
        ("mail <EMAIL_ADDRESS> [REDACTED] see [URL]", "mail xxemail see xxurl"),
        ("line one  \n\n  line <US_SSN> two ", "line one\nline two"),
    ],
)
def test_clean_imc_text(raw: str, expected: str) -> None:
    assert external.clean_imc_text(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123456 is your Facebook login code. Do not share it.", True),
        ("Your OTP for txn of INR 500 is 4411", True),
        ("Your OTP expires today, verify at xxurl to keep receiving codes", False),
        ("Your parcel is held, pay the fee at xxurl", False),
    ],
)
def test_is_otp_notice(text: str, expected: bool) -> None:
    assert external.is_otp_notice(text) is expected


def _write_imc(tmp_path: Path) -> Path:
    rows = pd.DataFrame(
        {
            "text": [
                "Your parcel is held at the depot, pay the fee: <URL>",
                "Account locked, unusual login detected. Verify now <URL>",
                "Huge mattress sale this weekend, use code <DATE_TIME>",
                "Thanks for your order, receipt: <URL>",  # "others": excluded
                "<DATE_TIME> is your WhatsApp code. Do not share this code",  # OTP notice
                "hi <URL>",  # too short after cleaning
                "Su paquete esta retenido, pague aqui <URL>",
            ],
            "scam_type": [
                "delivery",
                "banking",
                "spam",
                "others",
                "telecom",
                "delivery",
                "delivery",
            ],
            "language": [
                "English",
                "English",
                "English",
                "English",
                "English",
                "English",
                "Spanish",
            ],
            "sender_id": ["phone number"] * 7,
        }
    )
    path = tmp_path / "imc.csv"
    rows.to_csv(path, index=False)
    return path


def test_load_imc25_filters_and_maps_labels(tmp_path: Path) -> None:
    df = external.load_imc25(_write_imc(tmp_path))

    assert list(df.columns) == ["text", "label", "scam_type", "language"]
    assert list(df["label"]) == ["smishing", "smishing", "spam", "smishing"]
    assert list(df["language"]) == ["english", "english", "english", "spanish"]
    assert not df["text"].str.contains("<").any()
    assert "others" not in set(df["scam_type"])


def test_load_hardneg_maps_labels_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "hardneg.csv"
    pd.DataFrame(
        {
            "text": [" Your OTP is 4411 ", "KYC blocked, click http://x.xyz"],
            "label": ["safe", "scam"],
            "scam_type": ["none", "fake_kyc"],
            "language": ["en", "hinglish"],
        }
    ).to_csv(path, index=False)

    df = external.load_hardneg(path)
    assert list(df["label"]) == ["ham", "smishing"]
    assert df["text"].iloc[0] == "Your OTP is 4411"

    path.write_text("text,label,scam_type,language\nhi,maybe,none,en\n")
    with pytest.raises(load.SchemaError, match="unexpected labels"):
        external.load_hardneg(path)


def test_loaders_reject_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("text\nhello\n")
    with pytest.raises(load.SchemaError, match="missing columns"):
        external.load_imc25(path)
    with pytest.raises(load.SchemaError, match="missing columns"):
        external.load_hardneg(path)


def test_downloads_are_verified_and_cached(tmp_path: Path) -> None:
    with mock.patch.object(external, "fetch_verified", return_value=b"a,b\n") as fetch:
        imc = external.download_imc25(tmp_path)
        external.download_imc25(tmp_path)
        hardneg = external.download_hardneg(tmp_path)

    assert imc.read_bytes() == hardneg.read_bytes() == b"a,b\n"
    assert fetch.call_args_list == [
        mock.call(external.IMC25_URL, external.IMC25_SHA256),
        mock.call(external.HARDNEG_URL, external.HARDNEG_SHA256),
    ]
