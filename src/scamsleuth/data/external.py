"""Additional real-world and stress-test datasets.

* **IMC 2025 smishing reports** (Agarwal et al., *Fishing for Smishing*, IMC 2025,
  CC BY 4.0): ~34k real smishing messages reported by users between 2017 and 2024,
  labelled by scam type. Used for training and for a modern, real-world test slice.
* **Indian Scam SMS, synthetic and audited** (Hugging Face ``Ridham115``, CC BY 4.0):
  LLM-written scams and *genuine look-alikes* (bank OTPs, courier and bill notices) in
  English, Hinglish and Roman-script code-mixed styles. Used only as a stress test for
  false alarms, never for training or headline accuracy.

Both files are pinned to an exact revision and verified by checksum.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pandas as pd

from scamsleuth.data.load import SchemaError, fetch_verified

IMC25_URL: Final = (
    "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/"
    "a6175560b57387199871e51fbef6bc523d2516b4/dataset/final_dataset_output.csv"
)
IMC25_SHA256: Final = "1bbd1e9e82c3ea023112207b80da268a5c4a07d2353c2b0898360ab037fa9a64"

HARDNEG_URL: Final = (
    "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited/resolve/"
    "45e92c38ba20e973ef3ff5d1893a0a710f622018/data/synthetic_audited.csv"
)
HARDNEG_SHA256: Final = "45bde5f156a1a42698bab209e32bcd41b3639013d772f644cc3208ed39fb61e2"

# IMC scam types kept, and the label each maps to. "others" is excluded: it mixes real
# scams with genuine notifications (receipts, appointment reminders, reset codes) that
# users reported out of context, so its labels are unreliable.
IMC25_LABELS: Final = {
    "banking": "smishing",
    "delivery": "smishing",
    "government": "smishing",
    "telecom": "smishing",
    "wrong number": "smishing",
    "hey mum/dad": "smishing",
    "spam": "spam",
}

# The source anonymised entities with Presidio-style tags. Links, phones and emails map to
# the same placeholder tokens that ``mask_entities`` produces for every other source.
# All other tags (names, dates, amounts, ID numbers) are removed: they occur only in this
# source, so keeping them would let a model detect the *source* instead of the scam.
_TAG_TOKENS: Final = {
    "URL": " xxurl ",
    "IP_ADDRESS": " xxurl ",
    "PHONE_NUMBER": " xxphone ",
    "EMAIL_ADDRESS": " xxemail ",
}
_TAG = re.compile(r"<([A-Z_]+)>|\[(?:REDACTED|hidden|URL)\]", re.IGNORECASE)
_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?)])")

# Genuine one-time-code notifications, reported because the user did not expect them.
# With no link or phone number to act on, the message itself is not the attack.
_OTP_NOTICE = re.compile(
    r"\b(?:otp|verification code|login code|security code|one[- ]time (?:pass|code|pin)"
    r"|is your [\w ]{0,30}code|do not share (?:this|the) code)\b",
    re.IGNORECASE,
)
_ACTIONABLE = re.compile(r"xxurl|xxphone|xxemail|https?://", re.IGNORECASE)

MIN_CHARS: Final = 20


def _replace_tag(match: re.Match[str]) -> str:
    tag = match.group(1)
    if tag is None:
        return " xxurl " if match.group().lower() == "[url]" else " "
    return _TAG_TOKENS.get(tag.upper(), " ")


def clean_imc_text(text: str) -> str:
    """Map anonymisation tags to placeholder tokens, drop the rest, tidy whitespace."""
    text = _TAG.sub(_replace_tag, text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACE.sub(" ", text))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def is_otp_notice(text: str) -> bool:
    return bool(_OTP_NOTICE.search(text)) and not _ACTIONABLE.search(text)


def download_imc25(dest_dir: Path) -> Path:
    path = dest_dir / "imc25_smishing.csv"
    if not path.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_verified(IMC25_URL, IMC25_SHA256))
    return path


def download_hardneg(dest_dir: Path) -> Path:
    path = dest_dir / "hardneg_synthetic.csv"
    if not path.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_verified(HARDNEG_URL, HARDNEG_SHA256))
    return path


def load_imc25(csv_path: Path) -> pd.DataFrame:
    """Clean IMC 2025 reports into ``text, label, scam_type, language`` (all languages).

    Drops the ``others`` category, OTP notices without an actionable link or number, and
    messages shorter than ``MIN_CHARS`` after cleaning (e.g. a bare link).
    """
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = {"text", "scam_type", "language"} - set(raw.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")

    df = raw.loc[raw["scam_type"].isin(IMC25_LABELS), ["text", "scam_type", "language"]].copy()
    df["text"] = df["text"].map(clean_imc_text)
    df = df.loc[(df["text"].str.len() >= MIN_CHARS) & ~df["text"].map(is_otp_notice)]
    df["label"] = df["scam_type"].map(IMC25_LABELS)
    df["language"] = df["language"].str.strip().str.lower()
    return df[["text", "label", "scam_type", "language"]].reset_index(drop=True)


def load_hardneg(csv_path: Path) -> pd.DataFrame:
    """Load the synthetic stress-test set as ``text, label, scam_type, language``."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = {"text", "label", "scam_type", "language"} - set(raw.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")
    unknown = set(raw["label"]) - {"safe", "scam"}
    if unknown:
        raise SchemaError(f"unexpected labels: {sorted(unknown)}")

    return pd.DataFrame(
        {
            "text": raw["text"].str.strip(),
            "label": raw["label"].map({"safe": "ham", "scam": "smishing"}),
            "scam_type": raw["scam_type"],
            "language": raw["language"],
        }
    )
