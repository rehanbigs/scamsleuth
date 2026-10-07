"""Hand-crafted features: the red flags a person would look for.

Features are computed on masked text (see :func:`~scamsleuth.features.entities.mask_entities`),
so a link counts the same whether it was written out or anonymised as ``xxurl`` by a data
source. Digit counts are deliberately excluded: one source removed numbers during
anonymisation, so digit features would identify the source rather than the scam.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

import numpy as np
from numpy.typing import NDArray

from scamsleuth.features.entities import mask_entities

_CUES: Final = {
    "urgency": r"urgent|immediately|asap|now|today|expire[sd]?|suspend(?:ed)?|block(?:ed)?"
    r"|within \d+|final (?:notice|attempt|reminder)|last chance|act fast|limited time",
    "reward": r"win(?:ner)?|won|prize|free|cash|reward|gift|bonus|claim|congratulations"
    r"|selected|refund|voucher|lottery",
    "authority": r"bank|account|gov|government|tax|police|irs|hmrc|customs|kyc|pan|court"
    r"|ministry|official|security|department",
    "action": r"click|tap|visit|call|reply|update|confirm|verify|log ?in|sign ?in|pay"
    r"|download|activate",
    "delivery": r"parcel|package|deliver(?:y|ed)?|courier|shipment|tracking|post(?:al)?",
    "greeting": r"dear (?:customer|user|client|member|sir|madam)|hello dear|hi mum|hi mom|hey mum",
}
_CUE_PATTERNS: Final = {name: re.compile(rf"\b(?:{p})\b") for name, p in _CUES.items()}
_TOKENS: Final = ("xxurl", "xxphone", "xxemail", "xxshortcode")
_CURRENCY = re.compile(r"[£$€₹₦]|\b(?:rs|inr|usd|gbp|eur|aud|rm)\b\.?")
_WORD = re.compile(r"\b\w+\b")
_PLACEHOLDER = re.compile(r"xx(?:url|phone|email|shortcode|date)")

FEATURE_NAMES: Final = (
    "n_chars",
    "n_words",
    "upper_ratio",
    "n_exclaim",
    "n_question",
    "n_currency",
    *(f"n_{token[2:]}" for token in _TOKENS),
    "has_contact",
    *(f"cue_{name}" for name in _CUES),
)


def _features(text: str) -> list[float]:
    masked = mask_entities(text)
    lower = masked.lower()
    letters = [c for c in _PLACEHOLDER.sub("", masked) if c.isalpha()]  # tokens are not text
    token_counts = [float(lower.count(token)) for token in _TOKENS]
    return [
        float(len(masked)),
        float(len(_WORD.findall(lower))),
        sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        float(masked.count("!")),
        float(masked.count("?")),
        float(len(_CURRENCY.findall(lower))),
        *token_counts,
        float(any(token_counts)),
        *(float(len(pattern.findall(lower))) for pattern in _CUE_PATTERNS.values()),
    ]


def lexical_features(texts: Iterable[str]) -> NDArray[np.float64]:
    """One row of :data:`FEATURE_NAMES` per message."""
    return np.array([_features(text) for text in texts], dtype=np.float64).reshape(
        -1, len(FEATURE_NAMES)
    )
