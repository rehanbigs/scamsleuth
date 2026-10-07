import numpy as np

from scamsleuth.features.lexical import FEATURE_NAMES, lexical_features


def _row(text: str) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, lexical_features([text])[0], strict=True))


def test_shape_and_names() -> None:
    features = lexical_features(["a", "b", "c"])
    assert features.shape == (3, len(FEATURE_NAMES))
    assert lexical_features([]).shape == (0, len(FEATURE_NAMES))


def test_scam_cues_fire_on_scam_and_not_on_chat() -> None:
    scam = _row("URGENT! Dear customer, your bank account is blocked. Verify at http://x.top now")
    chat = _row("Running late, see you at the station")

    for cue in ("cue_urgency", "cue_authority", "cue_action", "cue_greeting", "n_url"):
        assert scam[cue] >= 1, cue
        assert chat[cue] == 0, cue
    assert scam["has_contact"] == 1 and chat["has_contact"] == 0


def test_written_and_anonymised_links_count_the_same() -> None:
    written = _row("Pay the fee at http://parcel.top/pay today")
    anonymised = _row("Pay the fee at xxurl today")
    assert written["n_url"] == anonymised["n_url"] == 1


def test_short_codes_currency_and_case() -> None:
    row = _row("WIN £500 CASH! Txt WIN to 87066")
    assert row["n_shortcode"] == 1
    assert row["n_currency"] == 1
    assert row["upper_ratio"] == 11 / 15  # placeholders are not counted as letters
    assert np.isfinite(list(_row("!!!").values())).all()  # no letters: no division by zero
